import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.act = nn.LeakyReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # 1x1 Conv followed by 3x3 depth-wise conv for Q, K, V generation
        self.qkv_1x1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        # 1x1 Conv to refine attended features
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Generate Q, K, V features
        qkv = self.qkv_dwconv(self.qkv_1x1(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape to R^{C x HW} for channel-wise attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 Normalization
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Compute channel-wise attention map
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Apply attention map to value features
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # Refine attended features
        return self.project_out(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, res=True):
        super().__init__()
        hidden = hidden_channels or out_channels
        self.conv1 = ConvLayer(in_channels, hidden)
        self.conv2 = ConvLayer(hidden, out_channels)
        
        self.res = res
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + self.shortcut(x) if self.res else out

class ESABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, res=True):
        super().__init__()
        self.attn = EfficientSelfAttention(in_channels, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(in_channels, out_channels, res=res)
    
    def forward(self, x):
        # Residual connection around the attention module
        x_attn = x + self.attn(x)
        # Lightweight feed-forward network
        return self.ffn(x_attn)

class MACF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # Using GroupNorm (groups=4) as the normalization layer for small batch stability
        norm_groups = 4 
        
        # Channel-wise attention components
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.ca_conv1 = nn.Conv2d(in_channels * 4, in_channels * 2, 1)
        self.ca_norm1 = nn.GroupNorm(norm_groups, in_channels * 2)
        self.ca_conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.ca_norm2 = nn.GroupNorm(norm_groups, in_channels)

        # Spatial-wise attention components
        self.sa_conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.sa_norm1 = nn.GroupNorm(norm_groups, in_channels)
        self.sa_conv2 = nn.Conv2d(in_channels, in_channels, 1)
        self.sa_norm2 = nn.GroupNorm(norm_groups, in_channels)

        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()

    def forward(self, f_mri, f_ct):
        # Equation: F_cat = Concat(F_MR, F_CT)
        f_cat = torch.cat([f_mri, f_ct], dim=1)
        
        # 1. Channel-wise Attention Pathway
        # Generate pooled descriptors and concatenate
        pooled_desc = torch.cat([self.gap(f_cat), self.gmp(f_cat)], dim=1)
        
        # Successive 1x1 conv -> norm -> SiLU -> 1x1 conv -> norm
        a_ch = self.ca_norm1(self.ca_conv1(pooled_desc))
        a_ch = self.silu(a_ch)
        a_ch = self.ca_norm2(self.ca_conv2(a_ch))
        
        # 2. Spatial-wise Attention Pathway
        # Applied directly to F_cat using stacked 1x1 conv -> norm -> SiLU
        a_sp = self.sa_norm1(self.sa_conv1(f_cat))
        a_sp = self.silu(a_sp)
        a_sp = self.sa_norm2(self.sa_conv2(a_sp))
        
        # 3. Adaptive Fusion Weight (Alpha_l)
        # Equation: alpha_l = sigma(A_ch + A_sp)
        alpha_l = self.sigmoid(a_ch + a_sp)
        
        # 4. Fused Feature Representation
        # Equation: F_F = alpha_l * F_MR + (1 - alpha_l) * F_CT
        f_fused = (f_mri * alpha_l) + (f_ct * (1 - alpha_l))
        
        return f_fused