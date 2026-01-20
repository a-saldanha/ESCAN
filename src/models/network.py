import torch
import torch.nn as nn
from .escan_modules import ESABlock, ConvLayer, MACF

class ESCANEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_heads=[1, 2, 4, 8]):
        super().__init__()
        self.stem = ConvLayer(in_channels, base_channels, act=True)
        
        self.layer1 = ESABlock(base_channels, base_channels, num_heads=num_heads[0])
        
        self.down2 = nn.MaxPool2d(2)
        self.layer2 = ESABlock(base_channels, base_channels*2, num_heads=num_heads[1])
        
        self.down3 = nn.MaxPool2d(2)
        self.layer3 = ESABlock(base_channels*2, base_channels*4, num_heads=num_heads[2])
        
        self.down4 = nn.MaxPool2d(2)
        self.layer4 = ESABlock(base_channels*4, base_channels*8, num_heads=num_heads[3])

    def forward(self, x):
        f1 = self.layer1(self.stem(x))      # [B, 32, H, W]
        f2 = self.layer2(self.down2(f1))    # [B, 64, H/2, W/2]
        f3 = self.layer3(self.down3(f2))    # [B, 128, H/4, W/4]
        f4 = self.layer4(self.down4(f3))    # [B, 256, H/8, W/8]
        return [f1, f2, f3, f4]

class ESCANSharedDecoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=32, num_heads=[1, 2, 4, 8]):
        super().__init__()
        c = base_channels
        
        self.reduce1 = nn.Conv2d(c*8, c*4, 1) 
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block1 = ESABlock(c*4, c*4, num_heads=num_heads[2])
        
        self.reduce2 = nn.Conv2d(c*4, c*2, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block2 = ESABlock(c*2, c*2, num_heads=num_heads[1])
        
        self.reduce3 = nn.Conv2d(c*2, c, 1)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block3 = ESABlock(c, c, num_heads=num_heads[0])
        
        self.final = nn.Sequential(
            ConvLayer(c, c, act=True),
            nn.Conv2d(c, out_channels, 3, 1, 1)
        )

    def forward(self, bottleneck, skips):
        s1, s2, s3 = skips
        
        x = self.reduce1(bottleneck)
        x = self.up1(x)
        x = self.block1(x + s3)
        
        x = self.reduce2(x) 
        x = self.up2(x)     
        x = self.block2(x + s2)
        
        x = self.reduce3(x) 
        x = self.up3(x)     
        x = self.block3(x + s1)
        
        return self.final(x)

class ESCAN(nn.Module):
    def __init__(self, n_channels=32, init_type='Kaiming'):
        super().__init__()
        heads = [1, 2, 4, 8]
        
        self.enc_mri = ESCANEncoder(base_channels=n_channels, num_heads=heads)
        self.enc_ct = ESCANEncoder(base_channels=n_channels, num_heads=heads)
        
        self.macf1 = MACF(n_channels)
        self.macf2 = MACF(n_channels * 2)
        self.macf3 = MACF(n_channels * 4)
        self.macf4 = MACF(n_channels * 8)
        
        self.decoder = ESCANSharedDecoder(base_channels=n_channels, num_heads=heads)

        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'Xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Kaiming Uniform (He Init) tailored for LeakyReLU
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mri, ct):
        # Step 1: Modality-specific independent encoding
        f_mri = self.enc_mri(mri) 
        f_ct = self.enc_ct(ct)
        
        # Step 2: Adaptive multi-level fusion via MACF
        f_fused = []
        f_fused.append(self.macf1(f_mri[0], f_ct[0]))
        f_fused.append(self.macf2(f_mri[1], f_ct[1]))
        f_fused.append(self.macf3(f_mri[2], f_ct[2]))
        f_fused.append(self.macf4(f_mri[3], f_ct[3]))
        
        # Step 3: Reconstruction through shared decoder
        # bottleneck = f_fused[3] (256ch), skips = f_fused[0:3]
        return self.decoder(bottleneck=f_fused[3], skips=f_fused[0:3])