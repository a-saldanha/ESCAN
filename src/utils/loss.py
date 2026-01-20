import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

class SobelXY(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        
        kernel_y = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        
        k_x = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        k_y = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('kernel_x', k_x)
        self.register_buffer('kernel_y', k_y)

    def forward(self, x):
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        # Gradient magnitude: ∇I = |I * Kx| + |I * Ky|
        return torch.abs(grad_x) + torch.abs(grad_y)

class CompositeFusionLoss(nn.Module):
    """
    End-to-end unsupervised composite loss function for ESCAN.
    Formula: L_total = λ2 * L_int + λ1 * L_ssim + λ3 * L_grad
    """
    def __init__(self, lambda_1=1.0, lambda_2=5.0, lambda_3=2.0):
        super().__init__()
        self.lambda_1 = lambda_1  
        self.lambda_2 = lambda_2  
        self.lambda_3 = lambda_3  
        self.sobel = SobelXY()

    def forward(self, i_mr, i_ct, i_f):
        """
        i_mr: MRI input tensor [0, 1]
        i_ct: CT input tensor [0, 1]
        i_f: Fused output tensor [0, 1]
        """
        # 1. Intensity Loss
        max_intensity = torch.max(i_mr, i_ct)
        l_int = F.mse_loss(i_f, max_intensity)
        
        # 2. Structural Similarity Loss
        ssim_mr = ms_ssim(i_f, i_mr, data_range=1.0, size_average=True)
        ssim_ct = ms_ssim(i_f, i_ct, data_range=1.0, size_average=True)
        l_ssim = 0.5 * (1 - ssim_mr) + 0.5 * (1 - ssim_ct)
        
        # 3. Gradient Loss
        grad_mr = self.sobel(i_mr)
        grad_ct = self.sobel(i_ct)
        grad_f = self.sobel(i_f)
        
        max_grad = torch.max(grad_mr, grad_ct)
        l_grad = F.l1_loss(grad_f, max_grad)
        
        l_total = (self.lambda_2 * l_int) + \
                  (self.lambda_1 * l_ssim) + \
                  (self.lambda_3 * l_grad)
                     
        return l_total, l_int, l_ssim, l_grad