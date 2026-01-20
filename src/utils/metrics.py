import torch
import numpy as np
from sewar.full_ref import mse, rmse, psnr, ssim, msssim, uqi, vifp, scc

def calculate_entropy(img_float):
    img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    hist, _ = np.histogram(img_uint8, bins=256, range=(0, 255))
    
    prob = hist / np.sum(hist)
    
    prob = prob[prob > 0]
    
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def calculate_sd(img_float):
    return np.std(img_float)

def compute_full_metrics(mri_tensor, ct_tensor, fused_tensor):
    """
    Computes metrics for ESCAN.
    Inputs are inherently [0, 1].
    """
    def to_float_np(t):
        arr = t.squeeze().cpu().detach().numpy()
        arr = np.clip(arr, 0, 1)
        return arr.astype(np.float32)

    mri = to_float_np(mri_tensor)
    ct = to_float_np(ct_tensor)
    fused = to_float_np(fused_tensor)

    results = {}

    results['EN'] = calculate_entropy(fused)
    results['SD'] = calculate_sd(fused)

    metric_configs = {
        'SSIM':   (ssim,   {'MAX': 1.0}),
        'MSSSIM': (msssim, {'MAX': 1.0}),
        'VIF':    (vifp,   {'sigma_nsq': 2}), 
    }

    for name, (func, kwargs) in metric_configs.items():
        try:
            val_mri = func(mri, fused, **kwargs)
            val_ct = func(ct, fused, **kwargs)
            
            if isinstance(val_mri, tuple): val_mri = val_mri[0]
            if isinstance(val_ct, tuple): val_ct = val_ct[0]
            
            val_mri = np.real(val_mri)
            val_ct = np.real(val_ct)
            
            avg_score = (val_mri + val_ct) / 2.0
            results[name] = float(avg_score)
        except Exception as e:
            results[name] = 0.0

    return results