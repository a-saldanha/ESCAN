import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from src.models.network import ESCAN
from src.utils.metrics import compute_full_metrics

def load_single_file(path, resize_dim=256):
    path = Path(path)
    if path.suffix == '.npy':
        arr = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(arr)
    else:
        img = Image.open(path).convert('L')
        tensor = TF.to_tensor(img) 

    if resize_dim:
        tensor = TF.resize(tensor, [resize_dim, resize_dim], 
                           interpolation=TF.InterpolationMode.BILINEAR)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    min_val, max_val = tensor.min(), tensor.max()
    if max_val - min_val > 1e-8:
        tensor = (tensor - min_val) / (max_val - min_val) 
    else:
        tensor = torch.zeros_like(tensor)

    return tensor

def save_fused_image(tensor, path):
    arr = tensor.squeeze().cpu().detach().numpy()
    
    arr = np.clip(arr, 0, 1) * 255
    
    Image.fromarray(arr.astype(np.uint8)).save(path)

def find_pairs(mri_path, ct_path):
    mri_p = Path(mri_path)
    ct_p = Path(ct_path)

    if mri_p.is_file() and ct_p.is_file():
        return [(mri_p, ct_p)]

    if mri_p.is_dir() and ct_p.is_dir():
        pairs = []
        valid_exts = {'.npy', '.png', '.jpg', '.jpeg'}
        
        mri_files = sorted([f for f in mri_p.iterdir() if f.suffix.lower() in valid_exts])
        
        for m_file in mri_files:
            candidates = [
                m_file.name,
                m_file.name.replace('_mri', '_ct'),
                m_file.name.replace('_mr', '_ct')
            ]
            for cand in candidates:
                c_file = ct_p / cand
                if c_file.exists():
                    pairs.append((m_file, c_file))
                    break
        return pairs
    return []

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ESCAN Model: {args.checkpoint}")
    model = ESCAN(n_channels=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    pairs = find_pairs(args.mri, args.ct)
    if not pairs:
        print("No matching pairs found!")
        return
    
    print(f"Found {len(pairs)} pairs to process.")
    results_list = []

    for m_path, c_path in tqdm(pairs):
        try:
            mri = load_single_file(m_path, args.resize).to(device)
            ct = load_single_file(c_path, args.resize).to(device)
            
            with torch.no_grad():
                fused = model(mri, ct)
            
            fname_stem = m_path.stem 
            save_name_stem = fname_stem.replace('_mri', '_fused').replace('_mr', '_fused')
            if '_fused' not in save_name_stem:
                 save_name_stem = f"fused_{save_name_stem}"

            save_name = f"{save_name_stem}.png"
            save_path = output_dir / save_name
            
            save_fused_image(fused, save_path)
            
            metrics = compute_full_metrics(mri, ct, fused)
            metrics['Filename'] = save_name 
            results_list.append(metrics)
            
        except Exception as e:
            print(f"Failed to process {m_path.name}: {e}")

    if results_list:
        df = pd.DataFrame(results_list)
        target_cols = ['Filename', 'EN', 'SD', 'SSIM', 'MSSSIM', 'VIF']
        final_cols = [c for c in target_cols if c in df.columns]
        df = df[final_cols]
        
        avg_row = df.mean(numeric_only=True)
        avg_row['Filename'] = 'AVERAGE'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        
        csv_path = output_dir / "escan_inference_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*40)
        print("ESCAN INFERENCE SUMMARY (Averages)")
        print("="*40)
        print(df.iloc[-1]) 
        print(f"\nFull results saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESCAN Inference Script")
    parser.add_argument('--mri', required=True, help="MRI File OR Directory")
    parser.add_argument('--ct', required=True, help="CT File OR Directory")
    parser.add_argument('--checkpoint', required=True, help="Path to .pth model")
    parser.add_argument('--output_dir', default='./escan_results', help="Where to save outputs")
    parser.add_argument('--resize', type=int, default=256, help="Force resize input (default: 256)")
    
    args = parser.parse_args()
    run_inference(args)