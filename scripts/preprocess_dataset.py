import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

def normalize_to_zero_to_one(tensor): #min-max normalization to [0, 1]
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val - min_val > 1e-8:
        tensor = (tensor - min_val) / (max_val - min_val)
    else:
        tensor = torch.zeros_like(tensor)
        
    return tensor

def process_file(file_path, save_path, target_size=256):
    path = Path(file_path)
    
    if path.suffix == '.npy':
        arr = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(arr)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
    else:
        img = Image.open(path).convert('L')
        tensor = TF.to_tensor(img)
        
    tensor = TF.resize(tensor, [target_size, target_size], 
                       interpolation=TF.InterpolationMode.BILINEAR)
                       
    tensor = normalize_to_zero_to_one(tensor)
    
    np.save(save_path, tensor.numpy())

def run_preprocessing(raw_dir, out_dir):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    
    for split in ['train', 'val', 'test']:
        for modality in ['MRI', 'CT']:
            in_path = raw_dir / split / modality
            out_path = out_dir / split / modality
            
            if not in_path.exists():
                continue
                
            out_path.mkdir(parents=True, exist_ok=True)
            files = [f for f in in_path.iterdir() if f.is_file()]
            
            print(f"Processing {len(files)} files for {split}/{modality}...")
            for f in tqdm(files):
                save_name = out_path / f"{f.stem}.npy"
                process_file(f, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for ESCAN")
    parser.add_argument('--raw_dir', required=True, help="Path to raw dataset")
    parser.add_argument('--out_dir', required=True, help="Path to save processed .npy files")
    args = parser.parse_args()
    
    run_preprocessing(args.raw_dir, args.out_dir)
    print("Preprocessing complete! Data is now strictly 256x256 and bounded to [0, 1].")