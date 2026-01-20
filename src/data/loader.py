import os
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset, DataLoader

class ESCANDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        self.root_dir = Path(root_dir) / split
        self.mri_dir = self.root_dir / 'MRI'
        self.ct_dir = self.root_dir / 'CT'
        self.augment = augment
        
        self.pairs = self._index_pairs()

    def _index_pairs(self):
        """Robust pairing logic matching MRI to CT files."""
        pairs = []
        if not self.mri_dir.exists(): 
            return []
        
        mri_files = sorted([f for f in os.listdir(self.mri_dir) if f.endswith('.npy')])
        
        for m_name in mri_files:
            if '_mri_' in m_name:
                c_name = m_name.replace('_mri_', '_ct_')
            elif '_mr_' in m_name:
                c_name = m_name.replace('_mr_', '_ct_')
            elif '_mr.' in m_name:
                c_name = m_name.replace('_mr.', '_ct.')
            else:
                c_name = m_name

            if (self.ct_dir / c_name).exists():
                pairs.append((m_name, c_name))
        
        return pairs

    def _load_tensor(self, path):
        """Loads a preprocessed .npy file directly to a [1, H, W] tensor."""
        arr = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(arr)
        
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            
        return tensor

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        m_name, c_name = self.pairs[idx]
        
        mri = self._load_tensor(self.mri_dir / m_name)
        ct = self._load_tensor(self.ct_dir / c_name)
        
        if self.augment:
            if random.random() > 0.5:
                mri = TF.hflip(mri)
                ct = TF.hflip(ct)
                
            if random.random() > 0.5:
                mri = TF.vflip(mri)
                ct = TF.vflip(ct)
                
            angle = random.uniform(-10, 10)
            mri = TF.rotate(mri, angle)
            ct = TF.rotate(ct, angle)

        return {'mri': mri, 'ct': ct, 'filename': m_name}

def get_dataloaders(config):
    """Factory to create loaders based on the config dict."""
    train_ds = ESCANDataset(
        config['data_root'], split='train', augment=True
    )
    val_ds = ESCANDataset(
        config['data_root'], split='val', augment=False
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader