import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DIODEDepthDataset(Dataset):
    def __init__(self, hdf5_path, img_size=(256, 384)):
        self.hdf5_path = hdf5_path
        self.resize = T.Resize(img_size)
        
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = len(f.keys())
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[f'sample_{idx:06d}']
            rgb = group['rgb'][:]
            depth = group['depth'][:]
        
        # Нормализация
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)
        
        # Ресайз
        rgb = self.resize(rgb)
        depth = self.resize(depth)
        
        return {'image': rgb, 'depth': depth}