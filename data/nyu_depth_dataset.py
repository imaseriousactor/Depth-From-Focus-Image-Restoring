import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NYUDepthDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
        
    def __len__(self):
        return len(self.h5_files)
        
    def __getitem__(self, idx):
        h5_path = os.path.join(self.data_dir, self.h5_files[idx])
        
        with h5py.File(h5_path, 'r') as f:
            image = f['rgb'][:]
            depth = f['depth'][:]
            
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            
        image = image.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)
        
        if self.transform:
            sample = {'image': image, 'depth': depth}
            sample = self.transform(sample)
            image = sample['image']
            depth = sample['depth']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            depth = torch.from_numpy(depth).unsqueeze(0)
            
        return {
            'image': image,
            'depth': depth,
            'filename': self.h5_files[idx]
        }

def test_dataloader(data_dir="data/raw/val/official", batch_size=2):
    print("Testing NYUDepthDataset...")
    dataset = NYUDepthDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Image shape:", batch['image'].shape)
        print("Depth shape:", batch['depth'].shape)
        print("Filenames:", batch['filename'])
        break
        
    print("DataLoader test passed successfully.")

if __name__ == "__main__":
    test_dataloader()