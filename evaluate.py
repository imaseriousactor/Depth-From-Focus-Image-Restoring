import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T

from data.nyu_depth_dataset import NYUDepthDataset
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.vit_hybrid import ViTHybrid

MODEL_CONFIGS = {
    "unet": {"class": UNet, "path": "checkpoints/unet_best.pth", "img_size": (256, 384)},
    "attention_unet": {"class": AttentionUNet, "path": "checkpoints/attention_unet_best.pth", "img_size": (256, 384)},
    "vit_hybrid": {"class": ViTHybrid, "path": "checkpoints/vit_hybrid_best.pth", "img_size": (128, 192)}
}

def load_model(name, device):
    cfg = MODEL_CONFIGS[name]
    model = cfg["class"](in_channels=3, out_channels=1)
    if os.path.exists(cfg["path"]):
        model.load_state_dict(torch.load(cfg["path"], map_location=device))
     
    else:
        print(f" No weights for {name}")
    model.to(device)
    model.eval()
    return model, cfg["img_size"]

def compute_metrics(pred, target):
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    mae = torch.mean(torch.abs(pred - target))
    
    mask = target > 0
    ratio = torch.max(pred[mask] / target[mask], target[mask] / pred[mask])
    delta1 = (ratio < 1.25).float().mean()
    
    return rmse.item(), mae.item(), delta1.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = "data/raw/val/official"
    
    models = {}
    for name in MODEL_CONFIGS:
        model, img_size = load_model(name, device)
        models[name] = (model, img_size)
    
    print("\n" + "="*60)
    print("МЕТРИКИ")
    print("="*60)
    
    results = {}
    val_dataset = NYUDepthDataset(val_dir, img_size=(256, 384))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    for name, (model, img_size) in models.items():
        print(f"\n{name.upper()}")
        total_rmse = total_mae = total_delta1 = 0
        n = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                
                if img_size != (256, 384):
                    images = torch.nn.functional.interpolate(images, size=img_size, mode='bilinear')
                
                outputs = model(images)
                outputs = torch.nn.functional.interpolate(outputs, size=depths.shape[2:], mode='bilinear')
                
                rmse, mae, delta1 = compute_metrics(outputs, depths)
                total_rmse += rmse
                total_mae += mae
                total_delta1 += delta1
                n += 1
        
        results[name] = {
            "RMSE": total_rmse / n,
            "MAE": total_mae / n,
            "δ1": total_delta1 / n
        }
        
        print(f"  RMSE: {results[name]['RMSE']:.4f}")
        print(f"  MAE:  {results[name]['MAE']:.4f}")
        print(f"  δ1:   {results[name]['δ1']:.4f}")
    
    # Таблица
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'δ1':>10}")
    print("-" * 52)
    for name, m in results.items():
        print(f"{name:<20} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['δ1']:>10.4f}")

if __name__ == "__main__":
    main()