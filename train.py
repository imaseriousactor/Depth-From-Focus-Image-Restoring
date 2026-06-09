import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from data.nyu_depth_dataset import NYUDepthDataset
from models.unet import UNet
from models.attention_unet import AttentionUNet


def get_model(model_name):
    if model_name == "unet":
        return UNet(in_channels=3, out_channels=1)
    elif model_name == "attention_unet":
        return AttentionUNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(pred, target):
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    mae = torch.mean(torch.abs(pred - target))
    
    mask = target > 0
    if mask.sum() > 0:
        ratio = torch.max(pred[mask] / target[mask], target[mask] / pred[mask])
        delta1 = (ratio < 1.25).float().mean()
    else:
        delta1 = torch.tensor(0.0)
    
    return rmse.item(), mae.item(), delta1.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_delta1 = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, depths)
            
            rmse, mae, delta1 = compute_metrics(outputs, depths)
            
            total_loss += loss.item()
            total_rmse += rmse
            total_mae += mae
            total_delta1 += delta1
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'rmse': total_rmse / n,
        'mae': total_mae / n,
        'delta1': total_delta1 / n
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "attention_unet"
    epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    
    train_dir = "data/raw/train-000000/official"
    val_dir = "data/raw/val/official"
    
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory not found at {train_dir}")
        print("Using validation set for both training and validation (demo mode)")
        train_dir = val_dir
    
    train_dataset = NYUDepthDataset(train_dir)
    val_dataset = NYUDepthDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = get_model(model_name).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    writer = SummaryWriter(f"runs/{model_name}")
    
    best_val_loss = float('inf')
    
    print(f"Training {model_name} for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/delta1', val_metrics['delta1'], epoch)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val δ1:     {val_metrics['delta1']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
            print(f"   Saved best model")
        
        print()
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()