import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data.nyu_depth_dataset import NYUDepthDataset
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.vit_hybrid import ViTHybrid

def load_model(model_name, checkpoint_path, device):
    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=1)
    elif model_name == "attention_unet":
        model = AttentionUNet(in_channels=3, out_channels=1)
    elif model_name == "vit_hybrid":
        model = ViTHybrid(in_channels=3, out_channels=1, embed_dim=64)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f" Loaded {model_name} from {checkpoint_path}")
    else:
        print(f" Checkpoint not found for {model_name} at {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model

def get_prediction(model, image_tensor, device, target_size=(480, 640)):
    with torch.no_grad():
        # Resize input to model's expected size if needed (simplified here)
        img = image_tensor.unsqueeze(0).to(device)
        pred = model(img).squeeze(0).squeeze(0).cpu()
        # Resize prediction back to original image size for visualization
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=True).squeeze()
    return pred

def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/raw/val/official"
    
    # Load datasets and models
    dataset = NYUDepthDataset(data_dir)
    
    models = {
        "UNet": load_model("unet", "checkpoints/unet_best.pth", device),
        "Attention UNet": load_model("attention_unet", "checkpoints/attention_unet_best.pth", device),
        "ViT Hybrid": load_model("vit_hybrid", "checkpoints/vit_hybrid_best.pth", device)
    }
    
    # Pick 3 random images to visualize
    indices = [0, 10, 50] 
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image']
        true_depth = sample['depth'].squeeze(0)
        
        # Original Image
        axes[row, 0].imshow(image.permute(1, 2, 0))
        axes[row, 0].set_title(f"Original #{idx}")
        axes[row, 0].axis('off')
        
        # Ground Truth
        axes[row, 1].imshow(true_depth, cmap='magma')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')
        
        # Predictions
        for col, (name, model) in enumerate(models.items(), start=2):
            if model is not None:
                pred = get_prediction(model, image, device, target_size=(image.shape[1], image.shape[2]))
                pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                axes[row, col].imshow(pred_norm, cmap='magma')
                axes[row, col].set_title(name)
            else:
                axes[row, col].text(0.5, 0.5, "No weights", ha='center', va='center')
            axes[row, col].axis('off')
            
    plt.tight_layout()
    plt.savefig("notebooks/outputs/model_comparison.png", dpi=150)
    plt.show()
    print(" Сравнение сохранено в notebooks/outputs/model_comparison.png")

if __name__ == "__main__":
    compare_models()