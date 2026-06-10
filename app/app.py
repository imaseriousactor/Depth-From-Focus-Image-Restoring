import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.vit_hybrid import ViTHybrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загружаем все три модели
models = {}
model_configs = {
    "UNet": {"class": UNet, "path": "checkpoints/unet_best.pth", "size": (480, 640)},
    "Attention UNet": {"class": AttentionUNet, "path": "checkpoints/attention_unet_best.pth", "size": (480, 640)},
    "ViT Hybrid": {"class": ViTHybrid, "path": "checkpoints/vit_hybrid_best.pth", "size": (128, 192)}
}

for name, config in model_configs.items():
    try:
        model = config["class"](in_channels=3, out_channels=1)
        model.load_state_dict(torch.load(config["path"], map_location=device))
        model.to(device)
        model.eval()
        models[name] = {"model": model, "size": config["size"]}
        print(f" Loaded {name}")
    except Exception as e:
        print(f" Could not load {name}: {e}")

def predict_depth(image, model_name):
    if model_name not in models:
        return None, f"Model {model_name} not loaded"
    
    model_info = models[model_name]
    model = model_info["model"]
    target_size = model_info["size"]
    
    # Препроцессинг
    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Resize до размера, на котором обучалась модель
    img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=True)
    
    # Предсказание
    with torch.no_grad():
        pred = model(img_resized.to(device)).squeeze(0).squeeze(0).cpu()
    
    # Resize обратно к оригинальному размеру
    pred_resized = F.interpolate(
        pred.unsqueeze(0).unsqueeze(0), 
        size=(image.size[1], image.size[0]), 
        mode='bilinear', 
        align_corners=True
    ).squeeze()
    
    # Нормализация для визуализации
    pred_norm = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
    
    # Создаём красивую картинку с colormap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(pred_norm.numpy(), cmap='magma')
    ax.set_title(f"Depth Prediction ({model_name})")
    ax.axis('off')
    plt.tight_layout()
    
    return fig, f"Prediction generated with {model_name}"

# Создаём интерфейс
demo = gr.Interface(
    fn=predict_depth,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=list(models.keys()), value="UNet", label="Select Model")
    ],
    outputs=[
        gr.Plot(label="Depth Map"),
        gr.Text(label="Status")
    ],
    title=" Depth from Focus - Image Restoration",
    description="""
    Upload an image to predict depth using trained neural networks.
    
    **Available models:**
    - **UNet**: Best metrics (RMSE: 1.005), but may have artifacts
    - **Attention UNet**: Uses attention gates for better feature selection
    - **ViT Hybrid**: Vision Transformer + CNN decoder, better global context
    
    **Key insight:** On small datasets (645 images), CNN architectures (UNet) 
    outperform Transformer-based models (ViT), which require millions of images.
    """,
    examples=[
        ["examples/sample1.jpg", "UNet"],
        ["examples/sample2.jpg", "ViT Hybrid"],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)