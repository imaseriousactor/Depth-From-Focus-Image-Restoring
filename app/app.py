import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(nn.ModuleList([DoubleConv(ch, f), nn.MaxPool2d(2)]))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.decoders.append(nn.ModuleList([
                nn.ConvTranspose2d(f * 2, f, 2, stride=2),
                AttentionBlock(f, f, f // 2),
                DoubleConv(f * 2, f)
            ]))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc[0](x)
            skips.append(x)
            x = enc[1](x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for (up, att, conv), skip in zip(self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([att(x, skip), skip], dim=1)
            x = conv(x)
        return self.final(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim), nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        H_out, W_out = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, -1, H_out, W_out)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, embed_dim=64, out_channels=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
                                    nn.BatchNorm2d(embed_dim // 2), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, padding=1),
                                    nn.BatchNorm2d(embed_dim // 4), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(embed_dim // 4, out_channels, 1)
    def forward(self, x):
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        return self.final(x)

class ViTHybrid(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, embed_dim=64):
        super().__init__()
        self.encoder = ViTEncoder(in_channels, embed_dim)
        self.decoder = CNNDecoder(embed_dim, out_channels)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = {}
model_configs = {
    "Attention UNet": {"class": AttentionUNet, "path": "attention_unet_best.pth", "size": (256, 384), "kwargs": {}},
    "ViT Hybrid": {"class": ViTHybrid, "path": "vit_hybrid_best.pth", "size": (128, 192), "kwargs": {"embed_dim": 32}}
}

for name, config in model_configs.items():
    try:
        model = config["class"](in_channels=3, out_channels=1, **config["kwargs"])
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
    img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    # Предсказание
    with torch.no_grad():
        pred = model(img_resized.to(device)).squeeze(0).squeeze(0).cpu()
    
    # Нормализация для визуализации (0-1)
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    # Создаём красивую картинку с colormap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(pred_norm.numpy(), cmap='magma')
    ax.set_title(f"Depth Prediction ({model_name})", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    return fig, f"Prediction generated with {model_name}"


demo = gr.Interface(
    fn=predict_depth,
    inputs=[
        gr.Image(type="pil", label="Upload RGB Image"),
        gr.Dropdown(choices=list(models.keys()), value="Attention UNet", label="Select Model")
    ],
    outputs=[
        gr.Plot(label="Depth Map"),
        gr.Text(label="Status")
    ],
    title=" Depth from Focus - Image Restoration",
    description="""
    Upload an image to predict depth using trained neural networks.
    
    **Available models:**
    - **Attention UNet**: CNN with attention gates (RMSE: 1.66m on NYU)
    - **ViT Hybrid**: Vision Transformer + CNN decoder (RMSE: 1.41m on NYU)
    
    **Key insight:** On small datasets (3448 images), Transformer-based ViT Hybrid 
    outperforms CNN-based Attention UNet, contrary to our initial hypothesis.
    
    Trained on NYU Depth V2 (indoor) + DIODE (indoor + outdoor) datasets.
    """
)

if __name__ == "__main__":
    demo.launch()