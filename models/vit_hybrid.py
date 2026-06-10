import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
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
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
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
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1)

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

def test_vit_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTHybrid(in_channels=3, out_channels=1, embed_dim=64).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {total_params:,}")
    
    dummy_input = torch.randn(2, 3, 256, 384).to(device)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1, 256, 384), "Output shape mismatch!"
    print("ViT Hybrid test passed successfully.")

if __name__ == "__main__":
    test_vit_hybrid()