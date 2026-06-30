import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

# ========== МОДЕЛИ (из Kaggle notebook) ==========
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(nn.Sequential(DoubleConv(ch, f), nn.MaxPool2d(2)))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.decoders.append(nn.ModuleList([
                nn.ConvTranspose2d(f * 2, f, 2, stride=2),
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
        for (up, conv), skip in zip(self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return self.final(x)

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


# ========== DATASETS ==========
class NYUDepthDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 384)):
        import glob
        self.data_dir = data_dir
        self.h5_files = sorted(glob.glob(str(data_dir) + "/**/*.h5", recursive=True))
        self.resize = T.Resize(img_size)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, 'r') as f:
            image = f['rgb'][:]
            depth = f['depth'][:]
        
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        
        image = self.resize(image)
        depth = self.resize(depth)
        
        return {'image': image, 'depth': depth}


class DIODEDepthDataset(Dataset):
    def __init__(self, hdf5_path, img_size=(256, 384), start_idx=0, end_idx=None):
        self.hdf5_path = hdf5_path
        self.resize = T.Resize(img_size)
        
        with h5py.File(hdf5_path, 'r') as f:
            total = len(f.keys())
            self.start_idx = start_idx
            self.end_idx = end_idx if end_idx is not None else total
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[f'sample_{real_idx:06d}']
            rgb = group['rgb'][:]
            depth = group['depth'][:]
        
        rgb = rgb.astype(np.float32) / 255.0
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        
        depth = depth.astype(np.float32)
        if depth.ndim == 4:
            depth = depth.squeeze()
        elif depth.ndim == 3:
            depth = depth[0]
        
        depth = torch.from_numpy(depth).unsqueeze(0)
        
        rgb = self.resize(rgb)
        depth = self.resize(depth)
        
        return {'image': rgb, 'depth': depth}


# ========== НАСТРОЙКИ ==========
MODEL_CONFIGS = {
    "unet": {"class": UNet, "path": "checkpoints/unet_best.pth", "img_size": (256, 384), "embed_dim": None},
    "attention_unet": {"class": AttentionUNet, "path": "checkpoints/attention_unet_best.pth", "img_size": (256, 384), "embed_dim": None},
    "vit_hybrid": {"class": ViTHybrid, "path": "checkpoints/vit_hybrid_best.pth", "img_size": (128, 192), "embed_dim": 32}
}

# Константы нормализации из обучения
NYU_DEPTH_SCALE = 10.0
DIODE_DEPTH_SCALE = 50.0


def load_model(name, device):
    cfg = MODEL_CONFIGS[name]
    if name == "vit_hybrid":
        model = cfg["class"](in_channels=3, out_channels=1, embed_dim=cfg["embed_dim"])
    else:
        model = cfg["class"](in_channels=3, out_channels=1)
    
    if os.path.exists(cfg["path"]):
        model.load_state_dict(torch.load(cfg["path"], map_location=device))
        print(f" Loaded {name}")
    else:
        print(f"⚠️ No weights for {name}")
    model.to(device)
    model.eval()
    return model, cfg["img_size"]


def compute_metrics(pred, target):
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    # Маска валидных пикселей: считаем метрики ТОЛЬКО там, где есть
    # реальная разметка глубины (иначе нули неба/дали в DIODE портят RMSE/MAE)
    mask = (target > 0) & torch.isfinite(target) & torch.isfinite(pred)
    if mask.sum() == 0:
        return float('nan'), float('nan'), float('nan')

    pred_v = pred[mask]
    target_v = target[mask]

    rmse = torch.sqrt(torch.mean((pred_v - target_v) ** 2))
    mae = torch.mean(torch.abs(pred_v - target_v))

    # Для δ1 нужны положительные предсказания (выход модели не ограничен)
    pred_pos = torch.clamp(pred_v, min=1e-3)
    ratio = torch.max(pred_pos / target_v, target_v / pred_pos)
    delta1 = (ratio < 1.25).float().mean()

    return rmse.item(), mae.item(), delta1.item()


def evaluate_on_dataset(model, img_size, loader, depth_scale, device):
    """Оценивает модель на конкретном датасете и возвращает метрики в метрах."""
    total_rmse = total_mae = total_delta1 = 0
    n = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            if img_size != depths.shape[2:]:
                images = F.interpolate(images, size=img_size, mode='bilinear', align_corners=False)
            
            outputs = model(images)
            outputs = F.interpolate(outputs, size=depths.shape[2:], mode='bilinear', align_corners=False)
            
            # Нормализуем target так же, как в обучении
            depths_normalized = depths / depth_scale
            
            rmse, mae, delta1 = compute_metrics(outputs, depths_normalized)

            # Пропускаем батчи без валидных пикселей
            if np.isnan(rmse):
                continue

            # Переводим метрики обратно в метры
            total_rmse += rmse * depth_scale
            total_mae += mae * depth_scale
            total_delta1 += delta1
            n += 1
    
    return {
        "RMSE": total_rmse / n,
        "MAE": total_mae / n,
        "δ1": total_delta1 / n
    } if n > 0 else {"RMSE": float('nan'), "MAE": float('nan'), "δ1": float('nan')}


def visualize_predictions(models, dataset, depth_scale, device,
                          n_samples=4, indices=None,
                          out_path="results/qualitative.png",
                          dataset_title="", cmap="magma"):
    """
    Сетка: строки — примеры, столбцы — [RGB | Ground Truth | предсказания моделей].
    Глубина показывается в метрах, общая цветовая шкала на весь датасет.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if indices is None:
        indices = np.linspace(0, len(dataset) - 1, n_samples).astype(int).tolist()
    n_samples = len(indices)
    model_names = list(models.keys())
    ncols = 2 + len(model_names)

    print(f"  Определяем диапазон глубины для {dataset_title}...")
    all_depths = []
    for i in range(min(50, len(dataset))):  # Берём первые 50 примеров
        sample = dataset[i]
        depth = sample['depth'].squeeze(0).cpu().numpy()
        valid = (depth > 0) & np.isfinite(depth)
        if valid.sum() > 0:
            all_depths.extend(depth[valid].flatten())
    
    if len(all_depths) > 0:
        vmin = float(np.percentile(all_depths, 1))
        vmax = float(np.percentile(all_depths, 99))
        print(f"  Диапазон: {vmin:.2f} - {vmax:.2f} м")
    else:
        vmin, vmax = 0.0, 10.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("gray")  # невалидные пиксели — серым

    fig, axes = plt.subplots(n_samples, ncols,
                             figsize=(3.5 * ncols, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    col_titles = ["RGB", "Ground Truth"] + [n.upper() for n in model_names]

    for r, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image']
        depth_gt = sample['depth'].squeeze(0).cpu().numpy()

        gt_mask = (depth_gt > 0) & np.isfinite(depth_gt)
        H, W = depth_gt.shape

        rgb = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title("RGB" if r == 0 else "", fontsize=11)

        gt_show = np.where(gt_mask, depth_gt, np.nan)
        im_gt = axes[r, 1].imshow(gt_show, cmap=colormap, norm=norm)
        axes[r, 1].set_title("Ground Truth" if r == 0 else "", fontsize=11)

        img_batch = image.unsqueeze(0).to(device)
        for c, name in enumerate(model_names):
            model, img_size = models[name]
            inp = F.interpolate(img_batch, size=img_size,
                                mode='bilinear', align_corners=False)
            with torch.no_grad():
                out = model(inp)
            out = F.interpolate(out, size=(H, W),
                                mode='bilinear', align_corners=False)
            pred = out.squeeze().cpu().numpy() * depth_scale  # в метры
            pred = np.clip(pred, 0, None)
            
            pred_show = np.where(gt_mask, pred, np.nan)
            axes[r, c + 2].imshow(pred_show, cmap=colormap, norm=norm)
            axes[r, c + 2].set_title(name.upper() if r == 0 else "", fontsize=11)

        for c in range(ncols):
            axes[r, c].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label('Depth (meters)', fontsize=12)

    if dataset_title:
        fig.suptitle(dataset_title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Визуализация сохранена: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    # ========== ЗАГРУЗКА ДАТАСЕТОВ ==========
    nyu_val_dir = "data/raw/val/official"
    diode_hdf5 = "data/diode_hdf5/diode_train.hdf5"
    
    # NYU val (645 изображений)
    nyu_val_dataset = NYUDepthDataset(nyu_val_dir, img_size=(256, 384))
    nyu_val_loader = DataLoader(nyu_val_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"\n NYU Val: {len(nyu_val_dataset)} изображений")
    
    # DIODE — используем все изображения как тестовый сет
    if os.path.exists(diode_hdf5):
        diode_dataset = DIODEDepthDataset(diode_hdf5, img_size=(256, 384))
        diode_loader = DataLoader(diode_dataset, batch_size=4, shuffle=False, num_workers=0)
        print(f" DIODE: {len(diode_dataset)} изображений")
    else:
        diode_loader = None
        print("⚠️ DIODE не найден")
    
    print("\n" + "="*60)
    print(" ЗАГРУЗКА МОДЕЛЕЙ")
    print("="*60)
    
    models = {}
    for name in MODEL_CONFIGS:
        model, img_size = load_model(name, device)
        models[name] = (model, img_size)

    # ========== ВИЗУАЛИЗАЦИЯ НА ПРИМЕРАХ ==========
    print("\n" + "="*60)
    print(" ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ")
    print("="*60)
    visualize_predictions(
        models, nyu_val_dataset, NYU_DEPTH_SCALE, device,
        n_samples=4, out_path="results/qualitative_nyu.png",
        dataset_title="NYU Depth V2 (интерьеры)"
    )
    if diode_loader is not None:
        visualize_predictions(
            models, diode_dataset, DIODE_DEPTH_SCALE, device,
            n_samples=4, out_path="results/qualitative_diode.png",
            dataset_title="DIODE (outdoor)"
        )

    print("\n" + "="*60)
    print(" ОЦЕНКА НА NYU DEPTH V2 (интерьеры, 0.5-10 м)")
    print("="*60)
    
    nyu_results = {}
    for name, (model, img_size) in models.items():
        print(f"\n{name.upper()}...")
        metrics = evaluate_on_dataset(model, img_size, nyu_val_loader, NYU_DEPTH_SCALE, device)
        nyu_results[name] = metrics
        print(f"  RMSE: {metrics['RMSE']:.4f} м")
        print(f"  MAE:  {metrics['MAE']:.4f} м")
        print(f"  δ1:   {metrics['δ1']*100:.1f}%")
    
    if diode_loader is not None:
        print("\n" + "="*60)
        print(" ОЦЕНКА НА DIODE (улицы + интерьеры, 0.5-50 м)")
        print("="*60)
        
        diode_results = {}
        for name, (model, img_size) in models.items():
            print(f"\n{name.upper()}...")
            metrics = evaluate_on_dataset(model, img_size, diode_loader, DIODE_DEPTH_SCALE, device)
            diode_results[name] = metrics
            print(f"  RMSE: {metrics['RMSE']:.4f} м")
            print(f"  MAE:  {metrics['MAE']:.4f} м")
            print(f"  δ1:   {metrics['δ1']*100:.1f}%")
    else:
        diode_results = None
    
    print("\n" + "="*60)
    print(" ИТОГОВАЯ ТАБЛИЦА")
    print("="*60)
    
    print(f"\n{'='*70}")
    print(f"{'Model':<20} | {'NYU RMSE':>10} {'MAE':>10} {'δ1':>8} | {'DIODE RMSE':>11} {'MAE':>10} {'δ1':>8}")
    print(f"{'-'*70}")
    for name in nyu_results:
        nyu = nyu_results[name]
        if diode_results and name in diode_results:
            diode = diode_results[name]
            print(f"{name:<20} | {nyu['RMSE']:>10.3f} {nyu['MAE']:>10.3f} {nyu['δ1']*100:>7.1f}% | "
                  f"{diode['RMSE']:>11.3f} {diode['MAE']:>10.3f} {diode['δ1']*100:>7.1f}%")
        else:
            print(f"{name:<20} | {nyu['RMSE']:>10.3f} {nyu['MAE']:>10.3f} {nyu['δ1']*100:>7.1f}% | "
                  f"{'N/A':>11} {'N/A':>10} {'N/A':>8}")
    
    if diode_results:
        print("\n" + "="*60)
        print(" АНАЛИЗ ДОМЕННОГО РАЗРЫВА (Domain Gap)")
        print("="*60)
        for name in nyu_results:
            if name in diode_results:
                nyu_rmse = nyu_results[name]['RMSE']
                diode_rmse = diode_results[name]['RMSE']
                gap = diode_rmse - nyu_rmse
                print(f"{name}: NYU={nyu_rmse:.3f}м, DIODE={diode_rmse:.3f}м, gap={gap:+.3f}м")
    


if __name__ == "__main__":
    main()