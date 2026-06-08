import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

def get_sample_paths(data_dir: str = "data/raw/val/official"):
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    return [os.path.join(data_dir, f) for f in h5_files[:3]]

def read_h5_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        image = f['rgb'][:]
        depth = f['depth'][:]
    return image, depth

def visualize_sample(h5_path, idx: int, save_dir: str = "notebooks/outputs"):
    os.makedirs(save_dir, exist_ok=True)
    
    image, depth = read_h5_file(h5_path)
    
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    
    d_min, d_max = depth.min(), depth.max()
    if d_max == d_min:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"NYU Depth V2 - Sample #{idx}", fontsize=14, fontweight='bold')
    
    axes[0].imshow(image)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_norm, cmap='magma')
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(image)
    axes[2].contour(depth_norm, levels=10, colors='cyan', alpha=0.7, linewidths=1.5)
    axes[2].set_title("Image + Depth Contours")
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Loading NYU Depth V2 dataset from HDF5 files...")
    try:
        h5_files = get_sample_paths()
        print(f"Found {len(h5_files)} HDF5 files.")
        
        for idx, h5_path in enumerate(h5_files):
            print(f"Visualizing sample #{idx}: {os.path.basename(h5_path)}")
            visualize_sample(h5_path, idx=idx)
            
        print("Done. Data loaded and visualized successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()