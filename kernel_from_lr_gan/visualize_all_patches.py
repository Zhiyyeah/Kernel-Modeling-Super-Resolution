import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.gridspec import GridSpec

# ================= 配置区域 =================
PATCH_DIR = '/Users/zy/Downloads/GOCI-2/patches_all'
OUTPUT_DIR = os.path.join(PATCH_DIR, 'visualizations')

def vis_GOCI_npy(npy_path):
    data = np.load(npy_path)  # [5, H, W]
    
    # 提取RGB波段
    r = data[3]
    g = data[2]
    b = data[1]
    
    # 归一化到0-1用于显示
    def normalize_for_display(img, percentile_clip=True):
        if percentile_clip:
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99)
        else:
            vmin = img.min()
            vmax = img.max()
        
        if vmax <= vmin:
            vmax = vmin + 1e-6
        
        normalized = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        return normalized
    
    r_norm = normalize_for_display(r)
    g_norm = normalize_for_display(g)
    b_norm = normalize_for_display(b)
    
    rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)  # [H, W, 3]
    
    plt.figure(figsize=(6,6))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(os.path.basename(npy_path))
    plt.savefig(npy_path.replace('.npy', '_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    path = '/Users/zy/Downloads/GOCI-2/patches_all'
    npy_files = glob.glob(os.path.join(path, '*.npy'))

    for npy_file in npy_files:
        vis_GOCI_npy(npy_file)