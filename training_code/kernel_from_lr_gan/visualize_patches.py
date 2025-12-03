import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.gridspec import GridSpec

# ================= 配置区域 =================
PATCH_DIR = '/Users/zy/Python_code/My_Git/match_cor/Imagery_WaterLand/Patches_128_Clean'
OUTPUT_DIR = os.path.join(PATCH_DIR, 'visualizations')

# 显示设置
NUM_SAMPLES = 16  # 每次显示的patch数量
PERCENTILE_CLIP = (1, 99)  # 百分位裁剪，增强对比度

# RGB波段索引 (从0开始): 443nm(0), 490nm(1), 555nm(2), 660nm(3), 865nm(4)
# 真彩色RGB: 660nm(R), 555nm(G), 490nm(B)
RGB_BANDS = [3, 2, 1]  # 对应 660, 555, 490

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建输出目录: {path}")

def read_patch(nc_path):
    """读取单个patch的NetCDF文件"""
    try:
        with Dataset(nc_path, 'r') as ds:
            data = ds.variables['L_TOA'][:]  # [5, H, W]
            return data
    except Exception as e:
        print(f"读取错误 {nc_path}: {e}")
        return None

def normalize_for_display(img, percentile_clip=True):
    """归一化到0-1用于显示"""
    if percentile_clip:
        vmin = np.percentile(img, PERCENTILE_CLIP[0])
        vmax = np.percentile(img, PERCENTILE_CLIP[1])
    else:
        vmin = img.min()
        vmax = img.max()
    
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    normalized = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return normalized

def create_rgb_composite(patch_data):
    """创建RGB合成图像"""
    # 提取RGB波段
    r = patch_data[RGB_BANDS[0]]
    g = patch_data[RGB_BANDS[1]]
    b = patch_data[RGB_BANDS[2]]
    
    # 分别归一化每个波段
    r_norm = normalize_for_display(r)
    g_norm = normalize_for_display(g)
    b_norm = normalize_for_display(b)
    
    # 堆叠为RGB图像
    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    return rgb

def visualize_patches_grid(nc_files, start_idx=0, num_samples=16):
    """网格显示多个patches"""
    ensure_dir(OUTPUT_DIR)
    
    # 计算网格尺寸
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for i in range(num_samples):
        idx = start_idx + i
        if idx >= len(nc_files):
            break
        
        nc_path = nc_files[idx]
        file_name = os.path.basename(nc_path)
        
        # 读取数据
        patch_data = read_patch(nc_path)
        if patch_data is None:
            continue
        
        # 创建RGB合成
        rgb = create_rgb_composite(patch_data)
        
        # 显示
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(rgb)
        ax.set_title(file_name.replace('.nc', ''), fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'GOCI-2 Water Patches (RGB: 660-555-490 nm)\nSamples {start_idx}-{start_idx+num_samples-1}', 
                 fontsize=14, y=0.995)
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, f'patches_grid_{start_idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()

def visualize_single_patch_multiband(nc_path):
    """显示单个patch的所有波段"""
    file_name = os.path.basename(nc_path)
    patch_data = read_patch(nc_path)
    if patch_data is None:
        return
    
    band_names = ['443nm (Blue)', '490nm (Cyan)', '555nm (Green)', 
                  '660nm (Red)', '865nm (NIR)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 显示各个波段
    for i in range(5):
        band = patch_data[i]
        band_norm = normalize_for_display(band)
        
        im = axes[i].imshow(band_norm, cmap='gray')
        axes[i].set_title(f'{band_names[i]}\nRange: [{band.min():.2f}, {band.max():.2f}]')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # RGB合成
    rgb = create_rgb_composite(patch_data)
    axes[5].imshow(rgb)
    axes[5].set_title('RGB Composite\n(660-555-490 nm)')
    axes[5].axis('off')
    
    plt.suptitle(f'Multi-band Visualization: {file_name}', fontsize=14)
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, f'multiband_{file_name.replace(".nc", ".png")}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 获取所有nc文件
    nc_files = sorted(glob.glob(os.path.join(PATCH_DIR, '*.nc')))
    print(f"找到 {len(nc_files)} 个 patch 文件。")
    
    if len(nc_files) == 0:
        print("未找到任何nc文件！")
        return
    
    # 1. 显示前几个patches的详细多波段视图
    print(f"\n生成前3个patches的多波段视图...")
    for i in range(min(3, len(nc_files))):
        visualize_single_patch_multiband(nc_files[i])
    
    # 2. 生成网格总览图（分批显示）
    print(f"\n生成网格总览图...")
    num_batches = int(np.ceil(len(nc_files) / NUM_SAMPLES))
    for batch in range(min(num_batches, 5)):  # 最多生成5批
        start_idx = batch * NUM_SAMPLES
        visualize_patches_grid(nc_files, start_idx=start_idx, num_samples=NUM_SAMPLES)
    
    print(f"\n完成！所有可视化结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
