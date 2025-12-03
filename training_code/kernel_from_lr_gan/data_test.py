import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

PATCH_SIZE = 128  # 覆盖默认值
def create_patches(data, patch_size=PATCH_SIZE, nan_threshold=0.05, output_dir=None, prefix='patch'):
    """
    将数据划分成patch并过滤掉NaN过多的patch
    
    参数:
        data (np.ndarray): 输入数据 [C, H, W] 或 [H, W]
        patch_size (int): patch大小，默认256
        nan_threshold (float): NaN像素比例阈值，超过此值的patch将被丢弃，默认0.1 (10%)
        output_dir (str, optional): 保存patch的目录，如果为None则不保存
        prefix (str): patch文件名前缀
    
    返回:
        list: 有效patch列表
        int: 总patch数
        int: 保留的patch数
    """
    if data.ndim == 2:
        # 如果是2D数据，添加通道维度
        data = data[np.newaxis, :, :]
    
    channels, height, width = data.shape
    
    patches = []
    total_patches = 0
    kept_patches = 0
    
    # 计算可以生成的patch数量
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    print(f"\n开始生成patch...")
    print(f"输入数据尺寸: {data.shape}")
    print(f"Patch大小: {patch_size}x{patch_size}")
    print(f"可生成的patch网格: {h_patches}x{w_patches} = {h_patches*w_patches}个")
    
    for i in range(h_patches):
        for j in range(w_patches):
            total_patches += 1
            
            # 提取patch
            h_start = i * patch_size
            h_end = h_start + patch_size
            w_start = j * patch_size
            w_end = w_start + patch_size
            
            patch = data[:, h_start:h_end, w_start:w_end]
            
            # 计算NaN比例
            nan_ratio = np.sum(np.isnan(patch)) / patch.size
            
            # 如果NaN比例超过阈值，跳过此patch
            if nan_ratio > nan_threshold:
                continue
            
            kept_patches += 1
            patches.append(patch)
            
            # 保存patch
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                patch_filename = f"{prefix}_{i:03d}_{j:03d}.npy"
                patch_path = os.path.join(output_dir, patch_filename)
                np.save(patch_path, patch)
    
    print(f"总共生成: {total_patches}个patch")
    print(f"保留: {kept_patches}个patch (丢弃了{total_patches-kept_patches}个NaN过多的patch)")
    if output_dir is not None:
        print(f"Patch已保存至: {output_dir}")
    
    return patches, total_patches, kept_patches

def visualize_nir_threshold(nc_path, threshold_min=0.000001, threshold_max=7, 
                            output_path=None, dpi=600, show_plot=True,
                            create_patch=False, patch_size=PATCH_SIZE, nan_threshold=0.0, patch_output_dir=None):
    """
    可视化GOCI-2 NIR波段并根据阈值筛选水体区域
    
    参数:
        nc_path (str): NetCDF文件路径
        threshold_min (float): 最小阈值，默认0.000001
        threshold_max (float): 最大阈值，默认7
        output_path (str, optional): 输出图片路径，默认为None（自动生成）
        dpi (int): 输出图片分辨率，默认600
        show_plot (bool): 是否显示图片，默认True
        create_patch (bool): 是否创建patch，默认False
        patch_size (int): patch大小，默认256
        nan_threshold (float): NaN像素比例阈值，默认0.1 (10%)
        patch_output_dir (str, optional): patch保存目录，默认为None（自动生成）
    
    返回:
        dict: 包含统计信息的字典
    """
    print(f"读取文件: {nc_path}")
    
    # 读取所有波段数据
    with Dataset(nc_path, 'r') as ds:
        grp = ds.groups['geophysical_data']
        
        # 读取所有5个波段
        bands = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
        all_bands_data = []
        
        for band_name in bands:
            data = grp.variables[band_name][:]
            # 处理MaskedArray
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(fill_value=-9999.0)
            all_bands_data.append(data.astype(np.float32))
        
        # 堆叠成 [5, H, W] 的数组
        all_bands = np.stack(arrays=all_bands_data, axis=0)
        
        # 获取NIR波段用于可视化
        nir_raw = all_bands[4].copy()  # 865nm是第5个波段
    
    print(f"影像尺寸: {nir_raw.shape}")
    
    # 打印统计信息
    valid = nir_raw[nir_raw != -9999]
    if len(valid) > 0:
        print(f"\n865nm (NIR)波段统计: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}")
    
    # 将无效值设为NaN以便正确显示
    nir_raw[nir_raw == -9999] = np.nan
    
    # 创建阈值掩码图
    nir_mask = nir_raw.copy()
    nir_mask[(nir_raw >= threshold_max) | (nir_raw <= threshold_min)] = np.nan
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：完整的865nm NIR波段（调整动态范围以增强对比度）
    valid_nir = nir_raw[~np.isnan(nir_raw)]
    vmin = np.percentile(valid_nir, 2)
    vmax = np.percentile(valid_nir, 98)
    
    im0 = axes[0].imshow(nir_raw, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'865nm Near-Infrared Band (Enhanced)\nWater appears dark, Land/Cloud appears bright\nRange: [{vmin:.1f}, {vmax:.1f}]', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('Radiance (W m⁻² sr⁻¹ μm⁻¹)', fontsize=10)
    
    # 右图：阈值筛选的水体区域
    im1 = axes[1].imshow(nir_mask, cmap='viridis', interpolation='nearest', vmin=threshold_min, vmax=threshold_max)
    axes[1].set_title(f'{threshold_min} < 865nm < {threshold_max} (Water Body Regions)\nOnly showing pixels with radiance in [{threshold_min}, {threshold_max}]', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Radiance (W m⁻² sr⁻¹ μm⁻¹)', fontsize=10)
    
    # 统计信息
    water_pixels = np.sum((nir_raw < threshold_max) & (nir_raw > threshold_min))
    total_valid = np.sum(~np.isnan(nir_raw))
    water_ratio = water_pixels / total_valid * 100 if total_valid > 0 else 0
    
    print(f"\n865nm波段统计:")
    print(f"有效像素总数: {total_valid:,}")
    print(f"{threshold_min} < 865nm < {threshold_max} 的像素数: {water_pixels:,} ({water_ratio:.2f}%)")
    print(f"其他像素数: {total_valid - water_pixels:,} ({100-water_ratio:.2f}%)")
    
    plt.suptitle(f'GOCI-2 L1B - 865nm NIR Analysis\n{nc_path.split("/")[-1]}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        output_path = '/Users/zy/Downloads/GOCI-2/visualization_results'
    os.makedirs(output_path, exist_ok=True)
    
    img_filename = nc_path.split("/")[-1].replace('.nc', '_visualization_raw.png')
    img_path = os.path.join(output_path, img_filename)
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
    print(f"\n可视化结果已保存至: {img_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # 生成patch（如果需要）
    patches_info = None
    if create_patch:
        print("\n" + "="*60)
        print("开始生成Patch")
        print("="*60)
        
        # 创建阈值范围内的数据（所有波段）
        # 使用NIR波段作为mask
        mask = (nir_raw >= threshold_min) & (nir_raw <= threshold_max)
        
        # 对所有波段应用mask
        masked_all_bands = all_bands.copy()
        for i in range(5):
            masked_all_bands[i][~mask] = np.nan
            masked_all_bands[i][masked_all_bands[i] == -9999] = np.nan
        
        # 确定patch输出目录
        if patch_output_dir is None:
            patch_output_dir = os.path.join(
                os.path.dirname(nc_path),
                'patches_all'
            )
        
        # 生成patch
        patches, total, kept = create_patches(
            data=masked_all_bands,
            patch_size=patch_size,
            nan_threshold=nan_threshold,
            output_dir=patch_output_dir,
            prefix=os.path.splitext(os.path.basename(nc_path))[0]
        )
        
        patches_info = {
            'total_patches': total,
            'kept_patches': kept,
            'discarded_patches': total - kept,
            'patch_output_dir': patch_output_dir,
            'patch_size': patch_size,
            'nan_threshold': nan_threshold
        }
        
        print("="*60)
        print("Patch生成完成")
        print("="*60)
    
    # 返回统计信息
    result = {
        'total_valid': int(total_valid),
        'water_pixels': int(water_pixels),
        'water_ratio': float(water_ratio),
        'threshold_min': threshold_min,
        'threshold_max': threshold_max,
        'output_path': img_path
    }
    
    if patches_info is not None:
        result['patches'] = patches_info
    
    return result

def main():
    # 示例调用
    nc_path = '/Users/zy/Downloads/GOCI-2'
    nc_list = sorted([f for f in os.listdir(nc_path) if f.endswith('.nc')])
    print(f"找到 {len(nc_list)} 个nc文件。")
    
    for f in nc_list:
        print(f"\n{'='*80}")
        print(f"处理文件: {f}")
        print(f"{'='*80}")
        
        # 调用可视化函数（带patch生成）
        stats = visualize_nir_threshold(
            nc_path=os.path.join(nc_path, f),
            threshold_min=0.000001,
            threshold_max=7,
            dpi=600,
            show_plot=False,
            create_patch=True,  # 启用patch生成
            patch_size=PATCH_SIZE,  # patch大小
            nan_threshold=0.0,  # 0% NaN阈值
            patch_output_dir=None  # 自动生成目录
        )
        
        print(f"\n返回的统计信息: {stats}")

if __name__ == "__main__":
    main()
