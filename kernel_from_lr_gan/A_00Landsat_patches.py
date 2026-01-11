"""
处理Landsat数据：去云并裁剪成patches
处理逻辑与 A_01data_GOCI_nc_folder.py 保持一致
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob

PATCH_SIZE = 256  # 与GOCI处理保持一致


def create_patches(data, patch_size=PATCH_SIZE, nan_threshold=0.0, output_dir=None, prefix='patch'):
    """
    将数据划分成patch并过滤掉NaN过多的patch，输出为NC文件
    
    参数:
        data (np.ndarray): 输入数据 [C, H, W] 或 [H, W]
        patch_size (int): patch大小，默认256
        nan_threshold (float): NaN像素比例阈值，超过此值的patch将被丢弃，默认0.0
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
    
    # 步长为patch大小的一半（50% 重叠）
    stride = patch_size // 2
    h_patches = (height - patch_size) // stride + 1
    w_patches = (width - patch_size) // stride + 1
    
    print(f"\n开始生成patch...")
    print(f"输入数据尺寸: {data.shape}")
    print(f"Patch大小: {patch_size}x{patch_size}")
    print(f"步长: {stride} (50% 重叠)")
    print(f"可生成的patch网格: {h_patches}x{w_patches} = {h_patches*w_patches}个")
    
    band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
    
    for i in range(h_patches):
        for j in range(w_patches):
            total_patches += 1
            
            # 提取patch
            h_start = i * stride
            h_end = h_start + patch_size
            w_start = j * stride
            w_end = w_start + patch_size
            
            patch = data[:, h_start:h_end, w_start:w_end]
            
            # 计算NaN比例
            nan_ratio = np.sum(np.isnan(patch)) / patch.size
            
            # 如果NaN比例超过阈值，跳过此patch
            if nan_ratio > nan_threshold:
                continue
            
            kept_patches += 1
            patches.append(patch)
            
            # 保存patch为NC文件
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                patch_filename = f"{prefix}_{i:03d}_{j:03d}.nc"
                patch_path = os.path.join(output_dir, patch_filename)
                
                # 创建NC文件
                with Dataset(patch_path, 'w', format='NETCDF4') as ds:
                    # 创建维度
                    ds.createDimension('band', channels)
                    ds.createDimension('y', patch_size)
                    ds.createDimension('x', patch_size)
                    
                    # 创建组结构
                    grp_hr = ds.createGroup('hr')
                    
                    # 创建每个波段的变量
                    for c in range(channels):
                        band_name = band_names[c] if c < len(band_names) else f'band_{c}'
                        var = grp_hr.createVariable(band_name, 'f4', ('y', 'x'), zlib=True)
                        var[:] = patch[c, :, :]
                        var.long_name = f"TOA Radiance at {band_name.split('_')[-1]} nm"
                        var.units = "W m-2 sr-1 um-1"
                    
                    # 添加全局属性
                    ds.title = f"Landsat Patch {i:03d}_{j:03d}"
                    ds.patch_size = patch_size
                    ds.n_bands = channels
    
    print(f"总共生成: {total_patches}个patch")
    print(f"保留: {kept_patches}个patch (丢弃了{total_patches-kept_patches}个NaN过多的patch)")
    if output_dir is not None:
        print(f"Patch已保存至: {output_dir}")
    
    return patches, total_patches, kept_patches


def process_landsat_nc(nc_path, threshold_min=0.000001, threshold_max=7, 
                       output_path=None, dpi=600, show_plot=True,
                       create_patch=False, patch_size=PATCH_SIZE, nan_threshold=0.0, 
                       patch_output_dir=None):
    """
    处理Landsat NC文件：根据865nm波段去云并生成patches
    
    参数:
        nc_path (str): NetCDF文件路径
        threshold_min (float): 最小阈值，默认0.000001
        threshold_max (float): 最大阈值，默认7
        output_path (str, optional): 输出图片路径，默认为None（自动生成）
        dpi (int): 输出图片分辨率，默认600
        show_plot (bool): 是否显示图片，默认True
        create_patch (bool): 是否创建patch，默认False
        patch_size (int): patch大小，默认128
        nan_threshold (float): NaN像素比例阈值，默认0.0
        patch_output_dir (str, optional): patch保存目录，默认为None（自动生成）
    
    返回:
        dict: 包含统计信息的字典
    """
    print(f"读取文件: {nc_path}")
    
    # 读取所有波段数据
    with Dataset(nc_path, 'r') as ds:
        # 检查是否有geophysical_data组
        if 'geophysical_data' in ds.groups:
            grp = ds.groups['geophysical_data']
        else:
            grp = ds
        
        # 读取5个波段 (对应GOCI的5个波段)
        bands = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
        all_bands_data = []
        
        for band_name in bands:
            if band_name not in grp.variables:
                raise KeyError(f"波段 {band_name} 不存在于文件中")
            
            data = grp.variables[band_name][:]
            
            # 处理MaskedArray
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(fill_value=np.nan)
            
            all_bands_data.append(data.astype(np.float32))
        
        # 堆叠成 [5, H, W] 的数组
        all_bands = np.stack(arrays=all_bands_data, axis=0)
        
        # 获取865nm波段用于去云
        nir_raw = all_bands[4].copy()  # 865nm是第5个波段
    
    print(f"影像尺寸: {nir_raw.shape}")
    
    # 打印统计信息
    valid = nir_raw[~np.isnan(nir_raw)]
    if len(valid) > 0:
        print(f"\n865nm (NIR)波段统计: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}")
    
    # 创建阈值掩码图（去云）
    nir_mask = nir_raw.copy()
    nir_mask[(nir_raw >= threshold_max) | (nir_raw <= threshold_min)] = np.nan
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：完整的865nm NIR波段（调整动态范围以增强对比度）
    valid_nir = nir_raw[~np.isnan(nir_raw)]
    if len(valid_nir) > 0:
        vmin = np.percentile(valid_nir, 2)
        vmax = np.percentile(valid_nir, 98)
    else:
        vmin, vmax = 0, 1
    
    im0 = axes[0].imshow(nir_raw, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'865nm Near-Infrared Band (Enhanced)\nWater appears dark, Land/Cloud appears bright\nRange: [{vmin:.1f}, {vmax:.1f}]', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('Radiance (W m⁻² sr⁻¹ μm⁻¹)', fontsize=10)
    
    # 右图：阈值筛选的水体区域（去云后）
    im1 = axes[1].imshow(nir_mask, cmap='viridis', interpolation='nearest', vmin=threshold_min, vmax=threshold_max)
    axes[1].set_title(f'{threshold_min} < 865nm < {threshold_max} (Cloud-free Water Body)\nOnly showing pixels with radiance in [{threshold_min}, {threshold_max}]', 
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
    print(f"云/陆地像素数: {total_valid - water_pixels:,} ({100-water_ratio:.2f}%)")
    
    plt.suptitle(f'Landsat - 865nm NIR Analysis (Cloud Removal)\n{os.path.basename(nc_path)}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        output_path = '/Users/zy/Downloads/Landsat/visualization_results'
    os.makedirs(output_path, exist_ok=True)
    
    img_filename = os.path.basename(nc_path).replace('.nc', '_visualization.png')
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
        print("开始生成Patch（去云后的数据）")
        print("="*60)
        
        # 创建阈值范围内的数据（所有波段）
        # 使用NIR波段作为mask进行去云
        mask = (nir_raw >= threshold_min) & (nir_raw <= threshold_max)
        
        # 对所有波段应用mask
        masked_all_bands = all_bands.copy()
        for i in range(5):
            masked_all_bands[i][~mask] = np.nan
        
        # 确定patch输出目录
        if patch_output_dir is None:
            patch_output_dir = '/Users/zy/Downloads/Landsat/patches_all'
        
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
    """
    批量处理 /Users/zy/Downloads/Landsat 文件夹中的所有NC文件
    """
    # Landsat数据目录
    landsat_dir = '/Users/zy/Downloads/Landsat'
    
    # 查找所有NC文件
    nc_files = sorted(glob.glob(os.path.join(landsat_dir, '*.nc')))
    
    if len(nc_files) == 0:
        print(f"在 {landsat_dir} 中没有找到 .nc 文件")
        return
    
    print(f"找到 {len(nc_files)} 个Landsat NC文件。\n")
    
    # 处理每个文件
    all_stats = []
    for idx, nc_file in enumerate(nc_files):
        print(f"\n{'='*80}")
        print(f"处理文件 [{idx+1}/{len(nc_files)}]: {os.path.basename(nc_file)}")
        print(f"{'='*80}")
        
        try:
            # 调用处理函数（带patch生成和去云）
            stats = process_landsat_nc(
                nc_path=nc_file,
                threshold_min=0.000001,  # 与GOCI处理保持一致
                threshold_max=9,         # 与GOCI处理保持一致
                dpi=600,
                show_plot=False,
                create_patch=True,       # 启用patch生成
                patch_size=PATCH_SIZE,   # 256x256
                nan_threshold=0.0,       # 0% NaN阈值（与GOCI一致）
                patch_output_dir=None    # 使用默认目录
            )
            
            all_stats.append(stats)
            print(f"\n处理完成！")
            
        except Exception as e: 
            print(f"\n处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总统计
    print("\n" + "="*80)
    print("处理完成汇总")
    print("="*80)
    print(f"成功处理: {len(all_stats)}/{len(nc_files)} 个文件")
    
    if all_stats:
        total_patches = sum(s.get('patches', {}).get('kept_patches', 0) for s in all_stats)
        print(f"总共生成: {total_patches} 个有效patch")
        print(f"Patch保存位置: {all_stats[0]['patches']['patch_output_dir']}")


if __name__ == "__main__":
    main()
