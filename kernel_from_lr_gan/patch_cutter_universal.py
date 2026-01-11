"""
通用 Patch 裁剪器 - 适用于 Landsat 和 GOCI-2 数据
输入的是文件夹路径
1、自动遍历文件集中的nc文件
2、读取所有波段数据，应用水体掩码
3、根据配置的 patch 大小和步长裁剪成多个 patch
4、保存为 NetCDF 格式，包含必要的元数据
5、可选：生成 NIR 波段的可视化图像
"""

import os
from pathlib import Path
from typing import List, Tuple
from math import ceil, sqrt

import numpy as np
from netCDF4 import Dataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# 可配置参数
# ============================================================================
PATCH_SIZE = 256  # patch 尺寸
STRIDE_RATIO = 0.5  # 步长比例（0.5 表示 50% 重叠）
NAN_THRESHOLD = 0.0  # 允许的 NaN 比例，超过此值的 patch 将被丢弃
THRESHOLD_MIN = 0.000001  # NIR 最小阈值（水体筛选）
THRESHOLD_MAX = 7.0  # NIR 最大阈值（水体筛选）
BAND_NAMES = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
NIR_BAND_INDEX = 4  # 865nm 波段在列表中的索引
INVALID_VALUE = -9999.0  # NetCDF 中的无效值标记


# ============================================================================
# 核心函数
# ============================================================================
def read_nc_bands(nc_path: str) -> Tuple[np.ndarray, dict]:
    """
    读取 NetCDF 文件中的所有波段数据。
    
    参数:
        nc_path: NetCDF 文件路径
    
    返回:
        data: [5, H, W] 的数组
        metadata: 包含原始文件信息的字典
    """
    with Dataset(nc_path, 'r') as ds:
        grp = ds.groups['geophysical_data']
        
        all_bands_data = []
        for band_name in BAND_NAMES:
            data = grp.variables[band_name][:]
            # 处理 MaskedArray
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(fill_value=INVALID_VALUE)
            all_bands_data.append(data.astype(np.float32))
        
        # 堆叠成 [5, H, W]
        all_bands = np.stack(arrays=all_bands_data, axis=0)
        
        # 提取元数据
        metadata = {
            'source_file': os.path.basename(nc_path),
            'band_names': BAND_NAMES,
            'invalid_value': INVALID_VALUE
        }
        
        return all_bands, metadata


def apply_water_mask(data: np.ndarray, threshold_min: float, threshold_max: float) -> np.ndarray:
    """
    应用水体掩码（基于 NIR 波段阈值）。
    
    参数:
        data: [5, H, W] 的数组
        threshold_min: NIR 最小阈值
        threshold_max: NIR 最大阈值
    
    返回:
        masked_data: [5, H, W] 的掩码后数组
    """
    # 将无效值设为 NaN
    data[data == INVALID_VALUE] = np.nan
    
    # 提取 NIR 波段
    nir_band = data[NIR_BAND_INDEX].copy()
    
    # 创建水体掩码
    water_mask = (nir_band >= threshold_min) & (nir_band <= threshold_max)
    
    # 应用掩码到所有波段
    masked_data = data.copy()
    for i in range(data.shape[0]):
        masked_data[i][~water_mask] = np.nan
    
    # 统计信息
    total_valid = np.sum(~np.isnan(nir_band))
    water_pixels = np.sum(water_mask)
    water_ratio = water_pixels / total_valid * 100 if total_valid > 0 else 0
    
    print(f"  有效像素总数: {total_valid:,}")
    print(f"  水体像素数: {water_pixels:,} ({water_ratio:.2f}%)")
    
    return masked_data


def create_patches_nc(
    data: np.ndarray,
    patch_size: int,
    stride_ratio: float,
    nan_threshold: float,
    output_dir: str,
    prefix: str,
    metadata: dict
) -> Tuple[int, int]:
    """
    将数据划分成 patch 并保存为 NetCDF 格式。
    
    参数:
        data: [C, H, W] 的输入数组
        patch_size: patch 大小
        stride_ratio: 步长比例（0 < stride_ratio <= 1）
        nan_threshold: NaN 像素比例阈值
        output_dir: 保存目录
        prefix: 文件名前缀
        metadata: 元数据字典
    
    返回:
        total_patches: 总 patch 数
        kept_patches: 保留的 patch 数
    """
    channels, height, width = data.shape
    stride = int(patch_size * stride_ratio)
    
    h_patches = (height - patch_size) // stride + 1
    w_patches = (width - patch_size) // stride + 1
    
    print(f"  数据尺寸: {data.shape}")
    print(f"  Patch 大小: {patch_size}x{patch_size}, 步长: {stride} ({int(stride_ratio*100)}% 重叠)")
    print(f"  可生成的 patch 网格: {h_patches}x{w_patches} = {h_patches*w_patches} 个")
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_patches = 0
    kept_patches = 0
    
    for i in range(h_patches):
        for j in range(w_patches):
            total_patches += 1
            
            # 提取 patch
            h_start = i * stride
            h_end = h_start + patch_size
            w_start = j * stride
            w_end = w_start + patch_size
            
            patch = data[:, h_start:h_end, w_start:w_end]
            
            # 计算 NaN 比例
            nan_ratio = np.sum(np.isnan(patch)) / patch.size
            
            # 如果 NaN 比例超过阈值，跳过此 patch
            if nan_ratio > nan_threshold:
                continue
            
            kept_patches += 1
            
            # 保存为 NetCDF
            patch_filename = f"{prefix}_{i:03d}_{j:03d}.nc"
            patch_path = os.path.join(output_dir, patch_filename)
            
            save_patch_as_nc(patch, patch_path, metadata, i, j, h_start, w_start)
    
    print(f"  总共生成: {total_patches} 个 patch")
    print(f"  保留: {kept_patches} 个 patch (丢弃 {total_patches - kept_patches} 个)")
    print(f"  Patch 已保存至: {output_dir}")
    
    return total_patches, kept_patches


def save_patch_as_nc(
    patch: np.ndarray,
    output_path: str,
    metadata: dict,
    grid_i: int,
    grid_j: int,
    h_offset: int,
    w_offset: int
) -> None:
    """
    将单个 patch 保存为 NetCDF 文件。
    
    参数:
        patch: [C, H, W] 的 patch 数据
        output_path: 输出文件路径
        metadata: 元数据字典
        grid_i: 网格行索引
        grid_j: 网格列索引
        h_offset: 垂直偏移量
        w_offset: 水平偏移量
    """
    channels, height, width = patch.shape
    
    with Dataset(output_path, 'w', format='NETCDF4') as ds:
        # 创建维度
        ds.createDimension('bands', channels)
        ds.createDimension('y', height)
        ds.createDimension('x', width)
        
        # 创建变量
        data_var = ds.createVariable('data', 'f4', ('bands', 'y', 'x'))
        data_var[:] = patch
        
        # 添加元数据
        ds.source_file = metadata.get('source_file', 'unknown')
        ds.band_names = ','.join(metadata.get('band_names', []))
        ds.invalid_value = metadata.get('invalid_value', -9999.0)
        ds.grid_i = grid_i
        ds.grid_j = grid_j
        ds.h_offset = h_offset
        ds.w_offset = w_offset
        ds.patch_size = height


def visualize_nir_overview(
    nir_band: np.ndarray,
    nir_mask: np.ndarray,
    output_path: str,
    filename: str,
    threshold_min: float,
    threshold_max: float,
    dpi: int = 300
) -> None:
    """
    生成 NIR 波段的可视化图（可选）。
    
    参数:
        nir_band: NIR 原始数据
        nir_mask: NIR 掩码后数据
        output_path: 输出目录
        filename: 文件名
        threshold_min: 最小阈值
        threshold_max: 最大阈值
        dpi: 图像分辨率
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：完整 NIR 波段
    valid_nir = nir_band[~np.isnan(nir_band)]
    if len(valid_nir) > 0:
        vmin = np.percentile(valid_nir, 2)
        vmax = np.percentile(valid_nir, 98)
    else:
        vmin, vmax = 0, 1
    
    im0 = axes[0].imshow(nir_band, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'865nm NIR Band (Enhanced)\nRange: [{vmin:.1f}, {vmax:.1f}]', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 右图：阈值筛选的水体区域
    im1 = axes[1].imshow(nir_mask, cmap='viridis', interpolation='nearest', 
                         vmin=threshold_min, vmax=threshold_max)
    axes[1].set_title(f'Water Body Regions\n{threshold_min} < NIR < {threshold_max}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'NIR Analysis - {filename}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    img_filename = filename.replace('.nc', '_nir_overview.png')
    img_path = os.path.join(output_path, img_filename)
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  可视化结果已保存至: {img_path}")


def process_single_nc(
    nc_path: str,
    output_base_dir: str,
    create_visualization: bool = False
) -> dict:
    """
    处理单个 NetCDF 文件。
    
    参数:
        nc_path: NetCDF 文件路径
        output_base_dir: 输出基础目录
        create_visualization: 是否生成可视化图
    
    返回:
        stats: 统计信息字典
    """
    filename = os.path.basename(nc_path)
    print(f"\n处理文件: {filename}")
    print("=" * 80)
    
    # 读取数据
    print("读取波段数据...")
    data, metadata = read_nc_bands(nc_path)
    
    # 应用水体掩码
    print("应用水体掩码...")
    masked_data = apply_water_mask(data, THRESHOLD_MIN, THRESHOLD_MAX)
    
    # 可选：生成可视化
    if create_visualization:
        print("生成可视化图...")
        nir_raw = data[NIR_BAND_INDEX].copy()
        nir_mask = masked_data[NIR_BAND_INDEX].copy()
        vis_dir = os.path.join(output_base_dir, 'visualizations')
        visualize_nir_overview(nir_raw, nir_mask, vis_dir, filename, 
                              THRESHOLD_MIN, THRESHOLD_MAX)
    
    # 创建 patch
    print("生成 patch...")
    patch_dir = os.path.join(output_base_dir, 'patches')
    prefix = os.path.splitext(filename)[0]
    
    total, kept = create_patches_nc(
        data=masked_data,
        patch_size=PATCH_SIZE,
        stride_ratio=STRIDE_RATIO,
        nan_threshold=NAN_THRESHOLD,
        output_dir=patch_dir,
        prefix=prefix,
        metadata=metadata
    )
    
    stats = {
        'filename': filename,
        'total_patches': total,
        'kept_patches': kept,
        'discarded_patches': total - kept
    }
    
    return stats


def process_folder(
    input_folder: str,
    output_folder: str,
    create_visualization: bool = False
) -> List[dict]:
    """
    处理文件夹中的所有 NetCDF 文件。
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        create_visualization: 是否生成可视化图
    
    返回:
        all_stats: 所有文件的统计信息列表
    """
    input_path = Path(input_folder)
    nc_files = sorted(input_path.glob('*.nc'))
    
    if not nc_files:
        print(f"在 {input_folder} 中未找到 .nc 文件")
        return []
    
    print(f"找到 {len(nc_files)} 个 .nc 文件")
    print("=" * 80)
    
    all_stats = []
    
    for nc_file in nc_files:
        try:
            stats = process_single_nc(
                str(nc_file),
                output_folder,
                create_visualization
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"处理文件 {nc_file.name} 时出错: {e}")
            continue
    
    # 打印汇总统计
    print("\n" + "=" * 80)
    print("处理完成！汇总统计:")
    print("=" * 80)
    total_kept = sum(s['kept_patches'] for s in all_stats)
    total_discarded = sum(s['discarded_patches'] for s in all_stats)
    print(f"总共处理文件数: {len(all_stats)}")
    print(f"总保留 patch 数: {total_kept}")
    print(f"总丢弃 patch 数: {total_discarded}")
    
    return all_stats


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    主函数：配置路径并执行处理。
    """
    # ========== 配置区 ==========
    INPUT_FOLDER = r"H:\GOCI-2"  # 输入文件夹路径（包含 .nc 文件）
    OUTPUT_FOLDER = r"H:\GOCI-2\patch_output_nc"  # 输出文件夹路径
    CREATE_VISUALIZATION = True  # 是否生成可视化图
    
    # 打印配置
    print("=" * 80)
    print("通用 Patch 裁剪器 - Landsat/GOCI-2")
    print("=" * 80)
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print(f"Patch 大小: {PATCH_SIZE}")
    print(f"步长比例: {STRIDE_RATIO} ({int(STRIDE_RATIO*100)}% 重叠)")
    print(f"NaN 阈值: {NAN_THRESHOLD}")
    print(f"NIR 阈值范围: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print(f"波段列表: {BAND_NAMES}")
    print("=" * 80)
    
    # 执行处理
    stats = process_folder(INPUT_FOLDER, OUTPUT_FOLDER, CREATE_VISUALIZATION)
    
    print("\n所有任务完成！")


if __name__ == "__main__":
    main()
