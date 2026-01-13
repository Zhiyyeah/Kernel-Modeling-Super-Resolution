from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from skimage.restoration import denoise_nl_means, estimate_sigma

# ==========================================
# 1. 核心读取与去噪功能 (保持不变)
# ==========================================

def read_nc(file_path:str, group_name, band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']):
    """读取NC文件"""
    data = []
    for band_name in band_names:
        try:
            band_data = xr.open_dataset(file_path, group=group_name)[band_name]
            data.append(band_data)
        except Exception as e:
            print(f"Warning: Could not read {band_name}: {e}")
            
    if not data:
        raise ValueError("No bands were read!")
        
    data_con = xr.concat(data, dim='band')
    data_con = data_con.where(data_con != 0, np.nan)
    return data_con, band_names

def denoise_band_float_nlm(img_float, h_factor=1.15, patch_size=7, patch_distance=11, verbose=True):
    """
    使用 Scikit-Image 进行浮点级 NLM 去噪
    """
    # 1. 处理 NaN
    valid_mask = ~np.isnan(img_float)
    if not valid_mask.any():
        return img_float, 0.0 
    
    fill_value = np.nanmean(img_float)
    img_filled = np.nan_to_num(img_float, nan=fill_value).astype(np.float32)
    
    # 2. 估计噪声 Sigma
    estimated_sigma = estimate_sigma(img_filled, average_sigmas=True)
    
    # 3. 设定去噪强度 h
    h_val = h_factor * estimated_sigma
    if verbose:
        print(f"    -> Sigma: {estimated_sigma:.6f} | h: {h_val:.6f}")

    # 4. 执行 NLM 去噪
    denoised = denoise_nl_means(
        img_filled, 
        h=h_val, 
        sigma=estimated_sigma,
        fast_mode=True, 
        patch_size=patch_size,
        patch_distance=patch_distance
    )
    
    # 5. 恢复 NaN
    denoised_final = np.where(valid_mask, denoised, np.nan)
    
    return denoised_final, estimated_sigma

# ==========================================
# 2. 修改后的评估模块 (只保留对比图和残差图)
# ==========================================

def calculate_metrics_simple(original, denoised):
    """只计算基础指标用于显示"""
    residual = original - denoised
    valid_mask = ~np.isnan(residual)
    res_clean = residual[valid_mask]
    
    if len(res_clean) == 0:
        return residual, 0, 0
        
    rmse = np.sqrt(np.mean(res_clean**2))
    std_res = np.std(res_clean)
    
    return residual, rmse, std_res

def evaluate_denoising(original, denoised, band_name, save_dir):
    """
    精简版评估：原始图 vs 去噪图 vs 残差图
    """
    save_dir = Path(save_dir)
    
    # --- 计算基础残差信息 ---
    residual, rmse, std_res = calculate_metrics_simple(original, denoised)
    print(f"[{band_name}] RMSE: {rmse:.5f}, Residual Std: {std_res:.5f}")

    # --- 计算原始图和去噪图的均值和标准差 ---
    orig_mean = np.nanmean(original)
    orig_std = np.nanstd(original)
    denoised_mean = np.nanmean(denoised)
    denoised_std = np.nanstd(denoised)
    
    print(f"    Original  -> Mean: {orig_mean:.5f}, Std: {orig_std:.5f}")
    print(f"    Denoised  -> Mean: {denoised_mean:.5f}, Std: {denoised_std:.5f}")

    # --- 绘图配置 ---
    # 统一色阶范围，确保对比公平 (使用 2%-98% 拉伸)
    vmin = np.nanpercentile(original, 2)
    vmax = np.nanpercentile(original, 98)
    
    # 设置画布：1行3列
    fig = plt.figure(figsize=(20, 6))
    plt.suptitle(f"Denoising Result: {band_name} (RMSE: {rmse:.4f})", fontsize=16)

    # === 子图 1: 原始带噪图 ===
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(original, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f"1. Original (Noisy)")
    # ax1.set_title(f"1. Original (Noisy)\nMean: {orig_mean:.5f}, Std: {orig_std:.5f}")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')

    # === 子图 2: 去噪后结果 ===
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(denoised, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("2. Denoised (Clean)")
    # ax2.set_title(f"2. Denoised (Clean)\nMean: {denoised_mean:.5f}, Std: {denoised_std:.5f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04) 
    ax2.axis('off')

    # === 子图 3: 残差空间分布 ===
    # 残差图使用 Coolwarm 色带，红正蓝负，0在中间
    ax3 = plt.subplot(1, 3, 3)
    res_limit = std_res * 3  # 限制显示范围以便看清噪点
    im3 = ax3.imshow(residual, cmap='coolwarm', vmin=-res_limit, vmax=res_limit)
    ax3.set_title("3. Residual Map (Removed Noise)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    plt.tight_layout()
    out_path = save_dir / f"{band_name}_compare.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    -> Saved comparison to {out_path}")

# ==========================================
# 3. 主程序
# ==========================================

def process_nc_file(file_path, output_dir, h_factor=1.8, plot=False, verbose=True):
    """
    处理单个NC文件进行去噪
    
    参数:
        file_path: 输入NC文件路径
        output_dir: 输出目录
        h_factor: 去噪强度因子 (推荐 1.0 - 2.5)
        plot: 是否生成对比图
        verbose: 是否显示详细信息
    
    返回:
        (success, output_path, error_msg)
    """
    import shutil
    
    try:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        if verbose:
            print(f"Loading: {file_path}")
        
        # 读取数据
        band_names_list = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
        arr, band_names = read_nc(str(file_path), 'geophysical_data', band_names_list)
        
        bands, height, width = arr.shape
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名（在原文件名基础上加_denoised）
        input_filename = file_path.stem  # 获取文件名（不含扩展名）
        output_filename = f"{input_filename}_denoised.nc"
        output_path = output_dir / output_filename
        
        # 如果需要画图，创建绘图目录
        if plot:
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储去噪后的数据
        denoised_data_list = []
        sigma_values = {}  # 记录每个波段的sigma值
        
        for i in range(bands):
            band_name = band_names[i]
            if verbose:
                print(f"\n--- Processing {band_name} ---")
            
            current_band = arr[i].values
            
            # 1. 去噪 (h_factor 可根据需要调整，推荐 1.0 - 2.5)
            denoised_img, sigma = denoise_band_float_nlm(current_band, h_factor=h_factor, patch_size=7, patch_distance=11, verbose=verbose)
            sigma_values[band_name] = sigma  # 保存sigma值
            
            # 2. 将去噪后的数据转换为xarray DataArray
            denoised_da = xr.DataArray(
                denoised_img,
                dims=arr[i].dims,
                coords=arr[i].coords,
                attrs=arr[i].attrs,
                name=band_name
            )
            denoised_data_list.append(denoised_da)
            
            # 3. 如果需要，绘图 (原始 vs 去噪 vs 残差)
            if plot:
                evaluate_denoising(current_band, denoised_img, band_name, plot_dir)
        
        # 保存所有波段到一个nc文件
        if verbose:
            print(f"\n--- Saving denoised data to {output_path} ---")
        
        # 先复制原始nc文件结构
        shutil.copy2(str(file_path), str(output_path))
        if verbose:
            print(f"  -> Copied original file structure")
        
        # 在新文件中添加 denoised 组
        denoised_dataset = xr.Dataset({da.name: da for da in denoised_data_list})
        
        # 添加全局属性：噪声sigma和去噪强度h
        denoised_dataset.attrs['h_factor'] = h_factor
        denoised_dataset.attrs['denoising_method'] = 'Non-Local Means (NLM)'
        denoised_dataset.attrs['patch_size'] = 7
        denoised_dataset.attrs['patch_distance'] = 11
        
        # 添加每个波段的sigma值
        for band_name, sigma in sigma_values.items():
            h_val = h_factor * sigma
            denoised_dataset.attrs[f'{band_name}_sigma'] = sigma
            denoised_dataset.attrs[f'{band_name}_h'] = h_val
        
        # 添加平均sigma和h值
        avg_sigma = np.mean(list(sigma_values.values()))
        avg_h = h_factor * avg_sigma
        denoised_dataset.attrs['average_sigma'] = avg_sigma
        denoised_dataset.attrs['average_h'] = avg_h
        
        denoised_dataset.to_netcdf(str(output_path), mode='a', group='denoised')
        if verbose:
            print(f"✓ Denoised data saved successfully in 'denoised' group!")
            print(f"  -> Average Sigma: {avg_sigma:.6f}, Average h: {avg_h:.6f}")
        
        return True, output_path, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}")
        return False, None, error_msg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to .nc file")
    parser.add_argument("--output", type=str, default="H:\\GOCI-2\\patch_output_nc\\patches_denoised", 
                        help="Output directory for denoised .nc files (default: H:\\GOCI-2\\patch_output_nc\\patches_denoised)")
    parser.add_argument("--h_factor", type=float, default=1.8,
                        help="Denoising strength factor (default: 1.8, recommended: 1.0-2.5)")
    parser.add_argument("--plot", action="store_true", 
                        help="Generate comparison plots (default: False)")
    args = parser.parse_args()

    # 调用处理函数
    success, output_path, error = process_nc_file(
        file_path=args.file_path,
        output_dir=args.output,
        h_factor=args.h_factor,
        plot=args.plot,
        verbose=True
    )
    
    if not success:
        print(f"\nProcessing failed: {error}")
        exit(1)

if __name__ == "__main__":
    main()