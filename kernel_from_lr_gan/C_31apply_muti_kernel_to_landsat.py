"""
将 muti_kernel/train.py 生成的模糊核应用于 Landsat 数据进行降分辨率
- 输入: Landsat NC 文件 (任意分辨率，如 128x128 或 256x256)
- 输出: 降分辨率后的影像 (输入分辨率 / 下采样倍数，默认 8 倍)
  例如: 128x128 -> 16x16 (8 倍)，或 256x256 -> 32x32 (8 倍)

使用说明:
1) 默认从 muti_kernel/kernelgan_out/final_results/kernel_per_band.npy 读取分波段核
   - 如果想使用训练过程中的某次 batch 核，可改为 iter_kernels/batch_kernels_iterXXXX.npy
2) 默认处理 r"H:\Landsat\patches_all" 下的所有 .nc 文件
3) 输出写回原 nc 文件的 lr 组，同时在 output_dir 下保存对比图
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset
import matplotlib.pyplot as plt


def load_kernel(kernel_path: str) -> torch.Tensor:
    """加载训练好的模糊核，支持 [C, kH, kW] 或 [B, C, kH, kW]。"""
    kernel = np.load(kernel_path)
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32))

    # 如果是 batch 核 [B, C, kH, kW]，先对 batch 求平均，保持分波段核
    if kernel_tensor.ndim == 4:
        kernel_tensor = kernel_tensor.mean(dim=0)

    if kernel_tensor.ndim == 2:
        kernel_tensor = kernel_tensor.unsqueeze(0)  # [1, kH, kW]

    print(f"加载模糊核: {os.path.basename(kernel_path)}")
    print(f"  形状: {kernel_tensor.shape}")
    print(f"  总和: {kernel_tensor.sum().item():.6f}")
    return kernel_tensor


def load_landsat_nc(nc_path: str) -> tuple[torch.Tensor, list[str]]:
    """读取 Landsat NC 文件，返回影像和波段名。"""
    with Dataset(nc_path, 'r') as ds:
        grp = ds.groups['hr']
        band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
        bands = []
        for band_name in band_names:
            data = grp.variables[band_name][:]
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(fill_value=np.nan)
            bands.append(data.astype(np.float32))
        img_array = np.stack(bands, axis=0)
        img_tensor = torch.from_numpy(img_array)

    print(f"影像形状: {img_tensor.shape}")
    print(f"数值范围: [{img_tensor.min().item():.2f}, {img_tensor.max().item():.2f}]")
    return img_tensor, band_names


def apply_kernel_degradation(img: torch.Tensor, kernel: torch.Tensor, downscale_factor: int = 8) -> torch.Tensor:
    """使用模糊核对影像进行降分辨率。"""
    C, H, W = img.shape

    # 确保核的维度正确
    if kernel.ndim == 2:
        kernel = kernel.unsqueeze(0).repeat(C, 1, 1)
    elif kernel.ndim == 3:
        assert kernel.shape[0] == C, f"核的波段数({kernel.shape[0]})与影像波段数({C})不匹配"
    else:
        raise ValueError(f"不支持的核维度: {kernel.shape}")

    kH, kW = kernel.shape[-2:]

    # 归一化核（确保每个波段的核总和为 1）
    kernel_normalized = kernel.clone()
    for i in range(C):
        ksum = kernel[i].sum()
        if ksum > 0:
            kernel_normalized[i] = kernel[i] / ksum

    # 重塑为卷积权重 [C,1,kH,kW]，分组卷积实现逐通道卷积
    conv_kernel = kernel_normalized.unsqueeze(1)

    img_batch = img.unsqueeze(0)  # [1, C, H, W]

    pad_h = kH // 2
    pad_w = kW // 2
    img_batch_padded = F.pad(img_batch, (pad_w, pad_w, pad_h, pad_h), mode='replicate')

    blurred = F.conv2d(input=img_batch_padded, weight=conv_kernel, padding=0, groups=C)

    # 下采样
    lr_img = blurred
    steps = int(np.log2(downscale_factor))
    for _ in range(steps):
        lr_img = F.avg_pool2d(lr_img, kernel_size=2, stride=2)

    return lr_img.squeeze(0)


def visualize_comparison(hr_img: torch.Tensor, lr_img: torch.Tensor, band_names: list[str], output_dir: str, filename_prefix: str) -> None:
    """可视化 HR 与 LR 对比并保存。"""
    C = hr_img.shape[0]
    n_show = min(C, 5)
    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        hr_band = hr_img[i].numpy()
        lr_band = lr_img[i].numpy()
        vmin = min(hr_band.min(), lr_band.min())
        vmax = max(hr_band.max(), lr_band.max())

        ax = axes[0, i]
        im = ax.imshow(hr_band, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'HR - {band_names[i]}\n{hr_band.shape}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, i]
        im = ax.imshow(lr_band, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'LR - {band_names[i]}\n{lr_band.shape}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'{filename_prefix} - HR vs LR Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{filename_prefix}_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {out_path}")


def process_landsat_folder(landsat_dir: str, kernel_path: str, output_dir: str, downscale_factor: int = 8, visualize_top_n: int = 5) -> None:
    """批量处理 Landsat NC 文件，写回 lr 组并保存对比图。"""
    kernel = load_kernel(kernel_path)
    nc_files = sorted(glob.glob(os.path.join(landsat_dir, '*.nc')))
    if len(nc_files) == 0:
        print(f"在 {landsat_dir} 中没有找到 .nc 文件")
        return

    print(f"\n找到 {len(nc_files)} 个 Landsat NC 文件")
    os.makedirs(output_dir, exist_ok=True)

    for idx, nc_file in enumerate(nc_files):
        print(f"\n{'='*60}")
        print(f"处理 [{idx+1}/{len(nc_files)}]: {os.path.basename(nc_file)}")
        print(f"使用核: {os.path.basename(kernel_path)}")
        print(f"{'='*60}")
        try:
            img, band_names = load_landsat_nc(nc_file)
            lr_img = apply_kernel_degradation(img, kernel, downscale_factor=downscale_factor)
            print(f"降分辨率结果: {img.shape} -> {lr_img.shape}")

            nc_path_abs = os.path.abspath(nc_file)
            with Dataset(nc_path_abs, 'a', format='NETCDF4') as ds:
                if 'lr' not in ds.groups:
                    if 'y_lr' not in ds.dimensions:
                        ds.createDimension('y_lr', lr_img.shape[1])
                    if 'x_lr' not in ds.dimensions:
                        ds.createDimension('x_lr', lr_img.shape[2])
                    grp_lr = ds.createGroup('lr')
                else:
                    grp_lr = ds.groups['lr']

                lr_data = lr_img.numpy()
                for c, band_name in enumerate(band_names[:lr_img.shape[0]]):
                    if band_name not in grp_lr.variables:
                        var = grp_lr.createVariable(band_name, 'f4', ('y_lr', 'x_lr'), zlib=True)
                    else:
                        var = grp_lr.variables[band_name]
                    var[:] = lr_data[c, :, :]
                    var.long_name = f"TOA Radiance at {band_name.split('_')[-1]} nm (LR)"
                    var.units = "W m-2 sr-1 um-1"

                ds.history = "Added lr group by applying learned blur kernel and downsampling"

            print(f"已更新: {nc_path_abs}")

            if idx < visualize_top_n:
                visualize_comparison(img, lr_img, band_names, output_dir, os.path.basename(nc_file).replace('.nc', ''))

        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"处理完成！结果保存在: {output_dir}")
    print(f"核文件: {kernel_path}")
    print(f"数据源: {landsat_dir}")
    print(f"{'='*60}")


def main() -> None:
    # ========== 配置参数 ==========
    landsat_dir = r"H:\Landsat\patches_all"  # Landsat NC 文件夹
    # 使用训练完成后的分波段核
    kernel_path = r"D:\Py_Code\Kernel-Modeling-Super-Resolution\kernel_from_lr_gan\muti_kernel\kernelgan_out\final_results\kernel_per_band.npy"
    # 如果想用迭代中的 batch 核，替换为 iter_kernels/batch_kernels_iterXXXX.npy
    # kernel_path = r"D:\Py_Code\Kernel-Modeling-Super-Resolution\kernel_from_lr_gan\muti_kernel\kernelgan_out\iter_kernels\batch_kernels_iter100.npy"

    output_dir = 'landsat_lr_output_muti'  # 输出对比图及写回的 nc 文件所在目录
    downscale_factor = 8

    print("=" * 60)
    print("Landsat 降分辨率处理 (使用 muti_kernel 训练核)")
    print("=" * 60)
    print(f"Landsat 文件夹: {landsat_dir}")
    print(f"模糊核: {kernel_path}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    if not os.path.exists(kernel_path):
        print(f"\n错误: 核文件不存在: {kernel_path}")
        return
    if not os.path.exists(landsat_dir):
        print(f"\n错误: Landsat 文件夹不存在: {landsat_dir}")
        return

    process_landsat_folder(
        landsat_dir=landsat_dir,
        kernel_path=kernel_path,
        output_dir=output_dir,
        downscale_factor=downscale_factor,
        visualize_top_n=5,
    )


if __name__ == "__main__":
    main()
