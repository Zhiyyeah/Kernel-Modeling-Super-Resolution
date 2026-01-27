"""
批量可视化 output/kernelgan_out_denoised_single_kernel 目录下的所有核(.npy)。
为每个核文件生成一张 PNG，并保存到 output/kernelgan_out_denoised_single_kernel/vis_all。
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def visualize_kernel_file(kernel_path: str, save_path: str) -> None:
    """可视化单个核文件并保存PNG"""
    kernel = np.load(kernel_path)
    name = os.path.basename(kernel_path)
    print(f"绘制: {name}, shape={kernel.shape}")

    # 单核: [H, W]; 多波段核: [B, H, W]
    if kernel.ndim == 2:
        kernels = [kernel]
        titles = ["Kernel"]
    elif kernel.ndim == 3:
        kernels = [kernel[i] for i in range(kernel.shape[0])]
        titles = [f"Band {i}" for i in range(kernel.shape[0])]
        # 添加平均核
        kernels.append(kernel.mean(axis=0))
        titles.append("Mean")
    else:
        print(f"  跳过: 不支持的维度 {kernel.ndim}")
        return

    n = len(kernels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    vmax = max(k.max() for k in kernels)
    for i, (k, title) in enumerate(zip(kernels, titles)):
        ax = axes[i]
        im = ax.imshow(k, cmap='gray', interpolation='nearest', vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nSum={k.sum():.4f}", fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余子图
    for j in range(len(kernels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存 -> {save_path}")


def main():
    kernel_dir = os.path.join('output', 'kernelgan_out_denoised_single_kernel_stable_10000iters')
    vis_dir = os.path.join(kernel_dir, 'vis_all')
    os.makedirs(vis_dir, exist_ok=True)

    kernel_files = sorted(glob.glob(os.path.join(kernel_dir, '*.npy')))
    if not kernel_files:
        print(f"在 {kernel_dir} 未找到 .npy 核文件")
        return

    print(f"找到 {len(kernel_files)} 个核文件，输出目录: {vis_dir}\n")
    for kf in kernel_files:
        out_name = os.path.basename(kf).replace('.npy', '.png')
        out_path = os.path.join(vis_dir, out_name)
        visualize_kernel_file(kf, out_path)

    print("\n完成全部可视化！")


if __name__ == '__main__':
    main()
