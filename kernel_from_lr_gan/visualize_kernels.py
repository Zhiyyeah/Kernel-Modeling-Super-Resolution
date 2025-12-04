import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def visualize_kernels(kernel_path, output_path=None):
    """
    可视化模糊核（支持单个核或多波段核）
    
    参数:
        kernel_path (str): .npy核文件路径
        output_path (str, optional): 输出图片路径，默认保存到同目录
    """
    if not os.path.exists(kernel_path):
        print(f"文件不存在: {kernel_path}")
        return
    
    # 加载核
    kernel = np.load(kernel_path)
    print(f"核文件: {os.path.basename(kernel_path)}")
    print(f"形状: {kernel.shape}")
    
    # 判断是单核还是多波段核
    if kernel.ndim == 2:
        # 单个核 [H, W]
        visualize_single_kernel(kernel, os.path.basename(kernel_path), output_path)
    elif kernel.ndim == 3:
        # 多波段核 [n_bands, H, W]
        visualize_multiband_kernels(kernel, os.path.basename(kernel_path), output_path)
    else:
        print(f"不支持的核维度: {kernel.ndim}")


def visualize_single_kernel(kernel, title, output_path=None):
    """可视化单个核"""
    print(f"总和: {kernel.sum():.6f}")
    print(f"最大值: {kernel.max():.6f}")
    print(f"最小值: {kernel.min():.6f}")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(kernel, cmap='gray', interpolation='nearest')
    ax.set_title(f'{title}\nSum: {kernel.sum():.4f}, Max: {kernel.max():.4f}', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Kernel Value', fontsize=10)
    
    # 在核上标注数值（如果核不太大）
    if kernel.shape[0] <= 15 and kernel.shape[1] <= 15:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                text_color = 'white' if kernel[i, j] > kernel.max() * 0.5 else 'black'
                ax.text(j, i, f'{kernel[i, j]:.3f}', 
                       ha='center', va='center', 
                       fontsize=7, color=text_color)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = title.replace('.npy', '_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存至: {output_path}")
    plt.show()


def visualize_multiband_kernels(kernels, title, output_path=None):
    """可视化多波段核"""
    n_bands = kernels.shape[0]
    band_names = ['443nm', '490nm', '555nm', '660nm', '865nm'][:n_bands]
    
    print(f"波段数: {n_bands}")
    for i in range(n_bands):
        print(f"  波段{i} ({band_names[i]}): sum={kernels[i].sum():.4f}, max={kernels[i].max():.4f}")
    
    # 创建子图 (n_bands + 1个，最后一个是平均核)
    ncols = min(3, n_bands + 1)
    nrows = (n_bands + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    
    vmax = kernels.max()
    
    # 绘制每个波段
    for i in range(n_bands):
        ax = axes[i]
        kernel = kernels[i]
        
        im = ax.imshow(kernel, cmap='gray', interpolation='nearest', vmin=0, vmax=vmax)
        ax.set_title(f'Band {i}: {band_names[i]}\nSum={kernel.sum():.4f}', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 绘制平均核
    kernel_mean = kernels.mean(axis=0)
    ax = axes[n_bands]
    im = ax.imshow(kernel_mean, cmap='gray', interpolation='nearest', vmin=0, vmax=vmax)
    ax.set_title(f'Mean Kernel\nSum={kernel_mean.sum():.4f}', 
                fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余子图
    for i in range(n_bands + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title.replace('.npy', ''), fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path is None:
        output_path = title.replace('.npy', '_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存至: {output_path}")
    plt.show()


if __name__ == "__main__":
    # 使用示例
    kernel_dir = 'kernelgan_out'
    
    # 查找核文件
    kernel_files = sorted(glob.glob(os.path.join(kernel_dir, '*.npy')))
    
    if len(kernel_files) == 0:
        print(f"在 {kernel_dir} 中没有找到 .npy 文件")
    else:
        print(f"找到 {len(kernel_files)} 个核文件\n")
        
        # 可视化每个核文件
        for kernel_file in kernel_files:
            print("=" * 60)
            visualize_kernels(kernel_file)
            print()
