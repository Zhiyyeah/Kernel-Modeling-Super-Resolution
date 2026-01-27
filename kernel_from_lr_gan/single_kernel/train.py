import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import netCDF4 as nc
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# 保证本文件夹内模块可导入，无论从何处执行脚本
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from networks import MultiBandLinearGenerator, PatchDiscriminator
from loss import lsgan_d_loss, lsgan_g_loss, kernel_regularization


def load_patches_from_folder(patch_dir: str) -> list:
    """
    从文件夹加载所有.nc格式的patch文件（从denoised组读取）
    参数:
        patch_dir (str): patch文件夹路径
    返回:
        list: patch文件路径列表
    """
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.nc')))
    
    if len(patch_files) == 0:
        raise ValueError(f"在 {patch_dir} 中没有找到 .nc 文件")
    
    print(f"✓ 找到 {len(patch_files)} 个patch文件")
    
    return patch_files


def load_full_patches(patch_files: list, batch_size: int, device: torch.device = None) -> torch.Tensor:
    """
    从NetCDF文件的denoised组中随机加载完整的patch（不裁剪）
    
    参数:
        patch_files (list): patch文件路径列表（.nc文件）
        batch_size (int): 批次大小
        device (torch.device, optional): 目标设备
    
    返回:
        torch.Tensor: 形状 [B, 5, H, W]，完整尺寸的patch
    """
    # 随机选择batch_size个文件
    selected_indices = torch.randint(low=0, high=len(patch_files), size=(batch_size,))
    
    # 5个波段的名称
    band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
    
    patches = []
    for idx in selected_indices:
        # 从NetCDF文件的denoised组读取5个波段
        with nc.Dataset(patch_files[idx.item()], 'r') as ds:
            denoised_group = ds.groups['denoised']
            bands = []
            for band_name in band_names:
                band_data = denoised_group.variables[band_name][:]
                bands.append(band_data)
            # 堆叠成 [5, H, W]
            patch = np.stack(bands, axis=0)
        
        patch_tensor = torch.from_numpy(patch.astype(np.float32))
        
        # 检查NaN值
        if torch.isnan(patch_tensor).any():
            nan_count = torch.isnan(patch_tensor).sum().item()
            nan_ratio = nan_count / patch_tensor.numel() * 100
            raise ValueError(
                f"Patch文件包含NaN值: {patch_files[idx.item()]}\n"
                f"NaN像素数: {nan_count}/{patch_tensor.numel()} ({nan_ratio:.2f}%)\n"
                f"这表示patch质量不足，应该在生成阶段就被过滤掉。"
            )
        
        patches.append(patch_tensor)
    
    result = torch.stack(tensors=patches, dim=0)  # [B, 5, H, W]
    
    if device is not None:
        result = result.to(device)
    
    return result


def crop_patches(full_patches: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    从完整patch中随机裁剪指定尺寸的子patch
    
    参数:
        full_patches (torch.Tensor): 完整patch，形状 [B, 5, H, W]
        crop_size (int): 裁剪尺寸
    
    返回:
        torch.Tensor: 裁剪后的patch，形状 [B, 5, crop_size, crop_size]
    """
    B, C, H, W = full_patches.shape
    
    if H < crop_size or W < crop_size:
        raise ValueError(f"Patch尺寸 {H}x{W} 小于裁剪尺寸 {crop_size}x{crop_size}")
    
    cropped = []
    for i in range(B):
        # 每个patch随机选择裁剪位置
        max_y = H - crop_size
        max_x = W - crop_size
        y0 = torch.randint(0, max_y + 1, (1,)).item() if max_y > 0 else 0
        x0 = torch.randint(0, max_x + 1, (1,)).item() if max_x > 0 else 0
        
        cropped_patch = full_patches[i, :, y0:y0+crop_size, x0:x0+crop_size]
        cropped.append(cropped_patch)
    
    return torch.stack(cropped, dim=0)


def main():
    # 无监督 KernelGAN 训练（5 波段）- 从patch文件夹采样
    use_cpu = False
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'Using device: {device}')

    # 数据路径配置（从denoised文件夹读取去噪后的数据）
    patch_dir = r"H:\GOCI-2\patch_output_nc\patches_denoised"
    print(f'\n 数据路径: {patch_dir}')
    patch_files = load_patches_from_folder(patch_dir)
    
    # 训练配置
    iters = 10000
    hr_patch_size = 256          # 高分辨率patch尺寸
    lr_crop_size = 32            # 从HR中裁剪的低分辨率尺寸
    batch_size = 16
    lr_rate = 4e-4
    reg_weight = 0.002            # Loss_Reg 权重
    grad_clip_norm = 20.0         # 全局梯度裁剪阈值
    outdir = r'output\kernelgan_out_denoised_single_kernel_stable_10000iters'
    log_every = 100              # 普通训练日志间隔
    kernel_log_every = 100       # 核详细统计输出间隔
    save_intermediate = True     # 是否保存中间核
    verbose = True               # 是否输出权重/梯度等详细信息

    # 数据配置说明
    print(f'\n 训练配置:')
    print(f'  - 高分辨率(HR)输入: {hr_patch_size}×{hr_patch_size}')
    print(f'  - 低分辨率(LR)裁剪: {lr_crop_size}×{lr_crop_size} (从HR随机裁剪)')
    print(f'  - 批次大小: {batch_size}')
    print(f'  - 总迭代次数: {iters}\n')
    
    # 创建输出目录和日志文件
    os.makedirs(outdir, exist_ok=True)
    log_file = os.path.join(outdir, 'training_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('Iteration,Loss_D,Loss_G_adv,Loss_Reg,Loss_Reg_weighted\n')
    print(f'✓ 训练日志将保存到: {log_file}\n')

    # 模型（生成器输入 5 通道，输出 5 通道；判别器接受 5 通道）
    G = MultiBandLinearGenerator(in_ch=5, mid_ch=32)
    G = G.to(device)
    D = PatchDiscriminator(in_ch=5, base_ch=64, num_blocks=4).to(device)

    opt_D = optim.Adam(params=D.parameters(), lr=lr_rate, betas=(0.5, 0.999))
    opt_G = optim.Adam(params=G.parameters(), lr=lr_rate, betas=(0.5, 0.999))

    def kernel_metrics(k: torch.Tensor) -> dict:
        """
        计算模糊核的多项统计指标用于监控训练进度。
        参数:
            k (torch.Tensor): 模糊核，形状 [kH, kW]，应已归一化（sum≈1）。
        返回:
            dict: 包含以下键值对：
                - k_shape (str): 核尺寸，如 '13x13'。
                - k_sum (float): 核元素总和（应接近 1.0）。
                - k_max (float): 核最大值。
                - k_min (float): 核最小值。
                - k_std (float): 核标准差。
                - sparsity (float): 稀疏度，>5%最大值的元素占比。
                - center_offset (float): 质心到几何中心的欧氏距离。
        """
        kH, kW = k.shape
        # 稀疏度：非零或>阈值元素占比
        thresh = k.max() * 0.05
        sparsity = float((k > thresh).float().mean().item())
        # 中心距离
        yy, xx = torch.meshgrid(torch.arange(kH, device=k.device), torch.arange(kW, device=k.device), indexing='ij')
        mass = k + 1e-12
        cy = (yy.float() * mass).sum() / mass.sum()
        cx = (xx.float() * mass).sum() / mass.sum()
        center_y = (kH - 1) / 2.0
        center_x = (kW - 1) / 2.0
        center_offset = float(((cy - center_y) ** 2 + (cx - center_x) ** 2).sqrt().item())
        return {
            'k_shape': f'{kH}x{kW}',
            'k_sum': float(k.sum().item()),
            'k_max': float(k.max().item()),
            'k_min': float(k.min().item()),
            'k_std': float(k.std().item()),
            'sparsity': sparsity,
            'center_offset': center_offset,
        }

    def ascii_kernel(k: torch.Tensor, size: int = 11) -> str:
        """
        将模糊核缩放到指定尺寸并转为 ASCII 字符块，便于快速目视检查集中度。
        参数:
            k (torch.Tensor): 模糊核，形状 [kH, kW]。
            size (int): 输出 ASCII 方阵的尺寸（size x size）。默认 11。
        返回:
            str: ASCII 艺术字符画，用灰度字符表示核的分布，
                从 ' ' （最弱）到 '@' （最强）。
        """
        import torch.nn.functional as F
        k2 = k.unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,H,W]
        k2 = F.interpolate(input=k2, size=(size, size), mode='bilinear', align_corners=False)
        k2 = k2[0,0]
        chars = " .:-=+*#%@"  # 10级
        mx = k2.max().item() + 1e-12
        out_lines = []
        for i in range(size):
            line = ''
            for j in range(size):
                v = k2[i,j].item() / mx
                idx = min(int(v * (len(chars)-1)), len(chars)-1)
                line += chars[idx]
            out_lines.append(line)
        return '\n'.join(out_lines)

    def generator_weight_stats(G: torch.nn.Module) -> str:
        """
        提取生成器每个波段链的首层与末层权重统计信息。
        参数:
            G (torch.nn.Module): MultiBandLinearGenerator 实例。
        返回:
            str: 每个波段链的权重范数和最大值，格式如：
                'B0(L0n=0.123,Ln=0.456) B1(L0n=0.234,Ln=0.567) ...'
        """
        vals = []
        # 每个波段链的首层与末层权重统计
        for b, chain in enumerate(G.chains):
            w0 = chain[0].weight.detach()
            w_last = chain[-1].weight.detach()
            vals.append(f"B{b}(L0n={w0.norm().item():.3f},Ln={w_last.norm().item():.3f})")
        return ' '.join(vals)

    prev_k = None  # 用于计算核变化幅度

    pbar = tqdm(range(iters), desc='Training', unit='iter')
    for t in pbar:
        # 非配对训练：分别采样高分辨率和低分辨率patch
        
        # 1. 直接加载完整的高分辨率patch [B,5,256,256]
        patches = load_full_patches(
            patch_files=patch_files,
            batch_size=batch_size,
            device=device
        )
        
        # 2. 从另一批完整patch中随机裁剪低分辨率区域 [B,5,32,32]
        # 这是独立采样的patch，保留真实的模糊特性
        full_patches_for_crop = load_full_patches(
            patch_files=patch_files,
            batch_size=batch_size,
            device=device
        )
        real_ds = crop_patches(full_patches_for_crop, crop_size=lr_crop_size)
        
        # 3. 生成器输出：通过学习到的退化核生成低分辨率 [B,5,32,32]
        fake_ds = G(patches)

        # 训练D
        D.train(); G.train()
        pred_real = D(real_ds)
        pred_fake = D(fake_ds.detach())
        loss_D = lsgan_d_loss(pred_real, pred_fake)
        opt_D.zero_grad(); loss_D.backward()
        grad_norm_D = clip_grad_norm_(parameters=D.parameters(), max_norm=grad_clip_norm)
        opt_D.step()

        # 训练G（对抗 + 核正则）
        pred_fake = D(fake_ds)
        loss_G_adv = lsgan_g_loss(pred_fake)
        ks_band = G.extract_effective_kernels()  # [C,kH,kW]
        # 对每个波段核分别做正则再求平均
        reg_list = []
        for i in range(ks_band.shape[0]):
            reg_list.append(kernel_regularization(
                k=ks_band[i], 
                alpha=0.5,    # Sum-to-1
                beta=0.5,     # Boundaries
                gamma=5.0,    # Sparse
                delta=1.0,    # Center
                epsilon=3.0   # CenterMax - 强制中心点最大
            ))
        loss_reg = torch.mean(input=torch.stack(tensors=reg_list))
        loss_reg_weighted = reg_weight * loss_reg
        k = ks_band.mean(dim=0)  # 用于后续单核统计（合并核）
        loss_G = loss_G_adv + loss_reg_weighted
        opt_G.zero_grad(); loss_G.backward()
        grad_norm_G = clip_grad_norm_(parameters=G.parameters(), max_norm=grad_clip_norm)
        opt_G.step()
        
        # 保存损失值到文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'{t+1},{loss_D.item():.6f},{loss_G_adv.item():.6f},{loss_reg.item():.6f},{loss_reg_weighted.item():.6f}\n')

        # 更新进度条显示
        pbar.set_postfix({
            'D': f'{loss_D.item():.4f}',
            'G_adv': f'{loss_G_adv.item():.4f}',
            'RegW': f'{loss_reg_weighted.item():.4f}',
            'gN_D': f'{float(grad_norm_D):.2f}',
            'gN_G': f'{float(grad_norm_G):.2f}'
        })
        
        # 详细日志
        if (t + 1) % log_every == 0:
            extra = ''
            if verbose:
                extra = generator_weight_stats(G)
            tqdm.write(
                f"[LOG] Iter {t+1}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | Reg: {loss_reg.item():.4f} "
                f"| gN_D: {float(grad_norm_D):.2f} | gN_G: {float(grad_norm_G):.2f} {extra}"
            )

        if (t + 1) % kernel_log_every == 0:
            km = kernel_metrics(k)
            delta = 0.0
            if prev_k is not None:
                delta = torch.norm(k - prev_k).item()
            prev_k = k.detach().clone()
            tqdm.write(f"  [Kernel] shape={km['k_shape']} sum={km['k_sum']:.4f} max={km['k_max']:.4f} min={km['k_min']:.4f} std={km['k_std']:.4f} sparsity(>5%max)={km['sparsity']:.3f} center_offset={km['center_offset']:.3f} delta_L2={delta:.5f}")
            # 多波段核
            ks_all = G.extract_effective_kernels()  # [C,kH,kW]
            k_merged = ks_all.mean(dim=0)
            if verbose:
                tqdm.write("  [Kernel ASCII merged]\n" + ascii_kernel(k_merged))
                # 输出前几个波段的核最大值
                band_max = ' '.join([f'b{i}_max={ks_all[i].max().item():.3f}' for i in range(min(ks_all.shape[0],3))])
                tqdm.write(f"  [Bands] {band_max}")
            if save_intermediate:
                os.makedirs(outdir, exist_ok=True)
                np.save(os.path.join(outdir, f'kernel_iter{t+1}.npy'), k_merged.cpu().numpy())
                # 同时保存分波段核
                np.save(os.path.join(outdir, f'kernel_per_band_iter{t+1}.npy'), ks_all.cpu().numpy())

    # 提取并保存最终核
    ks_final = G.extract_effective_kernels().cpu().numpy()
    k_final_merged = ks_final.mean(axis=0)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'kernel_per_band.npy'), ks_final)
    np.save(os.path.join(outdir, 'kernel_merged.npy'), k_final_merged)
    print(f"Saved per-band kernels -> kernel_per_band.npy ({ks_final.shape}), merged -> kernel_merged.npy sum={k_final_merged.sum():.6f}")


if __name__ == "__main__":
    main()
