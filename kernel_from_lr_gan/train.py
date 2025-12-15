import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# 保证本文件夹内模块可导入，无论从何处执行脚本
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from networks import MultiBandLinearGenerator, PatchDiscriminator
from loss import lsgan_d_loss, lsgan_g_loss, kernel_regularization


def load_patches_from_folder(patch_dir: str) -> tuple[list, int]:
    """
    从文件夹加载所有.npy格式的patch文件
    参数:
        patch_dir (str): patch文件夹路径
    返回:
        tuple[list, int]: 
            - patches_list: patch文件路径列表
            - original_size: 原始patch大小（从第一个文件读取）
    """
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.npy')))
    
    if len(patch_files) == 0:
        raise ValueError(f"在 {patch_dir} 中没有找到 .npy 文件")
    
    # 读取第一个patch来获取原始尺寸
    first_patch = np.load(patch_files[0])  # [5, H, W]
    original_size = first_patch.shape[1]  # 假设是正方形patch
    
    print(f"找到 {len(patch_files)} 个patch文件")
    print(f"原始patch尺寸: {first_patch.shape}")
    
    return patch_files, original_size


def sample_patches_from_files(patch_files: list, batch_size: int, target_size: int = 128, 
                               original_size: int = 128, device: torch.device = None) -> torch.Tensor:
    """
    从预先保存的patch文件中完全随机采样一批，并调整到目标尺寸
    
    参数:
        patch_files (list): patch文件路径列表
        batch_size (int): 批次大小
        target_size (int): 目标patch大小，默认128
        original_size (int): 原始patch大小，默认128
        device (torch.device, optional): 目标设备
    
    返回:
        torch.Tensor: 形状 [B, 5, target_size, target_size]
    """
    # 随机选择batch_size个文件
    selected_indices = torch.randint(low=0, high=len(patch_files), size=(batch_size,))
    
    patches = []
    for idx in selected_indices:
        # 加载patch [5, original_size, original_size]
        patch = np.load(patch_files[idx.item()])
        patch_tensor = torch.from_numpy(patch.astype(np.float32))
        
        # 检查NaN值：如果有NaN则直接报错
        if torch.isnan(patch_tensor).any():
            nan_count = torch.isnan(patch_tensor).sum().item()
            nan_ratio = nan_count / patch_tensor.numel() * 100
            raise ValueError(
                f"Patch文件包含NaN值: {patch_files[idx.item()]}\n"
                f"NaN像素数: {nan_count}/{patch_tensor.numel()} ({nan_ratio:.2f}%)\n"
                f"这表示patch质量不足，应该在生成阶段就被过滤掉。"
            )
        
        # 如果原始尺寸不等于目标尺寸，随机裁剪子patch
        if original_size != target_size:
            H, W = patch_tensor.shape[-2], patch_tensor.shape[-1]
            
            # 计算可裁剪的范围
            max_y = H - target_size
            max_x = W - target_size
            
            if max_y <= 0 or max_x <= 0:
                raise ValueError(
                    f"Patch尺寸 {H}x{W} 小于目标尺寸 {target_size}x{target_size}，无法裁剪"
                )
            
            # 完全随机选择裁剪起始位置
            y0 = torch.randint(0, max_y + 1, (1,)).item()
            x0 = torch.randint(0, max_x + 1, (1,)).item()
            
            # 裁剪patch
            patch_tensor = patch_tensor[:, y0:y0+target_size, x0:x0+target_size]
        
        patches.append(patch_tensor)
    
    result = torch.stack(tensors=patches, dim=0)  # [B, 5, target_size, target_size]
    
    if device is not None:
        result = result.to(device)
    
    return result


def main():
    # 无监督 KernelGAN 训练（5 波段）- 从patch文件夹采样
    use_cpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'Using device: {device}')

    # 数据路径配置
    patch_dir = r"H:\GOCI-2\patches_all"
    print(f'使用patch文件夹: {patch_dir}')
    patch_files, original_patch_size = load_patches_from_folder(patch_dir)

    print(f'原始patch尺寸: {original_patch_size}x{original_patch_size}')
    
    # 训练配置
    iters = 3000
    patch_size = 256
    batch_size = 8
    lr_rate = 1e-4
    outdir = './kernelgan_out'
    log_every = 100              # 普通训练日志间隔
    kernel_log_every = 300       # 核详细统计输出间隔
    mini_log_every = 10          # 轻量日志（更频繁）
    save_intermediate = True     # 是否保存中间核
    verbose = True               # 是否输出权重/梯度等详细信息

    # 加载数据
    print(f'将从 {len(patch_files)} 个文件中随机采样')
    print(f'原始patch尺寸: {original_patch_size}x{original_patch_size}')
    print(f'目标patch尺寸: {patch_size}x{patch_size}')

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

    for t in range(iters):
        # 非配对训练：分别采样高分辨率和低分辨率patch
        
        # 1. 采样高分辨率patch用于生成器输入 [B,5,128,128]
        patches = sample_patches_from_files(
            patch_files=patch_files,
            batch_size=batch_size,
            target_size=patch_size,  # 128
            original_size=original_patch_size,
            device=device
        )
        
        # 2. 采样真实的低分辨率patch作为判别器真值 [B,5,32, 32]
        # 这是独立采样的，不是通过数学下采样得到的
        real_ds = sample_patches_from_files(
            patch_files=patch_files,
            batch_size=batch_size,
            target_size=32,  # 直接采样32x32，保留真实的模糊特性
            original_size=original_patch_size,
            device=device
        )
        
        # 3. 生成器输出：通过学习到的退化核生成低分辨率 [B,5,32,32]
        fake_ds = G(patches)

        # 训练D
        D.train(); G.train()
        pred_real = D(real_ds)
        pred_fake = D(fake_ds.detach())
        loss_D = lsgan_d_loss(pred_real, pred_fake)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # 训练G（对抗 + 核正则）
        pred_fake = D(fake_ds)
        loss_G_adv = lsgan_g_loss(pred_fake)
        ks_band = G.extract_effective_kernels()  # [C,kH,kW]
        # 对每个波段核分别做正则再求平均
        reg_list = []
        for i in range(ks_band.shape[0]):
            reg_list.append(kernel_regularization(k=ks_band[i], alpha=0.5, beta=0.5, gamma=5.0, delta=1.0))
        loss_reg = torch.mean(input=torch.stack(tensors=reg_list))
        k = ks_band.mean(dim=0)  # 用于后续单核统计（合并核）
        loss_G = loss_G_adv + loss_reg
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        # 轻量日志
        if (t + 1) % mini_log_every == 0:
            print(f"Iter {t+1}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f}")
        if (t + 1) % log_every == 0:
            extra = ''
            if verbose:
                extra = generator_weight_stats(G)
            print(f"[LOG] Iter {t+1}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | Reg: {loss_reg.item():.4f} {extra}")

        if (t + 1) % kernel_log_every == 0:
            km = kernel_metrics(k)
            delta = 0.0
            if prev_k is not None:
                delta = torch.norm(k - prev_k).item()
            prev_k = k.detach().clone()
            print(f"  [Kernel] shape={km['k_shape']} sum={km['k_sum']:.4f} max={km['k_max']:.4f} min={km['k_min']:.4f} std={km['k_std']:.4f} sparsity(>5%max)={km['sparsity']:.3f} center_offset={km['center_offset']:.3f} delta_L2={delta:.5f}")
            # 多波段核
            ks_all = G.extract_effective_kernels()  # [C,kH,kW]
            k_merged = ks_all.mean(dim=0)
            if verbose:
                print("  [Kernel ASCII merged]\n" + ascii_kernel(k_merged))
                # 输出前几个波段的核最大值
                band_max = ' '.join([f'b{i}_max={ks_all[i].max().item():.3f}' for i in range(min(ks_all.shape[0],3))])
                print(f"  [Bands] {band_max}")
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
