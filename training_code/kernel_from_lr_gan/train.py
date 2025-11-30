import os
import sys
import numpy as np
import torch
import torch.optim as optim

# 保证本文件夹内模块可导入，无论从何处执行脚本
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from networks import LinearGenerator, PatchDiscriminator
from loss import lsgan_d_loss, lsgan_g_loss, kernel_regularization
from data import load_nc_to_5band, sample_patches


def main():
    # 仅支持 nc 输入的无监督 KernelGAN 训练（5 波段）
    use_cpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'Using device: {device}')

    # 必填：nc 文件路径（请修改为你的实际路径）
    nc_path = '/Users/zy/Python_code/My_Git/match_cor/output/img/4_landmasked/GK2_GOCI2_L1B_20210330_021530_LA_S007_subset_footprint_landmasked.nc'
    print(f'Using nc file: {nc_path}')
    # 训练配置
    iters = 3000
    patch_size = 64
    batch_size = 8
    lr_rate = 1e-4
    outdir = './kernelgan_out'
    log_every = 100              # 普通训练日志间隔
    kernel_log_every = 300       # 核详细统计输出间隔
    mini_log_every = 10          # 轻量日志（更频繁）
    save_intermediate = True     # 是否保存中间核
    verbose = True               # 是否输出权重/梯度等详细信息

    # 读取 5 波段图 [5,H,W] 和有效像素掩码 [H,W]
    if not (nc_path and os.path.exists(nc_path)):
        print('错误：请在 train.py 中设置有效的 nc_path')
        return
    img, valid_mask = load_nc_to_5band(nc_path)
    img = img.to(device)
    valid_mask = valid_mask.to(device)
    print(f'影像尺寸: {tuple(img.shape)}, 有效像素比例: {valid_mask.float().mean():.2%}')

    # 模型（生成器输入 5 通道）
    G = LinearGenerator(in_ch=5, mid_ch=64).to(device)
    D = PatchDiscriminator(in_ch=1, base_ch=64, num_blocks=4).to(device)

    opt_D = optim.Adam(D.parameters(), lr=lr_rate, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=lr_rate, betas=(0.5, 0.999))

    def kernel_metrics(k: torch.Tensor) -> dict:
        """计算核的多项统计指标用于监控。k: [kH,kW] (sum≈1)。"""
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
        """将核缩放到 size x size 并转为 ASCII 字符块，便于快速目视检查集中度。"""
        import torch.nn.functional as F
        k2 = k.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        k2 = F.interpolate(k2, size=(size, size), mode='bilinear', align_corners=False)
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
        vals = []
        for i, conv in enumerate(G.convs):
            w = conv.weight.detach()
            vals.append(f"L{i}(norm={w.norm().item():.3f},max={w.max().item():.3f})")
        return ' '.join(vals)

    prev_k = None  # 用于计算核变化幅度

    for t in range(iters):
        # 采样补丁 [B,5,P,P]（避开无效区域）
        patches = sample_patches(img, patch_size=patch_size, batch_size=batch_size, valid_mask=valid_mask)
        # 真实下采样（对 5 通道取均值后下采样为单通道）与生成器下采样（G输出就是下采样）
        real_ds_5ch = torch.nn.functional.avg_pool2d(patches, kernel_size=2, stride=2)  # [B,5,P/2,P/2]
        real_ds = real_ds_5ch.mean(dim=1, keepdim=True)  # [B,1,P/2,P/2] 跨波段平均为单通道
        fake_ds = G(patches)  # [B,1,P/2,P/2]

        # 训练D
        D.train(); G.train()
        pred_real = D(real_ds)
        pred_fake = D(fake_ds.detach())
        loss_D = lsgan_d_loss(pred_real, pred_fake)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # 训练G（对抗 + 核正则）
        pred_fake = D(fake_ds)
        loss_G_adv = lsgan_g_loss(pred_fake)
        k = G.extract_effective_kernel()
        loss_reg = kernel_regularization(k, alpha=0.5, beta=0.5, gamma=5.0, delta=1.0)
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
            if verbose:
                print("  [Kernel ASCII]\n" + ascii_kernel(k))
            if save_intermediate:
                os.makedirs(outdir, exist_ok=True)
                np.save(os.path.join(outdir, f'kernel_iter{t+1}.npy'), k.cpu().numpy())

    # 提取并保存最终核
    k = G.extract_effective_kernel().cpu().numpy()
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'kernel.npy'), k)
    print(f"Saved kernel to {os.path.join(outdir, 'kernel.npy')} | sum={k.sum():.6f}")


if __name__ == "__main__":
    main()
