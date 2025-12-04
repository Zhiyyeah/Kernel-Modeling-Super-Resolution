"""
短期训练测试脚本：运行200次迭代，测试新的正则化权重
"""
import torch
import numpy as np
import os
import sys
sys.path.insert(0, '.')
from train import (
    load_patches_from_folder, sample_patches_from_files, 
    MultiBandLinearGenerator, PatchDiscriminator,
    lsgan_d_loss, lsgan_g_loss, kernel_regularization
)

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 加载patch文件
patch_dir = '/Users/zy/Downloads/GOCI-2/patches_all'
patch_files, original_size = load_patches_from_folder(patch_dir)
print(f'找到 {len(patch_files)} 个patch文件，原始尺寸: {original_size}')

# 创建模型
G = MultiBandLinearGenerator(in_ch=5, mid_ch=32)
D = PatchDiscriminator(in_ch=5)
G.to(device)
D.to(device)

# 优化器
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))

# 短期训练（200 iterations）
iters = 200
batch_size = 8

print(f'\n开始训练：{iters} iterations，batch_size={batch_size}')
print('=' * 60)

for t in range(iters):
    # 非配对训练
    patches_hr = sample_patches_from_files(patch_files, batch_size=batch_size, target_size=128, 
                                          original_size=original_size, device=device)
    patches_lr = sample_patches_from_files(patch_files, batch_size=batch_size, target_size=16,
                                          original_size=original_size, device=device)
    
    patches_hr = patches_hr.to(device)
    patches_lr = patches_lr.to(device)
    
    # 生成低分辨率样本（通过模糊核）
    fake_ds = G(patches_hr)
    
    # 训练D
    D.train(); G.train()
    pred_real = D(patches_lr)
    pred_fake = D(fake_ds.detach())
    loss_D = lsgan_d_loss(pred_real, pred_fake)
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    
    # 训练G（注意：正则化权重改为0.01）
    pred_fake = D(fake_ds)
    loss_G_adv = lsgan_g_loss(pred_fake)
    ks_band = G.extract_effective_kernels()
    reg_list = []
    for i in range(ks_band.shape[0]):
        reg_list.append(kernel_regularization(k=ks_band[i], alpha=0.5, beta=0.5, gamma=5.0, delta=1.0))
    loss_reg = torch.mean(torch.stack(reg_list))
    k = ks_band.mean(dim=0)
    loss_G = loss_G_adv + 0.01 * loss_reg  # 新的权重
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    
    # 输出日志
    if (t + 1) % 50 == 0:
        km_sum = k.sum().item()
        km_max = k.max().item()
        km_min = k.min().item()
        print(f'Iter {t+1:3d}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | Reg: {loss_reg.item():.4f} | K_sum: {km_sum:.4f} | K_max: {km_max:.4f}')

# 检查最终核
print('\n最终核统计：')
ks_final = G.extract_effective_kernels()
k_final = ks_final.mean(dim=0)
km_sum = k_final.sum().item()
km_max = k_final.max().item()
km_min = k_final.min().item()
print(f'  形状: {ks_final.shape}')
print(f'  总和: {km_sum:.6f}')
print(f'  最大值: {km_max:.6f}')
print(f'  最小值: {km_min:.6f}')

# 保存核用于可视化
outdir = '/Users/zy/Python_code/My_Git/Kernel-Modeling-Super-Resolution/kernelgan_test'
os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, 'kernel_test.npy'), k_final.cpu().numpy())
np.save(os.path.join(outdir, 'kernel_per_band_test.npy'), ks_final.cpu().numpy())
print(f'\n核已保存到: {outdir}')
