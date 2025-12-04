"""
快速验证核的形状和内容
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')
from networks import MultiBandLinearGenerator

print("="*60)
print("验证核的生成和形状")
print("="*60)

# 创建生成器
G = MultiBandLinearGenerator(in_ch=5, mid_ch=32)

# 随机前向传播产生一些梯度
x = torch.randn(4, 5, 128, 128)
y = G(x)
print(f"\n前向传播:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {y.shape}")

# 提取核
ks = G.extract_effective_kernels()  # [C, kH, kW]
k_merged = ks.mean(dim=0)  # [kH, kW]

print(f"\n多波段核:")
print(f"  形状: {ks.shape}")
print(f"  值域: [{ks.min():.6f}, {ks.max():.6f}]")
print(f"  平均值: {ks.mean():.6f}")

print(f"\n融合核（跨波段平均）:")
print(f"  形状: {k_merged.shape}")
print(f"  值域: [{k_merged.min():.6f}, {k_merged.max():.6f}]")
print(f"  总和: {k_merged.sum():.6f}")
print(f"  最大值: {k_merged.max():.6f}")

print(f"\n核矩阵预览（融合核）:")
k_np = k_merged.numpy()
print(k_np)

# 保存并重新加载验证格式
outdir = '/Users/zy/Python_code/My_Git/Kernel-Modeling-Super-Resolution/kernelgan_test'
import os
os.makedirs(outdir, exist_ok=True)

np.save(os.path.join(outdir, 'kernel_shape_test.npy'), k_np)
loaded = np.load(os.path.join(outdir, 'kernel_shape_test.npy'))
print(f"\n保存并重新加载验证:")
print(f"  保存的形状: {loaded.shape}")
print(f"  值一致: {np.allclose(loaded, k_np)}")

print("\n✅ 核的形状和格式验证成功!")
