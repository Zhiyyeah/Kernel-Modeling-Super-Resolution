"""
数据加载与补丁采样模块
- 输入: 单张图片路径（NetCDF .nc 文件）或 patch 文件夹路径
- 输出: 批量的 Image Patches (Tensor) [B, 5, patch_size, patch_size]（5波段）
"""
import torch
import torch.nn.functional as F
import numpy as np
from netCDF4 import Dataset
import os
import glob

def load_nc_to_5band(nc_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    读取 NetCDF 文件的 geophysical_data 组的五个波段（保留多通道）。

    参数:
        nc_path (str): NetCDF 文件路径（.nc 文件）。

    返回:
        tuple[torch.Tensor, torch.Tensor]: 
            - image: 形状 [5, H, W]，归一化后的 5 波段图像，值范围 [0, 1]。
            - valid_mask: 形状 [H, W]，有效像素掩码（True=有效，False=NaN区域）。

    处理流程:
        1. 读取 5 个波段变量（L_TOA_443/490/555/660/865_masked）。
        2. 创建有效像素掩码（标记 <= -9998.5 为无效）。
        3. 对每个波段做百分位归一化（1%-99%，仅对有效像素）。
        4. 返回 5 通道张量和掩码。
    """
    ds = Dataset(nc_path, 'r')
    grp = ds.groups.get('geophysical_data')
    # 读取五个波段
    names = ['L_TOA_443_masked','L_TOA_490_masked','L_TOA_555_masked','L_TOA_660_masked','L_TOA_865_masked']
    bands = []
    for name in names:
        if name not in grp.variables:
            raise KeyError(f'变量缺失: {name}')
        arr = grp.variables[name][:].astype(np.float32)
        bands.append(arr)
    ds.close()
    
    stack = np.stack(arrays=bands, axis=0)  # [5, H, W]
    
    # 创建有效像素掩码：所有波段都有效才认为该像素有效
    valid_mask = np.all(a=stack > -9998.5, axis=0)  # [H, W]
    
    # 对每个波段做百分位归一化到 [0,1]（仅对有效像素）
    for c in range(stack.shape[0]):
        band = stack[c]
        valid_values = band[valid_mask]
        if len(valid_values) > 0:
            vmin = np.percentile(valid_values, 0.01)
            vmax = np.percentile(valid_values, 99.99)
            if vmax <= vmin:
                vmax = vmin + 1.0
            stack[c] = np.clip(a=(band - vmin)/(vmax - vmin), a_min=0, a_max=1)
        else:
            stack[c] = 0.0  # 如果没有有效值，填充0（但会被掩码标记为无效）
    
    # 返回 5 通道张量 [5, H, W] 和掩码 [H, W]
    t = torch.from_numpy(ndarray=stack.astype(np.float32))
    mask = torch.from_numpy(ndarray=valid_mask)
    return t, mask


def gradient_weight_map(img: torch.Tensor, valid_mask: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """
    计算图像梯度幅值作为采样权重图（梯度大的区域采样概率更高）。

    参数:
        img (torch.Tensor): 输入图像，形状 [C, H, W]（支持多通道）。
        valid_mask (torch.Tensor, optional): 有效像素掩码，形状 [H, W]，True=有效。
            默认 None（全部有效）。
        eps (float): 避免梯度为0的小常数。默认 1e-6。

    返回:
        torch.Tensor: 形状 [H, W]，归一化的概率图（和为1，无效区域概率为0）。
    """
    # 计算水平和垂直梯度（跨所有通道）
    gx = img[:, :, 1:] - img[:, :, :-1]  # [C, H, W-1]
    gy = img[:, 1:, :] - img[:, :-1, :]  # [C, H-1, W]
    
    # 梯度幅值（补齐边界后在通道维度求平均）
    mag = F.pad(input=torch.sqrt(F.pad(input=gx, pad=(0,1))**2 + F.pad(input=gy, pad=(0,0,0,1))**2 + eps), pad=(0,0,0,0))
    p = mag.mean(dim=0)  # [H, W] 跨通道平均
    
    # 将无效区域的梯度权重置零
    if valid_mask is not None:
        p = p * valid_mask.float()
    
    # 归一化为概率分布
    p = p - p.min()
    s = p.sum()
    if s <= 0:
        # 如果所有权重为0（全是无效区域），则对有效区域均匀分布
        if valid_mask is not None:
            p = valid_mask.float() / valid_mask.float().sum().clamp(min=1.0)
        else:
            p = torch.ones_like(input=p) / p.numel()
    else:
        p = p / s
    return p


def sample_patches(img: torch.Tensor, patch_size: int, batch_size: int, valid_mask: torch.Tensor = None, max_tries: int = 1000) -> torch.Tensor:
    """
    从单张图像中按梯度权重随机裁剪补丁（KernelGAN 采样策略），避开无效区域。

    参数:
        img (torch.Tensor): 输入图像，形状 [C, H, W]，多通道图像（如 5 波段）。
        patch_size (int): 补丁边长（如 64）。
        batch_size (int): 一批采样的补丁数量（如 8）。
        valid_mask (torch.Tensor, optional): 有效像素掩码，形状 [H, W]，
            True=有效，False=NaN区域。默认 None（全部有效）。
        max_tries (int): 每个补丁的最大重试次数。默认 1000。

    返回:
        torch.Tensor: 形状 [B, C, patch_size, patch_size]，批量补丁（确保补丁内全部为有效像素）。

    采样策略:
        1. 计算梯度权重图（无效区域权重为0）。
        2. 边界区域（patch_size//2）置零，避免越界。
        3. 按概率采样坐标，采样后检查补丁是否全部有效，无效则重采样。
        4. 裁剪对应的补丁。
    """
    H, W = img.shape[-2], img.shape[-1]
    p = gradient_weight_map(img, valid_mask)  # [H, W] 概率图
    
    # 将边界区域的概率置零（避免裁剪时越界）
    pad = patch_size // 2
    grid = p.clone()
    grid[:pad, :] = 0
    grid[-pad:, :] = 0
    grid[:, :pad] = 0
    grid[:, -pad:] = 0
    
    # 重新归一化
    s = grid.sum()
    if s <= 0:
        raise ValueError("没有有效区域可供采样（可能全是边界或无效像素），请检查输入影像")
    grid = grid / s
    
    # 采样补丁（如果有掩码，确保采样到全有效的补丁）
    patches = []
    flat = grid.view(-1)
    
    for _ in range(batch_size):
        for attempt in range(max_tries):
            # 采样一个坐标
            idx = torch.multinomial(input=flat, num_samples=1, replacement=True).item()
            y = idx // W
            x = idx % W
            y0 = y - pad
            x0 = x - pad
            
            # 检查补丁是否全部有效
            if valid_mask is not None:
                patch_mask = valid_mask[y0:y0+patch_size, x0:x0+patch_size]
                if not patch_mask.all():
                    continue  # 补丁内有无效像素，重新采样
            
            # 采样成功
            patch = img[:, y0:y0+patch_size, x0:x0+patch_size]
            patches.append(patch)
            break
        else:
            raise ValueError(f"尝试{max_tries}次后仍无法采样到全有效补丁，请检查有效区域是否足够大")
    
    return torch.stack(tensors=patches, dim=0)  # [B, C, patch_size, patch_size]


def load_patches_from_folder(patch_dir: str, target_size: int = 64) -> tuple[list, int]:
    """
    从文件夹加载所有.npy格式的patch文件
    
    参数:
        patch_dir (str): patch文件夹路径
        target_size (int): 目标patch大小，如果patch大小不匹配会进行裁剪/调整，默认64
    
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


def sample_patches_from_files(patch_files: list, batch_size: int, target_size: int = 64, 
                               original_size: int = 128, device: torch.device = None) -> torch.Tensor:
    """
    从预先保存的patch文件中随机采样一批，并调整到目标尺寸
    
    参数:
        patch_files (list): patch文件路径列表
        batch_size (int): 批次大小
        target_size (int): 目标patch大小，默认64
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
        
        # 处理NaN值：替换为0（或者可以选择跳过这个patch）
        patch_tensor = torch.nan_to_num(patch_tensor, nan=0.0)
        
        # 归一化到[0,1]（如果数据还未归一化）
        # 检查是否需要归一化（假设如果有值>1则需要归一化）
        if patch_tensor.max() > 1.0:
            for c in range(patch_tensor.shape[0]):
                band = patch_tensor[c]
                valid_values = band[band > 0]  # 排除0值
                if len(valid_values) > 0:
                    vmin = valid_values.min()
                    vmax = valid_values.max()
                    if vmax > vmin:
                        patch_tensor[c] = torch.clip(input=(band - vmin) / (vmax - vmin), min=0, max=1)
        
        # 如果原始尺寸不等于目标尺寸，进行调整
        if original_size != target_size:
            # 使用双线性插值调整大小
            patch_tensor = F.interpolate(input=patch_tensor.unsqueeze(0), 
                                        size=(target_size, target_size), 
                                        mode='bilinear', 
                                        align_corners=False).squeeze(0)
        
        patches.append(patch_tensor)
    
    result = torch.stack(tensors=patches, dim=0)  # [B, 5, target_size, target_size]
    
    if device is not None:
        result = result.to(device)
    
    return result


if __name__ == "__main__":
    # 简易测试：从随机 5 通道图采样补丁，模拟带 NaN 区域
    img = torch.rand(5, 1000, 1999)
    # 创建模拟掩码：创建连续的有效区域（更真实）
    valid_mask = torch.ones(1000, 1999, dtype=torch.bool)
    # 在随机位置创建一些无效区域块
    for _ in range(20):
        y, x = torch.randint(0, 900, (1,)).item(), torch.randint(0, 1900, (1,)).item()
        h, w = torch.randint(10, 100, (1,)).item(), torch.randint(10, 100, (1,)).item()
        valid_mask[y:y+h, x:x+w] = False
    patches = sample_patches(img, patch_size=64, batch_size=8, valid_mask=valid_mask)
    print(f"patches: {tuple(patches.shape)}")
    print(f"有效区域比例: {valid_mask.float().mean():.2%}")
