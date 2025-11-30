import torch
import torch.nn.functional as F


def lsgan_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN判别器损失：0.5 * [ (D(real)-1)^2 + (D(fake)-0)^2 ]，按均值聚合。"""
    loss_real = 0.5 * torch.mean((pred_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean((pred_fake - 0.0) ** 2)
    return loss_real + loss_fake


def lsgan_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN生成器损失：0.5 * (D(fake)-1)^2，按均值聚合。"""
    return 0.5 * torch.mean((pred_fake - 1.0) ** 2)


def kernel_regularization(
    k: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 5.0,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    四项核正则化损失：Sum-to-1、Boundaries、Sparse、Center。
    - k: [kH,kW] 单核（非负且和约为1）。
    - 参数建议：alpha=0.5, beta=0.5, gamma=5, delta=1
    返回标量loss。
    """
    kH, kW = k.shape
    # Sum-to-1
    sum1 = (k.sum() - 1.0) ** 2
    # Boundaries：边框元素惩罚
    top = k[0, :].pow(2).sum()
    bottom = k[-1, :].pow(2).sum()
    left = k[:, 0].pow(2).sum()
    right = k[:, -1].pow(2).sum()
    boundaries = top + bottom + left + right
    # Sparse：0.5次方求和（鼓励稀疏）
    sparse = torch.sqrt(torch.clamp(k, min=0)).sum()
    # Center：重心到中心的距离平方
    yy, xx = torch.meshgrid(torch.arange(kH, device=k.device), torch.arange(kW, device=k.device), indexing='ij')
    mass = torch.clamp(k, min=0) + 1e-12
    cy = (yy.float() * mass).sum() / mass.sum()
    cx = (xx.float() * mass).sum() / mass.sum()
    center_y = (kH - 1) / 2.0
    center_x = (kW - 1) / 2.0
    center = (cy - center_y) ** 2 + (cx - center_x) ** 2

    return alpha * sum1 + beta * boundaries + gamma * sparse + delta * center


if __name__ == "__main__":
    # 简易测试
    k = torch.zeros(25, 25)
    k[12, 12] = 1.0
    loss = kernel_regularization(k)
    print(f"kernel reg loss: {loss.item():.4f}")
