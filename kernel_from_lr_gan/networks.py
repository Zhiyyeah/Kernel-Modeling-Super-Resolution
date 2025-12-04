import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class MultiBandLinearGenerator(nn.Module):
    """
    多波段独立深度线性生成器：每个输入波段拥有一套独立的线性卷积链（无激活），
    允许每个波段对应不同的模糊核。对每个波段的特征做平均池化下采样后，
    直接保留 5 通道输出，不再跨波段求平均。这样判别器可同时看到各波段退化差异。

    每条链的卷积核尺寸顺序： [7, 5, 3, 1, 1, 1]

    核提取：逐条链合成其等效核，返回形状 [C, kH, kW] 的核列表；合并核仍提供跨波段平均视角。

    参数:
        in_ch (int): 输入通道数，即波段数量。默认 5（对应 5 个光谱波段）。
        mid_ch (int): 每条卷积链的中间层通道数。默认 32。
    """
    def __init__(self, in_ch: int = 5, mid_ch: int = 32):
        super().__init__()
        self.in_ch = in_ch
        ks = [7, 5, 3, 1, 1, 1]
        chains = []
        for _ in range(in_ch):
            convs = nn.ModuleList()
            # 第一层：1 -> mid_ch
            convs.append(nn.Conv2d(in_channels=1, out_channels=mid_ch, kernel_size=ks[0], stride=1, padding=ks[0]//2, bias=False))
            # 中间层：mid_ch -> mid_ch
            for ksize in ks[1:-1]:
                convs.append(nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=ksize, stride=1, padding=ksize//2, bias=False))
            # 最后一层：mid_ch -> 1
            convs.append(nn.Conv2d(in_channels=mid_ch, out_channels=1, kernel_size=ks[-1], stride=1, padding=ks[-1]//2, bias=False))
            chains.append(convs)
        self.chains = nn.ModuleList(chains)
        # 8倍下采样：3个2倍下采样级联 (2 * 2 * 2 = 8)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对每个波段独立应用线性卷积链，再进行8倍下采样。
        参数:
            x (torch.Tensor): 输入图像，形状 [B, C, H, W]，其中:
                - B: batch size
                - C: 波段数（通道数），与 in_ch 一致
                - H, W: 图像高度和宽度
        返回:
            torch.Tensor: 8倍下采样后的 5 通道输出，形状 [B, 5, H/8, W/8]。
        """
        # x: [B,C,H,W]
        outs = []
        for c in range(self.in_ch):
            h = x[:, c:c+1]
            for conv in self.chains[c]:
                h = conv(h)
            # 8倍下采样：连续进行3次2倍下采样
            h = self.down1(h)  # [B,1,H/2,W/2]
            h = self.down2(h)  # [B,1,H/4,W/4]
            h = self.down3(h)  # [B,1,H/8,W/8]
            outs.append(h)
        # 保留 5 通道输出: [B,5,H/8,W/8]
        return torch.cat(tensors=outs, dim=1)

    @torch.no_grad()
    def extract_effective_kernels(self) -> torch.Tensor:
        """
        提取每个波段的等效模糊核（通过逐层权重卷积合成）。

        返回:
            torch.Tensor: 每个波段的核，形状 [C, kH, kW]，其中:
                - C: 波段数（与 in_ch 一致）
                - kH, kW: 等效核的高度和宽度（通常为 13x13）
                每个核已归一化（非负且和为 1）。
        """
        kernel_list = []
        for convs in self.chains:
            weights = [conv.weight.detach().clone() for conv in convs]
            # 第一层: [mid_ch,1,k,k]
            K_cur = weights[0]

            def conv_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                a = A.unsqueeze(dim=0).unsqueeze(dim=0)
                b = torch.flip(input=B, dims=[0,1]).unsqueeze(dim=0).unsqueeze(dim=0)
                pad_h = b.shape[-2] - 1
                pad_w = b.shape[-1] - 1
                y = F.conv2d(input=a, weight=b, padding=(pad_h, pad_w))
                return y.squeeze(dim=0).squeeze(dim=0)

            for W in weights[1:]:
                C_out, C_mid, kH, kW = W.shape
                _, C_in, _, _ = K_cur.shape
                K_next = []
                for co in range(C_out):
                    row = []
                    for ci in range(C_in):
                        acc = None
                        for cm in range(C_mid):
                            kA = W[co, cm]
                            kB = K_cur[cm, ci]
                            kk = conv_kernel(kA, kB)
                            acc = kk if acc is None else acc + kk
                        row.append(acc)
                    row = torch.stack(tensors=row, dim=0)
                    K_next.append(row)
                K_cur = torch.stack(tensors=K_next, dim=0)
            k = K_cur.mean(dim=(0,1))
            k = torch.clamp(input=k, min=0)
            s = k.sum()
            if s <= 1e-12:
                s = torch.tensor(data=1.0, device=k.device)
            k = k / s
            kernel_list.append(k)
        return torch.stack(tensors=kernel_list, dim=0)  # [C,kH,kW]

    @torch.no_grad()
    def extract_merged_kernel(self) -> torch.Tensor:
        """
        提取跨波段平均的合并核（用于快速查看整体退化）。
        返回:
            torch.Tensor: 合并后的单个核，形状 [kH, kW]，
                对所有波段核取平均得到，已归一化（和为 1）。
        """
        ks = self.extract_effective_kernels()  # [C,kH,kW]
        return ks.mean(dim=0)


class PatchDiscriminator(nn.Module):
    """
    全卷积 Patch 判别器：
    - 第一层 7x7 卷积（spectral_norm），输入多通道
    - 后续若干 1x1 卷积（spectral_norm + BN + LeakyReLU）
    - 无池化；输出为 [B,1,H,W] 的得分热图

    参数:
        in_ch (int): 输入通道数。默认 1（单通道），多波段训练时设为 5。
        base_ch (int): 第一层输出的基础通道数。默认 64。
        num_blocks (int): 1x1 卷积块的数量。默认 4。
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, num_blocks: int = 4):
        super().__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=7, stride=1, padding=3)))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ch = base_ch
        for _ in range(num_blocks):
            layers.append(spectral_norm(nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, padding=0)))
            layers.append(nn.BatchNorm2d(num_features=ch))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, stride=1, padding=0)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入图像进行判别。

        参数:
            x (torch.Tensor): 输入图像，形状 [B, in_ch, H, W]。

        返回:
            torch.Tensor: 判别得分热图，形状 [B, 1, H, W]，
                每个空间位置的值表示该 patch 的真实性得分。
        """
        return self.net(x)


if __name__ == "__main__":
    # 自测：多波段生成器与 5 通道输入的判别器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = MultiBandLinearGenerator(in_ch=5).to(device)
    D = PatchDiscriminator(in_ch=5).to(device)
    x = torch.rand(20, 5, 64, 64, device=device)
    y = G(x)              # [20,5,8,8] (64/8=8)
    s = D(y)              # [20,1,8,8]
    ks = G.extract_effective_kernels()  # [5,kH,kW]
    km = G.extract_merged_kernel()      # [kH,kW]
    print(f"G out: {tuple(y.shape)} | D out: {tuple(s.shape)} | kernels: {tuple(ks.shape)} merged: {tuple(km.shape)} sum(merged)={km.sum().item():.4f}")
