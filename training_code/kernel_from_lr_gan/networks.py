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
    """
    def __init__(self, in_ch: int = 5, mid_ch: int = 32):
        super().__init__()
        self.in_ch = in_ch
        ks = [7, 5, 3, 1, 1, 1]
        chains = []
        for _ in range(in_ch):
            convs = nn.ModuleList()
            # 第一层：1 -> mid_ch
            convs.append(nn.Conv2d(1, mid_ch, ks[0], 1, ks[0]//2, bias=False))
            # 中间层：mid_ch -> mid_ch
            for ksize in ks[1:-1]:
                convs.append(nn.Conv2d(mid_ch, mid_ch, ksize, 1, ksize//2, bias=False))
            # 最后一层：mid_ch -> 1
            convs.append(nn.Conv2d(mid_ch, 1, ks[-1], 1, ks[-1]//2, bias=False))
            chains.append(convs)
        self.chains = nn.ModuleList(chains)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        outs = []
        for c in range(self.in_ch):
            h = x[:, c:c+1]
            for conv in self.chains[c]:
                h = conv(h)
            h = self.down(h)  # [B,1,H/2,W/2]
            outs.append(h)
        # 保留 5 通道输出: [B,5,H/2,W/2]
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def extract_effective_kernels(self) -> torch.Tensor:
        """返回每个波段的等效核列表，形状 [C, kH, kW]。"""
        kernel_list = []
        for convs in self.chains:
            weights = [conv.weight.detach().clone() for conv in convs]
            # 第一层: [mid_ch,1,k,k]
            K_cur = weights[0]

            def conv_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                a = A.unsqueeze(0).unsqueeze(0)
                b = torch.flip(B, dims=[0,1]).unsqueeze(0).unsqueeze(0)
                pad_h = b.shape[-2] - 1
                pad_w = b.shape[-1] - 1
                y = F.conv2d(a, b, padding=(pad_h, pad_w))
                return y.squeeze(0).squeeze(0)

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
                    row = torch.stack(row, dim=0)
                    K_next.append(row)
                K_cur = torch.stack(K_next, dim=0)
            k = K_cur.mean(dim=(0,1))
            k = torch.clamp(k, min=0)
            s = k.sum()
            if s <= 1e-12:
                s = torch.tensor(1.0, device=k.device)
            k = k / s
            kernel_list.append(k)
        return torch.stack(kernel_list, dim=0)  # [C,kH,kW]

    @torch.no_grad()
    def extract_merged_kernel(self) -> torch.Tensor:
        """返回跨波段平均的合并核（用于快速查看整体退化）。"""
        ks = self.extract_effective_kernels()  # [C,kH,kW]
        return ks.mean(dim=0)


class PatchDiscriminator(nn.Module):
    """
    全卷积 Patch 判别器：
    - 第一层 7x7 卷积（spectral_norm），输入单通道
    - 后续若干 1x1 卷积（spectral_norm + BN + LeakyReLU）
    - 无池化；输出为 [B,1,H,W] 的得分热图
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, num_blocks: int = 4):
        super().__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_ch, base_ch, 7, 1, 3)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        ch = base_ch
        for _ in range(num_blocks):
            layers.append(spectral_norm(nn.Conv2d(ch, ch, 1, 1, 0)))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(spectral_norm(nn.Conv2d(ch, 1, 1, 1, 0)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # 自测：多波段生成器与 5 通道输入的判别器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = MultiBandLinearGenerator(in_ch=5).to(device)
    D = PatchDiscriminator(in_ch=5).to(device)
    x = torch.rand(20, 5, 64, 64, device=device)
    y = G(x)              # [20,5,32,32]
    s = D(y)              # [20,1,32,32]
    ks = G.extract_effective_kernels()  # [5,kH,kW]
    km = G.extract_merged_kernel()      # [kH,kW]
    print(f"G out: {tuple(y.shape)} | D out: {tuple(s.shape)} | kernels: {tuple(ks.shape)} merged: {tuple(km.shape)} sum(merged)={km.sum().item():.4f}")
