import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class LinearGenerator(nn.Module):
    """
    深度线性生成器（Deep Linear Network）：无激活函数。
    层核尺寸按顺序为 [7, 5, 3, 1, 1, 1]。输入为多通道补丁（如 5 波段），输出为单通道下采样。
    前向输出为下采样（平均池化，因子=2）后的补丁，用于与真实下采样分布匹配。

    同时提供提取当前等效模糊核的方法：将各层卷积核按顺序做卷积并在通道维度上相加，得到单一二维核。
    """
    def __init__(self, in_ch: int = 5, mid_ch: int = 64):
        super().__init__()
        ks = [7, 5, 3, 1, 1, 1]
        chs = [in_ch, mid_ch, mid_ch, mid_ch, mid_ch, mid_ch, 1]  # 6层卷积，最后输出1通道
        layers = []
        self.convs = nn.ModuleList()
        for i in range(6):
            self.convs.append(nn.Conv2d(chs[i], chs[i+1], ks[i], stride=1, padding=ks[i]//2, bias=False))
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h)  # 无激活，线性组合
        out = self.down(h)
        return out

    @torch.no_grad()
    def extract_effective_kernel(self) -> torch.Tensor:
        """
        计算当前网络的等效二维模糊核（单核）。将各层卷积核逐层卷积合并，并在中间通道上求和。
        返回形状为 [kH, kW] 的张量（归一化到和为1，非负剪裁到>=0）。
        注意：为简化实现，输入/输出为单通道，隐藏通道的组合通过对所有路径相加实现。
        """
        # 收集每层权重：list of [C_out, C_in, kH, kW]
        weights = [conv.weight.detach().clone() for conv in self.convs]
        # 初始等效核映射：上一层输出通道到输入通道的核。第一层即其权重。
        # K_cur: [C_out_l, C_in_0, kH, kW]
        K_cur = weights[0]  # [mid_ch, 1, k1, k1] 或 [1,1,k,k] 取决于配置

        def conv_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            """
            对二维核做真正的卷积（A * B）。A,B: [kH,kW]，返回大小为 kA+kB-1。
            使用 F.conv2d 实现 full convolution：input=A，filter=flip(B)，并在输入上做 (kB-1) 的零填充。
            """
            a = A.unsqueeze(0).unsqueeze(0)
            b = torch.flip(B, dims=[0, 1]).unsqueeze(0).unsqueeze(0)
            pad_h = b.shape[-2] - 1
            pad_w = b.shape[-1] - 1
            y = F.conv2d(a, b, padding=(pad_h, pad_w))
            return y.squeeze(0).squeeze(0)

        # 逐层合并：K_next[c_out, c_in] = sum_{c_mid} conv(W_next[c_out, c_mid], K_cur[c_mid, c_in])
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
                # 堆叠输入通道维度
                row = torch.stack(row, dim=0)  # [C_in, kH_eff, kW_eff]
                K_next.append(row)
            K_cur = torch.stack(K_next, dim=0)  # [C_out, C_in, kH_eff, kW_eff]

        # 将输出/输入通道映射聚合为单核（取平均）
        k = K_cur.mean(dim=(0, 1))  # [kH_eff, kW_eff]
        # 非负与归一化
        k = torch.clamp(k, min=0)
        s = k.sum()
        if s <= 1e-12:
            s = torch.tensor(1.0, device=k.device)
        k = k / s
        return k


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
    # 简易自测（5 通道输入）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = LinearGenerator(in_ch=5).to(device)
    D = PatchDiscriminator().to(device)
    x = torch.rand(2, 5, 64, 64, device=device)
    y = G(x)
    s = D(y)
    k = G.extract_effective_kernel()
    print(f"G out: {tuple(y.shape)} | D out: {tuple(s.shape)} | kernel: {tuple(k.shape)} sum={k.sum().item():.4f}")
