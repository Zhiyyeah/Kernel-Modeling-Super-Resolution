Kernel-from-LR-GAN

目的
- 条件GAN：输入NetCDF（nc）中的低分辨率影像，输出对应的模糊核（25×25）。
- 使用条件WGAN-GP提高稳定性；生成器输出做非负与归一化约束，保证物理合理性。

数据要求（nc格式）
- 组：`geophysical_data`
- 变量：`L_TOA_443_masked`, `L_TOA_490_masked`, `L_TOA_555_masked`, `L_TOA_660_masked`, `L_TOA_865_masked`
- 缺失值：以 `-9999` 表示，读取后置为 `0` 并做每波段的分位归一化到 `[0,1]`
- 通道：将5通道线性组合为3通道（可按需改为5通道编码器）

快速开始（macOS / zsh）
1) 安装依赖
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r training_code/kernel_from_lr_gan/requirements.txt
```

2) 推理（在脚本内直接设置参数）
```zsh
python3 training_code/kernel_from_lr_gan/infer.py
```
在 `infer.py` 中设置：
- `nc_path`：输入 nc 文件路径
- `generator_weights`、`encoder_weights`：训练权重
- `out_kernel`：输出核保存路径

说明
- 条件WGAN-GP：`loss_D = -(E[D(real)] - E[D(fake)]) + λ_gp * GP`，`loss_G = -E[D(fake)] + λ_sparse * mean(k)`
- 生成器输出经 `ReLU` 保证非负；再进行归一化（和为1）
- 如需保留5通道输入，更新 `LRFeatureEncoder(in_ch=5)` 并重新训练
