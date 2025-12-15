import re
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


def _load_band_from_group(dataset: nc.Dataset, group_name: str, pattern: str = "L_TOA_490"):
    """
    从指定组中提取匹配 pattern 的变量值并展平成一维数组。
    返回: (values, matched_var_names)
    """
    if group_name not in dataset.groups:
        raise KeyError(f"文件中找不到组: {group_name}")

    group = dataset.groups[group_name]
    matched_vars = [name for name in group.variables.keys() if re.search(pattern, name)]
    if not matched_vars:
        raise KeyError(f"组 '{group_name}' 中未找到匹配 '{pattern}' 的变量")

    all_values = []
    for name in matched_vars:
        data = np.ma.asarray(group.variables[name][:])
        flattened = data.compressed() if np.ma.isMaskedArray(data) else data.ravel()
        clean = flattened[np.isfinite(flattened)]
        all_values.append(clean)

    merged = np.concatenate(all_values)
    return merged, matched_vars


def plot_490_hr_lr(nc_path: str, bins: int = 90, save_path: str | None = None):
    """
    从一个 NetCDF 文件中读取 hr 和 lr 组的 490 波段并绘制直方图对比。
    """
    dataset = nc.Dataset(nc_path, "r")
    try:
        hr_values, hr_vars = _load_band_from_group(dataset, "hr", pattern="L_TOA_490")
        lr_values, lr_vars = _load_band_from_group(dataset, "lr", pattern="L_TOA_490")
    finally:
        dataset.close()

    # 统一分箱范围：按分位数裁剪，避免少量异常值拉伸坐标
    both = np.concatenate([hr_values, lr_values])
    q1, q99 = np.nanpercentile(both, [1, 99])
    lower = max(q1, 0)
    upper = q99

    # 裁剪到共享范围
    hr_clip = hr_values[(hr_values >= lower) & (hr_values <= upper)]
    lr_clip = lr_values[(lr_values >= lower) & (lr_values <= upper)]

    # 统一的 bin edges
    bin_edges = np.linspace(lower, upper, bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    # 使用密度显示，消除像元数量差异的影响
    ax.hist(hr_clip, bins=bin_edges, alpha=0.6, label="HR", color="#5b8db8", density=True)
    ax.hist(lr_clip, bins=bin_edges, alpha=0.6, label="LR", color="#e39b64", density=True)

    title_var = hr_vars[0] if hr_vars else "L_TOA_490"
    ax.set_title(f"Histogram — {title_var} (HR vs LR)")
    ax.set_xlabel("Value (original)")
    ax.set_ylabel("Density")
    ax.set_xlim(lower, upper)
    ax.legend()
    ax.grid(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"已保存图像: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    nc_file = r"H:\Landsat\patches_all\LC08_L1TP_116034_20240407_20240412_02_T1_TOA_RAD_B1-2-3-4-5_native_042_005.nc"
    plot_490_hr_lr(nc_file, bins=90, save_path="hist_490_hr_lr.png")
