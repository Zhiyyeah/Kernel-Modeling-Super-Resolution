import re
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


def _load_band_values_from_group(file_path: str, pattern: str = "L_TOA_490"):
    """
    从 geophysical_data 组中提取匹配 pattern 的波段值并展平成一维数组。
    pattern 使用正则匹配变量名，只要包含 pattern 即视为匹配。
    返回: (values, matched_var_names)
    """
    dataset = nc.Dataset(file_path, "r")
    try:
        if "geophysical_data" not in dataset.groups:
            raise KeyError("文件中找不到 geophysical_data 组")

        group = dataset.groups["geophysical_data"]
        matched_vars = [name for name in group.variables.keys() if re.search(pattern, name)]
        if not matched_vars:
            raise KeyError(f"geophysical_data 中未找到匹配 '{pattern}' 的变量")

        all_values = []
        for name in matched_vars:
            data = np.ma.asarray(group.variables[name][:])
            flattened = data.compressed() if np.ma.isMaskedArray(data) else data.ravel()
            clean = flattened[np.isfinite(flattened)]  # 去掉 NaN/Inf
            all_values.append(clean)

        merged = np.concatenate(all_values)
        return merged, matched_vars
    finally:
        dataset.close()


def plot_490_hist(goci_path: str, landsat_path: str, bins: int = 90, save_path: str | None = None):
    """
    读取两个文件的 490 波段，绘制叠加直方图。
    """
    goci_values, goci_vars = _load_band_values_from_group(goci_path, pattern="L_TOA_490")
    landsat_values, landsat_vars = _load_band_values_from_group(landsat_path, pattern="L_TOA_490")

    # 统一分箱范围：按分位数裁剪，避免少量异常值拉伸坐标
    both = np.concatenate([goci_values, landsat_values])
    q1, q99 = np.nanpercentile(both, [0.0001, 99.99])
    # 可选过滤非正值（若数据应为正辐亮度）
    lower = max(q1, 0)
    upper = q99

    # 裁剪到共享范围
    goci_clip = goci_values[(goci_values >= lower) & (goci_values <= upper)]
    landsat_clip = landsat_values[(landsat_values >= lower) & (landsat_values <= upper)]

    # 使用统一的 bin edges，确保两者柱子对齐
    bin_edges = np.linspace(lower, upper, bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.hist(goci_clip, bins=bin_edges, alpha=0.6, label="GOCI", color="#5b8db8")
    ax.hist(landsat_clip, bins=bin_edges, alpha=0.6, label="Landsat", color="#e39b64")

    title_var = goci_vars[0] if goci_vars else "L_TOA_490"
    ax.set_title(f"L_TOA_490_normalized")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
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
    goci_file = r"D:\Py_Code\img_match_cor\output\norm\batch\GK2_GOCI2_L1B_20210330_021530_LA_S007_subset_footprint_to_landsat_grid_landmasked_norm.nc"
    landsat_file = r"D:\Py_Code\img_match_cor\output\norm\batch\LC08_L1TP_116036_20210330_20210409_02_T1_TOA_RAD_B1-2-3-4-5_native_landmasked_norm.nc"

    plot_490_hist(goci_file, landsat_file, bins=90, save_path="hist_490.png")
