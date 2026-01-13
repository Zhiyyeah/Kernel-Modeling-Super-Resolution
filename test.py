import numpy as np

arr= np.load("D:\Py_Code\Kernel-Modeling-Super-Resolution\denoise\denoise_results\L_TOA_490_clean.npy")
print(arr.shape)
print(f"arr.min() = {arr.min():.3f}, arr.max() = {arr.max():.3f}")