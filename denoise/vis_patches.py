from os import read
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def read_nc(file_path:str, group_name, band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']):
    data = []
    for band_name in band_names:
        band_data = xr.open_dataset(file_path, group=group_name)[band_name]
        data.append(band_data)
        data_con=xr.concat(data, dim='band')
        data_con = data_con.where(data_con != 0, np.nan)
    return data_con 

def main() -> None:
	parser = argparse.ArgumentParser(description="Read a .npy file and print basic info")
	parser.add_argument("npy_path", type=Path, help="Path to .npy file")
	args = parser.parse_args()

	path = args.npy_path
	print(f"Loading: {path}")
	
    #读取文件
	arr = read_nc(path, 'denoised')
	min_val = float(arr.min().values)
	max_val = float(arr.max().values)
	print(f"Min-Max: {min_val:.3f} - {max_val:.3f}")
	bands, height, width = arr.shape
	print(f"Bands: {bands}, Height: {height}, Width: {width}")
	first_band = arr[0]
	print(f"First band shape: {first_band.shape}")
    
    #画图
	plt.figure(figsize=(8, 6))
	plt.imshow(first_band, cmap='viridis')
	plt.title("First Band")
	plt.colorbar()
	# plt.show()
	plt.savefig("denoise/first_band.png", dpi=300)



if __name__ == "__main__":
	main()