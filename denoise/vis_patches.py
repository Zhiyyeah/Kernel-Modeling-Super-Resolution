from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
	parser = argparse.ArgumentParser(description="Read a .npy file and print basic info")
	parser.add_argument("npy_path", type=Path, help="Path to .npy file", default=r"H:\\GOCI-2\\patches_all\\GK2_GOCI2_L1B_20250103_061530_LA_S010_001_012.npy")
	args = parser.parse_args()

	path = args.npy_path
	print(f"Loading: {path}")
	arr = np.load(path)

	print(f"Shape: {arr.shape}")  # expected (5, 256, 256)
	print(f"Dtype: {arr.dtype}")
	print(f"Min/Max: {arr.min()} / {arr.max()}")
	bands, height, width = arr.shape
	print(f"Bands: {bands}, Height: {height}, Width: {width}")
	first_band = arr[0]
	print(f"First band shape: {first_band.shape}")
	plt.figure(figsize=(8, 6))
	plt.imshow(first_band, cmap='viridis')
	plt.title("First Band")
	plt.colorbar()
	# plt.show()
	plt.savefig("denoise/first_band.png", dpi=300)



if __name__ == "__main__":
	main()