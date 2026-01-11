"""Minimal reader that loads a .npy file from an input path."""

from pathlib import Path
import argparse
import numpy as np


def main() -> None:
	parser = argparse.ArgumentParser(description="Read a .npy file and print basic info")
	parser.add_argument("npy_path", type=Path, help="Path to .npy file")
	args = parser.parse_args()

	path = args.npy_path
	print(f"Loading: {path}")
	arr = np.load(path)

	print(f"Shape: {arr.shape}")  # expected (5, 256, 256)
	print(f"Dtype: {arr.dtype}")
	print(f"Min/Max: {arr.min()} / {arr.max()}")
	bands, height, width = arr.shape
	print(f"Bands: {bands}, Height: {height}, Width: {width}")


if __name__ == "__main__":
	main()
