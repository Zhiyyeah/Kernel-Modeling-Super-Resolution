"""
批量去噪处理脚本
扫描指定文件夹中的所有nc文件并调用denoise.py进行批量去噪处理
"""

from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 导入denoise.py中的处理函数
from denoise import process_nc_file


def main():
    parser = argparse.ArgumentParser(description="批量去噪处理工具")
    parser.add_argument("input_dir", type=str, 
                        help="输入文件夹路径（包含nc文件）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件夹路径（默认: 输入文件夹_denoised）")
    parser.add_argument("--h_factor", type=float, default=1.8,
                        help="去噪强度因子 (默认: 1.8, 推荐范围: 1.0-2.5)")
    parser.add_argument("--pattern", type=str, default="*.nc",
                        help="文件匹配模式 (默认: *.nc)")
    parser.add_argument("--verbose", action="store_true",
                        help="详细模式，显示每个文件的处理信息 (默认: 只显示进度条)")
    args = parser.parse_args()
    
    # 设置路径
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_denoised"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有nc文件
    nc_files = list(input_dir.glob(args.pattern))
    
    if not nc_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 '{args.pattern}' 的文件")
        return
    
    # 显示全局信息（无论是否verbose都显示）
    print(f"\n{'='*70}")
    print(f"批量去噪处理")
    print(f"{'='*70}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文件总数: {len(nc_files)}")
    print(f"去噪参数: h_factor={args.h_factor}, patch_size=7, patch_distance=11")
    print(f"{'='*70}\n")
    
    # 批量处理
    success_count = 0
    failed_files = []
    
    verbose = args.verbose
    
    if not verbose:
        # 静默模式：只显示进度条
        for nc_file in tqdm(nc_files, desc="去噪处理", unit="文件"):
            success, output_path, error = process_nc_file(
                file_path=nc_file,
                output_dir=output_dir,
                h_factor=args.h_factor,
                plot=False,
                verbose=False
            )
            if success:
                success_count += 1
            else:
                failed_files.append((nc_file.name, error))
    else:
        # 显示详细信息
        for i, nc_file in enumerate(nc_files, 1):
            print(f"\n[{i}/{len(nc_files)}]")
            success, output_path, error = process_nc_file(
                file_path=nc_file,
                output_dir=output_dir,
                h_factor=args.h_factor,
                plot=False,
                verbose=True
            )
            if success:
                success_count += 1
            else:
                failed_files.append((nc_file.name, error))
    
    # 输出统计信息
    print(f"\n{'='*70}")
    print(f"处理完成!")
    print(f"{'='*70}")
    print(f"成功: {success_count}/{len(nc_files)}")
    print(f"失败: {len(failed_files)}/{len(nc_files)}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for filename, error in failed_files:
            print(f"  - {filename}")
            if error and verbose:
                print(f"    错误: {error}")
    
    print(f"\n输出目录: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
