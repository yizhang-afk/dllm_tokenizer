#!/usr/bin/env python3
"""
数据预处理脚本
用于准备训练 BPE tokenizer 的语料库
支持大规模数据（>10GB）的分块处理
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Iterator, List
import psutil
from tqdm import tqdm


def check_encoding(file_path: str) -> bool:
    """检查文件是否为 UTF-8 编码"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def convert_to_utf8(file_path: str, output_path: str, encoding: str = 'latin-1'):
    """将文件转换为 UTF-8 编码"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"转换失败 {file_path}: {e}")
        return False


def get_file_lines(file_path: str) -> int:
    """快速统计文件行数"""
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def deduplicate_lines(input_files: List[str], output_file: str,
                     min_length: int = 10, max_length: int = 10000):
    """
    去重并过滤文本行

    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
        min_length: 最小行长度
        max_length: 最大行长度
    """
    seen = set()
    total_lines = 0
    duplicate_lines = 0
    filtered_lines = 0

    # 统计总行数用于进度条
    print("统计文件行数...")
    total_count = sum(get_file_lines(f) for f in input_files)

    print(f"开始处理 {len(input_files)} 个文件，共 {total_count:,} 行...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with tqdm(total=total_count, desc="处理进度") as pbar:
            for input_file in input_files:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        pbar.update(1)
                        total_lines += 1

                        # 去除首尾空白
                        line = line.strip()

                        # 过滤空行和过长/过短的行
                        if not line or len(line) < min_length or len(line) > max_length:
                            filtered_lines += 1
                            continue

                        # 去重
                        if line in seen:
                            duplicate_lines += 1
                            continue

                        seen.add(line)
                        out_f.write(line + '\n')

    unique_lines = total_lines - duplicate_lines - filtered_lines
    print(f"\n处理完成:")
    print(f"  总行数: {total_lines:,}")
    print(f"  去重行数: {duplicate_lines:,}")
    print(f"  过滤行数: {filtered_lines:,}")
    print(f"  保留行数: {unique_lines:,}")
    print(f"  输出文件: {output_file}")


def merge_files(input_files: List[str], output_file: str,
               chunk_size: int = 1000000):
    """
    合并多个文件（适合大规模数据）

    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
        chunk_size: 每次读取的行数
    """
    print(f"合并 {len(input_files)} 个文件到 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in tqdm(input_files, desc="合并文件"):
            with open(input_file, 'r', encoding='utf-8') as in_f:
                while True:
                    lines = []
                    for _ in range(chunk_size):
                        line = in_f.readline()
                        if not line:
                            break
                        lines.append(line)

                    if not lines:
                        break

                    out_f.writelines(lines)

    print(f"合并完成: {output_file}")


def monitor_memory():
    """监控内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024 / 1024 / 1024
    return mem_gb


def validate_files(file_paths: List[str]):
    """验证文件存在性和编码"""
    print("验证输入文件...")
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not check_encoding(file_path):
            print(f"警告: {file_path} 不是 UTF-8 编码")
            response = input("是否尝试转换? (y/n): ")
            if response.lower() == 'y':
                backup_path = file_path + '.backup'
                os.rename(file_path, backup_path)
                if convert_to_utf8(backup_path, file_path):
                    print(f"已转换 {file_path} 为 UTF-8")
                else:
                    os.rename(backup_path, file_path)
                    raise ValueError(f"无法转换 {file_path} 为 UTF-8")

    print("所有文件验证通过")


def main():
    parser = argparse.ArgumentParser(description='预处理训练数据')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='输入文件路径（可多个）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出文件路径')
    parser.add_argument('--deduplicate', action='store_true',
                       help='是否去重')
    parser.add_argument('--min-length', type=int, default=10,
                       help='最小行长度（默认10）')
    parser.add_argument('--max-length', type=int, default=10000,
                       help='最大行长度（默认10000）')
    parser.add_argument('--merge-only', action='store_true',
                       help='仅合并文件，不去重')

    args = parser.parse_args()

    # 验证文件
    validate_files(args.input)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"初始内存使用: {monitor_memory():.2f} GB")

    # 处理数据
    if args.merge_only:
        merge_files(args.input, args.output)
    elif args.deduplicate:
        deduplicate_lines(args.input, args.output,
                         args.min_length, args.max_length)
    else:
        # 默认：合并
        merge_files(args.input, args.output)

    print(f"最终内存使用: {monitor_memory():.2f} GB")
    print("数据预处理完成!")


if __name__ == '__main__':
    main()
