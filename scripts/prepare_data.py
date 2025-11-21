#!/usr/bin/env python3
"""
数据预处理脚本
用于准备训练 BPE tokenizer 的语料库
支持大规模数据（>10GB）的分块处理
支持 parquet 格式文件处理
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Iterator, List
import psutil
from tqdm import tqdm
import pandas as pd
import glob


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


def process_parquet_files(input_dir: str, output_file: str,
                         min_length: int = 10, max_length: int = 10000,
                         deduplicate: bool = False,
                         content_column: str = 'content'):
    """
    处理 parquet 文件并提取内容

    Args:
        input_dir: 包含 parquet 文件的目录
        output_file: 输出文件路径
        min_length: 最小行长度
        max_length: 最大行长度
        deduplicate: 是否去重
        content_column: 内容列名（默认为 'content'）
    """
    # 获取所有 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))

    if not parquet_files:
        raise ValueError(f"在 {input_dir} 中未找到 parquet 文件")

    print(f"找到 {len(parquet_files)} 个 parquet 文件")

    seen = set() if deduplicate else None
    total_lines = 0
    duplicate_lines = 0
    filtered_lines = 0
    written_lines = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for parquet_file in tqdm(parquet_files, desc="处理 parquet 文件"):
            try:
                # 使用迭代器逐块读取，避免内存溢出
                # 每次读取 10000 行
                for chunk in pd.read_parquet(parquet_file,
                                            columns=[content_column],
                                            engine='pyarrow',
                                            use_pandas_metadata=True):

                    # 处理每一行
                    for content in chunk[content_column]:
                        total_lines += 1

                        # 转换为字符串并去除首尾空白
                        if pd.isna(content):
                            filtered_lines += 1
                            continue

                        line = str(content).strip()

                        # 过滤空行和过长/过短的行
                        if not line or len(line) < min_length or len(line) > max_length:
                            filtered_lines += 1
                            continue

                        # 去重（如果启用）
                        if deduplicate:
                            if line in seen:
                                duplicate_lines += 1
                                continue
                            seen.add(line)

                        out_f.write(line + '\n')
                        written_lines += 1

                        # 定期报告内存使用
                        if written_lines % 100000 == 0:
                            mem_usage = monitor_memory()
                            tqdm.write(f"已处理 {written_lines:,} 行, 内存使用: {mem_usage:.2f} GB")

            except Exception as e:
                print(f"\n警告: 处理 {parquet_file} 时出错: {e}")
                continue

    print(f"\n处理完成:")
    print(f"  总行数: {total_lines:,}")
    if deduplicate:
        print(f"  去重行数: {duplicate_lines:,}")
    print(f"  过滤行数: {filtered_lines:,}")
    print(f"  保留行数: {written_lines:,}")
    print(f"  输出文件: {output_file}")


def process_parquet_batch(input_dir: str, output_file: str,
                         batch_size: int = 100000,
                         min_length: int = 10, max_length: int = 10000,
                         content_column: str = 'content'):
    """
    批量处理 parquet 文件（不去重，更快速）

    Args:
        input_dir: 包含 parquet 文件的目录
        output_file: 输出文件路径
        batch_size: 每批处理的行数
        min_length: 最小行长度
        max_length: 最大行长度
        content_column: 内容列名（默认为 'content'）
    """
    parquet_files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))

    if not parquet_files:
        raise ValueError(f"在 {input_dir} 中未找到 parquet 文件")

    print(f"找到 {len(parquet_files)} 个 parquet 文件")

    total_lines = 0
    filtered_lines = 0
    written_lines = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for parquet_file in tqdm(parquet_files, desc="批量处理 parquet 文件"):
            try:
                # 读取整个文件（如果文件不是特别大）
                df = pd.read_parquet(parquet_file, columns=[content_column])

                # 批量处理
                for i in range(0, len(df), batch_size):
                    batch = df[content_column].iloc[i:i+batch_size]

                    for content in batch:
                        total_lines += 1

                        if pd.isna(content):
                            filtered_lines += 1
                            continue

                        line = str(content).strip()

                        if not line or len(line) < min_length or len(line) > max_length:
                            filtered_lines += 1
                            continue

                        out_f.write(line + '\n')
                        written_lines += 1

                    if written_lines % 100000 == 0:
                        mem_usage = monitor_memory()
                        tqdm.write(f"已处理 {written_lines:,} 行, 内存使用: {mem_usage:.2f} GB")

            except Exception as e:
                print(f"\n警告: 处理 {parquet_file} 时出错: {e}")
                continue

    print(f"\n处理完成:")
    print(f"  总行数: {total_lines:,}")
    print(f"  过滤行数: {filtered_lines:,}")
    print(f"  保留行数: {written_lines:,}")
    print(f"  输出文件: {output_file}")


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
    parser.add_argument('--input', '-i', nargs='+',
                       help='输入文件路径（可多个，用于文本文件）')
    parser.add_argument('--input-dir', type=str,
                       help='输入目录（用于 parquet 文件）')
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
    parser.add_argument('--parquet', action='store_true',
                       help='处理 parquet 格式文件')
    parser.add_argument('--content-column', type=str, default='content',
                       help='parquet 文件中的内容列名（默认 content）')
    parser.add_argument('--batch-size', type=int, default=100000,
                       help='批处理大小（默认 100000）')

    args = parser.parse_args()

    # 检查输入参数
    if not args.parquet and not args.input:
        parser.error("需要指定 --input 或使用 --parquet --input-dir")

    if args.parquet and not args.input_dir:
        parser.error("使用 --parquet 时需要指定 --input-dir")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"初始内存使用: {monitor_memory():.2f} GB")

    # 处理数据
    if args.parquet:
        # 处理 parquet 文件
        if args.deduplicate:
            process_parquet_files(
                args.input_dir,
                args.output,
                min_length=args.min_length,
                max_length=args.max_length,
                deduplicate=True,
                content_column=args.content_column
            )
        else:
            # 批量处理（更快，不去重）
            process_parquet_batch(
                args.input_dir,
                args.output,
                batch_size=args.batch_size,
                min_length=args.min_length,
                max_length=args.max_length,
                content_column=args.content_column
            )
    else:
        # 处理文本文件（原有逻辑）
        validate_files(args.input)

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
