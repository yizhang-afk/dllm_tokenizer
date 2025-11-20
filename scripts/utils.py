#!/usr/bin/env python3
"""
工具函数集合
提供常用的辅助功能
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
import psutil


def get_file_size(file_path: str) -> float:
    """
    获取文件大小（MB）

    Args:
        file_path: 文件路径

    Returns:
        文件大小（MB）
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / 1024 / 1024


def get_file_lines(file_path: str) -> int:
    """
    快速统计文件行数

    Args:
        file_path: 文件路径

    Returns:
        行数
    """
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def get_memory_usage() -> Dict[str, float]:
    """
    获取当前内存使用情况

    Returns:
        包含内存信息的字典（GB）
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        'rss_gb': mem_info.rss / 1024 / 1024 / 1024,  # 实际物理内存
        'vms_gb': mem_info.vms / 1024 / 1024 / 1024,  # 虚拟内存
    }


def print_memory_usage(prefix: str = ""):
    """
    打印内存使用情况

    Args:
        prefix: 前缀文本
    """
    mem = get_memory_usage()
    print(f"{prefix}内存使用: RSS={mem['rss_gb']:.2f} GB, VMS={mem['vms_gb']:.2f} GB")


def check_utf8_encoding(file_path: str) -> bool:
    """
    检查文件是否为 UTF-8 编码

    Args:
        file_path: 文件路径

    Returns:
        True 如果是 UTF-8，否则 False
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def validate_json_file(file_path: str) -> bool:
    """
    验证 JSON 文件是否有效

    Args:
        file_path: JSON 文件路径

    Returns:
        True 如果有效，否则 False
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def load_json(file_path: str) -> Any:
    """
    加载 JSON 文件

    Args:
        file_path: JSON 文件路径

    Returns:
        解析后的 JSON 对象
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    保存 JSON 文件

    Args:
        data: 要保存的数据
        file_path: 输出文件路径
        indent: 缩进空格数
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_lines_lazy(file_path: str, strip: bool = True) -> Iterator[str]:
    """
    懒加载读取文件行（内存高效）

    Args:
        file_path: 文件路径
        strip: 是否去除首尾空白

    Yields:
        文件中的每一行
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if strip:
                line = line.strip()
            if line:
                yield line


def batch_read_lines(
    file_path: str,
    batch_size: int = 1000,
    strip: bool = True
) -> Iterator[List[str]]:
    """
    批量读取文件行

    Args:
        file_path: 文件路径
        batch_size: 批次大小
        strip: 是否去除首尾空白

    Yields:
        每批文本行列表
    """
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if strip:
                line = line.strip()
            if line:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        # 返回最后一批
        if batch:
            yield batch


def create_sample_data(output_path: str, num_samples: int = 100):
    """
    创建示例数据用于测试

    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    samples = []

    # 英文文本
    samples.extend([
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a powerful programming language.",
        "Data science combines statistics and programming.",
    ] * (num_samples // 10))

    # Python 代码
    code_samples = [
        "def hello():\n    print('Hello')",
        "for i in range(10):\n    print(i)",
        "class MyClass:\n    pass",
        "import numpy as np",
        "x = [i for i in range(10)]",
    ]
    samples.extend(code_samples * (num_samples // 10))

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples[:num_samples]:
            f.write(sample + '\n')

    print(f"已创建 {num_samples} 个示例样本: {output_path}")


def compare_tokenizers(
    tokenizer1_path: str,
    tokenizer2_path: str,
    test_texts: List[str]
):
    """
    比较两个 tokenizer 的性能

    Args:
        tokenizer1_path: 第一个 tokenizer 路径
        tokenizer2_path: 第二个 tokenizer 路径
        test_texts: 测试文本列表
    """
    from tokenizers import Tokenizer

    tok1 = Tokenizer.from_file(tokenizer1_path)
    tok2 = Tokenizer.from_file(tokenizer2_path)

    print(f"\n比较 Tokenizers:")
    print(f"  Tokenizer 1: {tokenizer1_path}")
    print(f"  Tokenizer 2: {tokenizer2_path}")
    print(f"  测试文本数: {len(test_texts)}")
    print("\n" + "=" * 80)

    for i, text in enumerate(test_texts, 1):
        enc1 = tok1.encode(text)
        enc2 = tok2.encode(text)

        print(f"\n文本 {i}: {repr(text[:50])}")
        print(f"  Tokenizer 1: {len(enc1.tokens)} tokens")
        print(f"  Tokenizer 2: {len(enc2.tokens)} tokens")
        print(f"  差异: {len(enc1.tokens) - len(enc2.tokens)}")


def get_tokenizer_stats(tokenizer_dir: str) -> Dict[str, Any]:
    """
    获取 tokenizer 统计信息

    Args:
        tokenizer_dir: tokenizer 目录路径

    Returns:
        统计信息字典
    """
    stats = {}

    # 检查文件
    files = ['tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.json', 'merges.txt']

    stats['files'] = {}
    for file in files:
        file_path = os.path.join(tokenizer_dir, file)
        if os.path.exists(file_path):
            stats['files'][file] = {
                'exists': True,
                'size_mb': get_file_size(file_path)
            }
        else:
            stats['files'][file] = {'exists': False}

    # 读取配置
    config_path = os.path.join(tokenizer_dir, 'tokenizer_config.json')
    if os.path.exists(config_path):
        config = load_json(config_path)
        stats['config'] = config

    # 读取特殊 tokens
    special_path = os.path.join(tokenizer_dir, 'special_tokens_map.json')
    if os.path.exists(special_path):
        special_tokens = load_json(special_path)
        stats['special_tokens'] = special_tokens

    # 词汇表大小
    vocab_path = os.path.join(tokenizer_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        vocab = load_json(vocab_path)
        stats['vocab_size'] = len(vocab)

    return stats


def print_tokenizer_stats(tokenizer_dir: str):
    """
    打印 tokenizer 统计信息

    Args:
        tokenizer_dir: tokenizer 目录路径
    """
    stats = get_tokenizer_stats(tokenizer_dir)

    print(f"\nTokenizer 统计信息: {tokenizer_dir}")
    print("=" * 60)

    # 文件信息
    print("\n文件:")
    for file, info in stats['files'].items():
        if info['exists']:
            print(f"  ✓ {file:30s} ({info['size_mb']:.2f} MB)")
        else:
            print(f"  ✗ {file:30s} (不存在)")

    # 词汇表大小
    if 'vocab_size' in stats:
        print(f"\n词汇表大小: {stats['vocab_size']:,}")

    # 特殊 tokens
    if 'special_tokens' in stats:
        print(f"\n特殊 Tokens:")
        for key, value in stats['special_tokens'].items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} 个")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    # 测试功能
    print("工具函数测试")
    print_memory_usage("当前")
