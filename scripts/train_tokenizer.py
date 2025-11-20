#!/usr/bin/env python3
"""
BPE Tokenizer 训练脚本
支持大规模数据训练，输出 HuggingFace 兼容的配置文件
"""

import os
import json
import argparse
from pathlib import Path
from typing import Iterator, List, Optional
import psutil
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFD
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


class BPETokenizerTrainer:
    """BPE Tokenizer 训练器"""

    # 特殊 tokens
    SPECIAL_TOKENS = [
        "<|endoftext|>",
        "<unk>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
    ]

    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        add_prefix_space: bool = False,
    ):
        """
        初始化训练器

        Args:
            vocab_size: 词汇表大小
            min_frequency: token 最小出现频率
            add_prefix_space: 是否在开头添加空格
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.add_prefix_space = add_prefix_space
        self.tokenizer = None

    def prepare_tokenizer(self):
        """初始化 tokenizer 组件"""
        print("初始化 tokenizer...")

        # 创建 BPE 模型
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        # 设置 normalizer（最小化处理，保持代码格式）
        # NFD: Unicode 标准化，但不做大小写转换
        self.tokenizer.normalizer = Sequence([NFD()])

        # 设置 pre-tokenizer（ByteLevel 适合代码和多语言）
        self.tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=self.add_prefix_space
        )

        # 设置 decoder
        self.tokenizer.decoder = ByteLevelDecoder()

        print("Tokenizer 初始化完成")

    def get_trainer(self) -> BpeTrainer:
        """配置训练器"""
        # 不在训练时添加特殊 tokens，训练后手动添加到末尾
        return BpeTrainer(
            vocab_size=self.vocab_size - len(self.SPECIAL_TOKENS),  # 预留位置给特殊 tokens
            min_frequency=self.min_frequency,
            special_tokens=[],  # 先不添加
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet(),
        )

    def train_from_files(self, files: List[str]):
        """
        从文件训练

        Args:
            files: 训练文件路径列表
        """
        print(f"开始训练，使用 {len(files)} 个文件...")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最小频率: {self.min_frequency}")
        print(f"特殊 tokens: {len(self.SPECIAL_TOKENS)}")

        trainer = self.get_trainer()

        # 监控内存
        print(f"初始内存使用: {self._monitor_memory():.2f} GB")

        # 训练
        self.tokenizer.train(files, trainer)

        print(f"训练完成，最终内存使用: {self._monitor_memory():.2f} GB")

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        length: Optional[int] = None
    ):
        """
        从迭代器训练（更节省内存）

        Args:
            iterator: 文本行迭代器
            length: 总行数（用于进度条）
        """
        print("开始从迭代器训练...")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最小频率: {self.min_frequency}")

        trainer = self.get_trainer()

        # 监控内存
        print(f"初始内存使用: {self._monitor_memory():.2f} GB")

        # 训练
        self.tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length
        )

        print(f"训练完成，最终内存使用: {self._monitor_memory():.2f} GB")

    def add_special_tokens(self):
        """在训练后添加特殊 tokens 到词汇表末尾"""
        print("添加特殊 tokens 到词汇表末尾...")
        for token in self.SPECIAL_TOKENS:
            self.tokenizer.add_special_tokens([token])
        print(f"已添加 {len(self.SPECIAL_TOKENS)} 个特殊 tokens")

    def configure_post_processor(self):
        """配置后处理器（添加特殊 tokens）"""
        # 对于 GPT 风格的 tokenizer，通常不需要自动添加特殊 tokens
        # 如果需要，可以在使用时手动添加 <|endoftext|>
        pass

    def save(self, output_dir: str, save_all: bool = True):
        """
        保存 tokenizer

        Args:
            output_dir: 输出目录
            save_all: 是否保存所有配置文件
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存主文件 tokenizer.json
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"已保存: {tokenizer_path}")

        if save_all:
            # 2. 保存 tokenizer_config.json
            self._save_tokenizer_config(output_dir)

            # 3. 保存 special_tokens_map.json
            self._save_special_tokens_map(output_dir)

            # 4. 保存 vocab.json 和 merges.txt
            self._save_vocab_and_merges(output_dir)

    def _save_tokenizer_config(self, output_dir: str):
        """保存 tokenizer_config.json（不包含 model_type）"""
        config = {
            "add_bos_token": False,
            "add_prefix_space": self.add_prefix_space,
            "added_tokens_decoder": self._get_added_tokens_decoder(),
            "bos_token": "<|endoftext|>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "tokenizer_class": "PreTrainedTokenizerFast",
        }

        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"已保存: {config_path}")

    def _save_special_tokens_map(self, output_dir: str):
        """保存 special_tokens_map.json"""
        # 提取非基础特殊 tokens
        additional_special_tokens = [
            token for token in self.SPECIAL_TOKENS
            if token not in ["<|endoftext|>", "<unk>"]
        ]

        special_tokens_map = {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<unk>",
            "additional_special_tokens": additional_special_tokens,
        }

        map_path = os.path.join(output_dir, "special_tokens_map.json")
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

        print(f"已保存: {map_path}")

    def _save_vocab_and_merges(self, output_dir: str):
        """保存 vocab.json 和 merges.txt"""
        # 保存 vocab.json
        vocab = self.tokenizer.get_vocab()
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"已保存: {vocab_path}")

        # 保存 merges.txt
        # 从已保存的 tokenizer.json 中读取 merges
        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        merges = tokenizer_data.get('model', {}).get('merges', [])
        merges_path = os.path.join(output_dir, "merges.txt")
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in merges:
                f.write(f"{merge}\n")
        print(f"已保存: {merges_path}")

    def _get_added_tokens_decoder(self):
        """生成 added_tokens_decoder 配置"""
        added_tokens = {}

        for i, token in enumerate(self.SPECIAL_TOKENS):
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                added_tokens[str(token_id)] = {
                    "content": token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                }

        return added_tokens

    def _monitor_memory(self) -> float:
        """监控内存使用（GB）"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024 / 1024


def batch_iterator(
    file_paths: List[str],
    batch_size: int = 1000
) -> Iterator[List[str]]:
    """
    批量迭代文件内容（内存高效）

    Args:
        file_paths: 文件路径列表
        batch_size: 每批大小

    Yields:
        文本行批次
    """
    for file_path in file_paths:
        batch = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

            # 处理最后一批
            if batch:
                yield batch


def main():
    parser = argparse.ArgumentParser(description='训练 BPE Tokenizer')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='训练文件路径（可多个）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录路径')
    parser.add_argument('--vocab-size', type=int, default=50000,
                       help='词汇表大小（默认50000）')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='最小频率（默认2）')
    parser.add_argument('--add-prefix-space', action='store_true',
                       help='是否添加前缀空格')
    parser.add_argument('--use-iterator', action='store_true',
                       help='使用迭代器训练（更省内存）')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='批次大小（仅在使用迭代器时有效）')

    args = parser.parse_args()

    # 验证输入文件
    for file_path in args.input:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    # 创建训练器
    trainer = BPETokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        add_prefix_space=args.add_prefix_space,
    )

    # 准备 tokenizer
    trainer.prepare_tokenizer()

    # 训练
    if args.use_iterator:
        # 使用迭代器训练（内存高效）
        print("使用迭代器模式训练...")
        iterator = batch_iterator(args.input, args.batch_size)
        trainer.train_from_iterator(iterator)
    else:
        # 直接从文件训练
        print("使用文件模式训练...")
        trainer.train_from_files(args.input)

    # 添加特殊 tokens 到词汇表末尾
    trainer.add_special_tokens()

    # 配置后处理器
    trainer.configure_post_processor()

    # 保存所有文件
    trainer.save(args.output, save_all=True)

    print(f"\n训练完成！tokenizer 已保存到: {args.output}")
    print(f"\n生成的文件:")
    print(f"  - tokenizer.json")
    print(f"  - tokenizer_config.json")
    print(f"  - special_tokens_map.json")
    print(f"  - vocab.json")
    print(f"  - merges.txt")


if __name__ == '__main__':
    main()
