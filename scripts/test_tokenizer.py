#!/usr/bin/env python3
"""
Tokenizer 测试脚本
测试训练好的 tokenizer 在各种场景下的表现
"""

import argparse
import os
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


# 测试样本
TEST_SAMPLES = {
    "英文文本": [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ],
    "Python 代码": [
        "def hello_world():\n    print('Hello, World!')",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        "class MyClass:\n    def __init__(self, name):\n        self.name = name",
        "import numpy as np\nimport pandas as pd\n\ndata = pd.read_csv('file.csv')",
    ],
    "混合文本": [
        "# This is a comment\nprint('Hello')",
        "Let's write some code: x = 42",
        "# 导入库\nimport sys\n# 打印版本\nprint(sys.version)",
    ],
    "FIM 场景": [
        "<|fim_prefix|>def calculate_sum(a, b):<|fim_suffix|>    return result<|fim_middle|>\n    result = a + b",
        "<|fim_prefix|>import <|fim_suffix|>\ndata = pd.read_csv('file.csv')<|fim_middle|>pandas as pd",
    ],
}


class TokenizerTester:
    """Tokenizer 测试器"""

    def __init__(self, tokenizer_path: str):
        """
        初始化测试器

        Args:
            tokenizer_path: tokenizer.json 文件路径或目录路径
        """
        if os.path.isdir(tokenizer_path):
            # 如果是目录，加载目录中的 tokenizer
            try:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_path
                )
                print(f"使用 PreTrainedTokenizerFast 加载: {tokenizer_path}")
            except:
                # 回退到加载 tokenizer.json
                json_path = os.path.join(tokenizer_path, "tokenizer.json")
                self.tokenizer = Tokenizer.from_file(json_path)
                print(f"使用 Tokenizer 加载: {json_path}")
        else:
            # 直接加载 tokenizer.json
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"使用 Tokenizer 加载: {tokenizer_path}")

    def test_encoding(self, text: str, verbose: bool = True):
        """
        测试编码

        Args:
            text: 输入文本
            verbose: 是否打印详细信息
        """
        # 编码
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            encoding = self.tokenizer(text, add_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            token_ids = encoding['input_ids']
            decoded = self.tokenizer.decode(token_ids)
        else:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids
            decoded = self.tokenizer.decode(token_ids)

        if verbose:
            print(f"\n原文本: {repr(text)}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"解码结果: {repr(decoded)}")
            print(f"是否一致: {text == decoded}")

        return tokens, token_ids, decoded

    def test_special_tokens(self):
        """测试特殊 tokens"""
        print("\n" + "=" * 60)
        print("测试特殊 Tokens")
        print("=" * 60)

        special_tokens = [
            "<pad>", "<eos>", "<bos>", "<unk>",
            "<indent>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        ]

        for token in special_tokens:
            if isinstance(self.tokenizer, PreTrainedTokenizerFast):
                # 使用 transformers tokenizer
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    decoded = self.tokenizer.decode([token_id])
                    print(f"  {token:20s} -> ID: {token_id:6d} -> 解码: {repr(decoded)}")
                else:
                    print(f"  {token:20s} -> 未找到")
            else:
                # 使用 tokenizers Tokenizer
                token_id = self.tokenizer.token_to_id(token)
                if token_id is not None:
                    decoded = self.tokenizer.decode([token_id])
                    print(f"  {token:20s} -> ID: {token_id:6d} -> 解码: {repr(decoded)}")
                else:
                    print(f"  {token:20s} -> 未找到")

    def test_samples(self, samples: dict):
        """
        测试样本集

        Args:
            samples: 测试样本字典
        """
        for category, texts in samples.items():
            print("\n" + "=" * 60)
            print(f"测试类别: {category}")
            print("=" * 60)

            for text in texts:
                try:
                    self.test_encoding(text, verbose=True)
                except Exception as e:
                    print(f"错误: {e}")
                    print(f"文本: {repr(text)}")

    def test_roundtrip(self, text: str):
        """
        测试编码-解码往返一致性

        Args:
            text: 输入文本
        """
        _, _, decoded = self.test_encoding(text, verbose=False)
        is_consistent = (text == decoded)

        print(f"\n往返测试:")
        print(f"  原文本: {repr(text)}")
        print(f"  解码后: {repr(decoded)}")
        print(f"  一致性: {'✓ 通过' if is_consistent else '✗ 失败'}")

        return is_consistent

    def test_vocab_info(self):
        """打印词汇表信息"""
        print("\n" + "=" * 60)
        print("词汇表信息")
        print("=" * 60)

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            vocab_size = self.tokenizer.vocab_size
            print(f"  词汇表大小: {vocab_size:,}")
            print(f"  PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            print(f"  BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
            print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
            print(f"  UNK token: {self.tokenizer.unk_token} (ID: {self.tokenizer.unk_token_id})")
        else:
            vocab_size = self.tokenizer.get_vocab_size()
            print(f"  词汇表大小: {vocab_size:,}")

    def test_tokenization_stats(self, text: str):
        """
        统计分词信息

        Args:
            text: 输入文本
        """
        tokens, _, _ = self.test_encoding(text, verbose=False)

        chars = len(text)
        num_tokens = len(tokens)
        ratio = chars / num_tokens if num_tokens > 0 else 0

        print(f"\n分词统计:")
        print(f"  字符数: {chars}")
        print(f"  Token 数: {num_tokens}")
        print(f"  字符/Token 比率: {ratio:.2f}")

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("开始 Tokenizer 测试")
        print("=" * 60)

        # 1. 词汇表信息
        self.test_vocab_info()

        # 2. 特殊 tokens
        self.test_special_tokens()

        # 3. 测试样本
        self.test_samples(TEST_SAMPLES)

        # 4. 往返一致性测试
        print("\n" + "=" * 60)
        print("往返一致性测试")
        print("=" * 60)
        consistency_tests = [
            "Hello, world!",
            "def test():\n    pass",
            "x = 42; y = 3.14",
        ]
        for text in consistency_tests:
            self.test_roundtrip(text)

        # 5. 分词统计
        print("\n" + "=" * 60)
        print("分词效率测试")
        print("=" * 60)
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        self.test_tokenization_stats(long_text)

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='测试 Tokenizer')
    parser.add_argument('--tokenizer', '-t', required=True,
                       help='Tokenizer 文件路径（tokenizer.json 或目录）')
    parser.add_argument('--text', type=str,
                       help='测试特定文本')
    parser.add_argument('--sample', type=str,
                       help='测试特定样本类别')

    args = parser.parse_args()

    # 检查文件存在性
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"文件不存在: {args.tokenizer}")

    # 创建测试器
    tester = TokenizerTester(args.tokenizer)

    # 执行测试
    if args.text:
        # 测试特定文本
        print("测试自定义文本:")
        tester.test_encoding(args.text)
        tester.test_tokenization_stats(args.text)
    elif args.sample:
        # 测试特定样本类别
        if args.sample in TEST_SAMPLES:
            tester.test_samples({args.sample: TEST_SAMPLES[args.sample]})
        else:
            print(f"未知的样本类别: {args.sample}")
            print(f"可用类别: {list(TEST_SAMPLES.keys())}")
    else:
        # 运行所有测试
        tester.run_all_tests()


if __name__ == '__main__':
    main()
