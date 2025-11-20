# BPE Tokenizer 训练工具

用于训练 BPE (Byte Pair Encoding) tokenizer 的完整工具集，支持英文和 Python 代码语料，适用于大规模数据（>10GB）训练。

## 特性

- ✅ **BPE 算法**: 使用 HuggingFace Tokenizers 库的高性能 Rust 实现
- ✅ **ByteLevel 编码**: 自然处理所有 Unicode 字符和代码格式
- ✅ **Fill-in-the-Middle**: 支持 FIM tokens（用于代码补全）
- ✅ **大规模数据**: 迭代器模式训练，内存高效
- ✅ **完整配置**: 生成所有 HuggingFace 兼容的配置文件
- ✅ **测试工具**: 全面的测试脚本验证 tokenizer 性能

## 项目结构

```
tokenizer/
├── data/
│   ├── raw/              # 原始训练数据
│   ├── processed/        # 预处理后的数据
│   └── samples/          # 测试样本
├── scripts/
│   ├── prepare_data.py   # 数据预处理脚本
│   ├── train_tokenizer.py # 训练脚本
│   ├── test_tokenizer.py  # 测试脚本
│   └── utils.py          # 工具函数
├── output/               # 训练输出
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.json
│   └── merges.txt
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- `tokenizers>=0.19.0` - HuggingFace Tokenizers 核心库
- `transformers>=4.36.0` - HuggingFace Transformers（用于加载和测试）
- `datasets>=2.16.0` - 数据集处理
- `tqdm>=4.65.0` - 进度条
- `psutil>=5.9.0` - 内存监控

## 使用指南

### 1. 准备训练数据

将英文文本和 Python 代码放入 `data/raw/` 目录。

#### 数据格式要求：
- **编码**: UTF-8
- **格式**: 每行一个文档或代码片段
- **大小**: 支持任意大小，推荐 >10GB 以获得更好的 tokenizer

#### 示例数据结构：
```
data/raw/
├── english_corpus.txt
├── python_code.txt
└── mixed_data.txt
```

### 2. 数据预处理（可选）

如果需要合并、去重或过滤数据：

```bash
# 合并多个文件
python scripts/prepare_data.py \
  --input data/raw/*.txt \
  --output data/processed/corpus.txt \
  --merge-only

# 合并并去重
python scripts/prepare_data.py \
  --input data/raw/*.txt \
  --output data/processed/corpus_dedup.txt \
  --deduplicate \
  --min-length 10 \
  --max-length 10000
```

**参数说明**:
- `--input`: 输入文件（可多个）
- `--output`: 输出文件路径
- `--deduplicate`: 启用去重
- `--min-length`: 最小行长度（默认 10）
- `--max-length`: 最大行长度（默认 10000）
- `--merge-only`: 仅合并，不去重

### 3. 训练 Tokenizer

#### 基础训练（小数据集 < 1GB）

```bash
python scripts/train_tokenizer.py \
  --input data/processed/corpus.txt \
  --output output \
  --vocab-size 50000
```
```bash
python scripts/train_tokenizer.py \
  --input data/samples/example.txt \
  --output output \
  --vocab-size 100
```
#### 大规模训练（>10GB 数据）

使用迭代器模式以节省内存：

```bash
python scripts/train_tokenizer.py \
  --input data/processed/corpus.txt \
  --output output \
  --vocab-size 50000 \
  --use-iterator \
  --batch-size 1000
```

#### 多文件训练

```bash
python scripts/train_tokenizer.py \
  --input data/processed/english.txt data/processed/code.txt \
  --output output \
  --vocab-size 80000 \
  --min-frequency 2
```

**参数说明**:
- `--input`: 训练文件（可多个）
- `--output`: 输出目录
- `--vocab-size`: 词汇表大小（默认 50000）
- `--min-frequency`: token 最小频率（默认 2）
- `--use-iterator`: 使用迭代器训练（推荐大数据）
- `--batch-size`: 批次大小（默认 1000）
- `--add-prefix-space`: 添加前缀空格（某些模型需要）

### 4. 测试 Tokenizer

#### 运行所有测试

```bash
python scripts/test_tokenizer.py --tokenizer output
```

#### 测试特定文本

```bash
python scripts/test_tokenizer.py \
  --tokenizer output \
  --text "def hello():\n    print('Hello, World!')"
```

#### 测试特定样本类别

```bash
python scripts/test_tokenizer.py \
  --tokenizer output \
  --sample "Python 代码"
```

可用样本类别：
- `英文文本`
- `Python 代码`
- `混合文本`
- `FIM 场景`

## 特殊 Tokens

训练的 tokenizer 包含以下特殊 tokens：

| Token | 用途 |
|-------|------|
| `<pad>` | Padding token |
| `<eos>` | End of sequence |
| `<bos>` | Beginning of sequence |
| `<unk>` | Unknown token |
| `<indent>` | 缩进标记（Python 专用） |
| `<\|fim_prefix\|>` | Fill-in-the-middle 前缀 |
| `<\|fim_middle\|>` | Fill-in-the-middle 中间 |
| `<\|fim_suffix\|>` | Fill-in-the-middle 后缀 |
| `<\|fim_pad\|>` | Fill-in-the-middle padding |

## 输出文件说明

训练完成后，`output/` 目录包含以下文件：

### 1. `tokenizer.json`
主 tokenizer 文件，包含：
- BPE 模型配置
- Normalizer 设置
- Pre-tokenizer 配置
- Decoder 配置
- 词汇表和合并规则

### 2. `tokenizer_config.json`
Tokenizer 配置文件，包含：
- `add_prefix_space`: 是否添加前缀空格
- `added_tokens_decoder`: 添加的 token 解码器
- `bos_token`, `eos_token`, `pad_token`, `unk_token`: 特殊 token
- `clean_up_tokenization_spaces`: 是否清理空格
- `tokenizer_class`: Tokenizer 类名

**注意**: 此文件不包含 `model_type`。

### 3. `special_tokens_map.json`
特殊 token 映射，包含：
- 基础特殊 tokens（bos, eos, pad, unk）
- 额外特殊 tokens（indent, FIM tokens）

### 4. `vocab.json`
词汇表文件，JSON 格式，映射 token 到 ID。

### 5. `merges.txt`
BPE 合并规则文件，记录 token 合并顺序。

## 在代码中使用 Tokenizer

### 使用 HuggingFace Transformers

```python
from transformers import PreTrainedTokenizerFast

# 加载 tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("output")

# 编码
text = "def hello():\n    print('Hello')"
encoded = tokenizer(text)
print(encoded['input_ids'])

# 解码
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)
```

### 使用 HuggingFace Tokenizers

```python
from tokenizers import Tokenizer

# 加载 tokenizer
tokenizer = Tokenizer.from_file("output/tokenizer.json")

# 编码
text = "def hello():\n    print('Hello')"
encoded = tokenizer.encode(text)
print(encoded.ids)
print(encoded.tokens)

# 解码
decoded = tokenizer.decode(encoded.ids)
print(decoded)
```

### Fill-in-the-Middle 使用示例

```python
# 构造 FIM 格式
prefix = "def calculate_sum(a, b):"
suffix = "    return result"
middle = "\n    result = a + b"

fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

# 编码
encoded = tokenizer(fim_text)
```

## 性能优化建议

### 内存优化
- **大数据集**: 使用 `--use-iterator` 标志
- **批次大小**: 调整 `--batch-size` 以平衡速度和内存
- **预处理**: 先去重和过滤数据以减少训练集大小

### 词汇表大小
- **小模型**: 30,000 - 50,000 tokens
- **中等模型**: 50,000 - 64,000 tokens（推荐）
- **大模型**: 80,000 - 100,000 tokens

更大的词汇表：
- ✅ 更好的压缩率（更少的 tokens 表示相同文本）
- ✅ 更精确的语义表示
- ❌ 更大的模型参数量
- ❌ 更长的训练时间

### 训练速度
- **SSD**: 使用 SSD 存储训练数据（I/O 密集）
- **多核**: HuggingFace Tokenizers 自动使用多核
- **预处理**: 预先合并和清理数据

## 故障排查

### 编码错误
**问题**: `UnicodeDecodeError` 或乱码

**解决方案**:
```bash
# 检查文件编码
file -I data/raw/file.txt

# 转换为 UTF-8
iconv -f GBK -t UTF-8 input.txt > output.txt
```

### 内存不足
**问题**: 训练时内存溢出

**解决方案**:
```bash
# 使用迭代器模式
python scripts/train_tokenizer.py \
  --input data/corpus.txt \
  --output output \
  --use-iterator \
  --batch-size 500  # 减小批次大小
```

### Tokenizer 加载失败
**问题**: 无法加载训练的 tokenizer

**解决方案**:
```python
# 检查文件完整性
from scripts.utils import print_tokenizer_stats
print_tokenizer_stats("output")

# 验证 JSON 文件
from scripts.utils import validate_json_file
print(validate_json_file("output/tokenizer.json"))
```

## 工具函数

`scripts/utils.py` 提供了实用工具：

```python
from scripts.utils import (
    get_file_size,
    get_memory_usage,
    print_tokenizer_stats,
    create_sample_data,
    compare_tokenizers,
)

# 获取 tokenizer 统计信息
print_tokenizer_stats("output")

# 创建测试数据
create_sample_data("data/samples/test.txt", num_samples=100)

# 比较两个 tokenizers
compare_tokenizers(
    "output1/tokenizer.json",
    "output2/tokenizer.json",
    ["test text 1", "test text 2"]
)
```

## 参考资料

- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers/)
- [BPE 算法论文](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer](https://huggingface.co/gpt2)
- [CodeGen Tokenizer](https://huggingface.co/Salesforce/codegen-350M-mono)

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
