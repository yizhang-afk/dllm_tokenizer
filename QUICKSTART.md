# 快速开始指南

## 5 分钟上手 BPE Tokenizer 训练

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2: 准备训练数据

将你的英文文本和 Python 代码文件放入 `data/raw/` 目录：

```bash
# 示例：复制你的数据文件
cp /path/to/your/english_corpus.txt data/raw/
cp /path/to/your/python_code.txt data/raw/
```

**数据要求**：
- UTF-8 编码
- 每行一个文档/代码片段
- 推荐数据量：>10GB 效果更好

### 步骤 3: 训练 Tokenizer

#### 小数据集 (<1GB)

```bash
python scripts/train_tokenizer.py \
  --input data/raw/your_data.txt \
  --output output \
  --vocab-size 50000
```

#### 大数据集 (>10GB) - 推荐

```bash
python scripts/train_tokenizer.py \
  --input data/raw/*.txt \
  --output output \
  --vocab-size 50000 \
  --use-iterator \
  --batch-size 1000
```

### 步骤 4: 测试 Tokenizer

```bash
# 运行所有测试
python scripts/test_tokenizer.py --tokenizer output

# 测试自定义文本
python scripts/test_tokenizer.py \
  --tokenizer output \
  --text "def hello():\n    print('Hello')"
```

### 步骤 5: 在代码中使用

```python
from transformers import PreTrainedTokenizerFast

# 加载 tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("output")

# 编码文本
text = "def hello():\n    print('Hello, World!')"
encoded = tokenizer(text)
print("Token IDs:", encoded['input_ids'])

# 解码
decoded = tokenizer.decode(encoded['input_ids'])
print("Decoded:", decoded)

# 查看 tokens
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
print("Tokens:", tokens)
```

## 示例：使用提供的样本数据

如果你想先测试一下流程：

```bash
# 使用示例数据训练（仅用于测试流程）
python scripts/train_tokenizer.py \
  --input data/samples/example.txt \
  --output output_test \
  --vocab-size 5000

# 测试
python scripts/test_tokenizer.py --tokenizer output_test
```

**注意**：示例数据很小，仅用于测试流程，实际训练需要大规模数据。

## 输出文件

训练完成后，`output/` 目录包含：

```
output/
├── tokenizer.json              # 主文件
├── tokenizer_config.json       # 配置
├── special_tokens_map.json     # 特殊 token 映射
├── vocab.json                  # 词汇表
└── merges.txt                  # BPE 合并规则
```

所有这些文件都可以直接用于 HuggingFace Transformers。

## Fill-in-the-Middle (FIM) 使用

训练的 tokenizer 支持代码补全场景：

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("output")

# FIM 格式
prefix = "def calculate_sum(a, b):"
suffix = "    return result"
middle = "\n    result = a + b"

# 组合 FIM
fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

# 编码
encoded = tokenizer(fim_text)
print(encoded['input_ids'])
```

## 常见问题

### Q: 训练需要多长时间？
A: 取决于数据量和硬件。参考速度：
- 1GB 数据：5-15 分钟
- 10GB 数据：30-60 分钟
- 100GB 数据：3-6 小时

### Q: 需要多少内存？
A: 使用 `--use-iterator` 模式：
- 小数据集：2-4GB
- 大数据集：4-8GB
- 超大数据集：8-16GB

### Q: 词汇表大小如何选择？
A: 推荐：
- 小模型：30,000 - 50,000
- 中等模型：50,000 - 64,000
- 大模型：64,000 - 100,000

### Q: 如何处理非 UTF-8 编码的数据？
A: 使用 `iconv` 转换：
```bash
iconv -f GBK -t UTF-8 input.txt > output.txt
```

或使用预处理脚本的自动转换功能。

## 下一步

- 查看完整文档：`README.md`
- 了解工具函数：`scripts/utils.py`
- 自定义配置：修改 `scripts/train_tokenizer.py` 中的参数

## 支持

遇到问题？检查：
1. 数据是否为 UTF-8 编码
2. 内存是否充足（使用 `--use-iterator`）
3. 文件路径是否正确
