# Parquet 数据处理指南

## 概述

`prepare_data.py` 脚本现已支持处理 parquet 格式的大规模数据集（如 50GB+ 的数据）。脚本针对内存效率进行了优化，可以处理超大规模数据集。

## 前置要求

安装必要的依赖：

```bash
pip install -r requirements.txt
```

主要新增依赖：
- `pandas>=2.0.0` - 数据处理
- `pyarrow>=12.0.0` - Parquet 文件读取

## 使用方法

### 基本用法（快速处理，不去重）

处理 `pre_train_data_python` 文件夹中的所有 parquet 文件：

```bash
python scripts/prepare_data.py \
    --parquet \
    --input-dir pre_train_data_python \
    --output data/train_corpus.txt
```

### 带去重的处理

如果需要去除重复内容（会消耗更多内存和时间）：

```bash
python scripts/prepare_data.py \
    --parquet \
    --input-dir pre_train_data_python \
    --output data/train_corpus.txt \
    --deduplicate
```

### 自定义过滤参数

设置最小和最大行长度：

```bash
python scripts/prepare_data.py \
    --parquet \
    --input-dir pre_train_data_python \
    --output data/train_corpus.txt \
    --min-length 50 \
    --max-length 5000
```

### 自定义内容列名

如果 parquet 文件中的内容列不是 `content`：

```bash
python scripts/prepare_data.py \
    --parquet \
    --input-dir pre_train_data_python \
    --output data/train_corpus.txt \
    --content-column text
```

### 调整批处理大小

对于内存较小的机器，可以减小批处理大小：

```bash
python scripts/prepare_data.py \
    --parquet \
    --input-dir pre_train_data_python \
    --output data/train_corpus.txt \
    --batch-size 50000
```

## 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--parquet` | 启用 parquet 文件处理模式 | - |
| `--input-dir` | parquet 文件所在目录 | - |
| `--output`, `-o` | 输出文本文件路径 | 必需 |
| `--deduplicate` | 启用去重（内存消耗大） | False |
| `--min-length` | 最小行长度（字符数） | 10 |
| `--max-length` | 最大行长度（字符数） | 10000 |
| `--content-column` | parquet 中的内容列名 | content |
| `--batch-size` | 批处理大小（行数） | 100000 |

## 处理流程

### 不去重模式（推荐用于大数据集）

1. 扫描目录中的所有 `.parquet` 文件
2. 按文件名排序处理
3. 逐个读取 parquet 文件
4. 提取 `content` 列
5. 过滤空行和不符合长度要求的行
6. 写入输出文件
7. 每处理 10 万行报告一次进度和内存使用

### 去重模式（适用于较小数据集或需要去重的场景）

在不去重模式的基础上，增加：
- 维护一个 set 来记录已见过的内容
- 跳过重复内容
- 注意：50GB 数据如果去重，可能需要数十 GB 内存

## 性能建议

### 50GB 数据集处理建议

1. **不去重模式**（推荐）
   - 内存需求：< 5GB
   - 处理时间：取决于磁盘 I/O 速度
   - 适合：初次处理大规模数据

2. **去重模式**（慎用）
   - 内存需求：可能超过 20-30GB（取决于数据重复率）
   - 处理时间：更长
   - 适合：数据规模较小或内存充足的情况

### 优化建议

- 使用 SSD 存储可以显著提升处理速度
- 如果内存不足，减小 `--batch-size`
- 对于超大数据集，考虑分批处理后再合并
- 可以使用 `--min-length` 和 `--max-length` 过滤掉异常数据

## 输出格式

输出文件为纯文本格式，每行一条记录：

```
'tx_drop'...
@mock.patch.object(host.Host, "list_instan...
self.assertEqual(5, drvr._get_vcpu_use...
def test_get_instance_capabilities(self):\...
```

每行内容已经：
- 去除首尾空白
- 过滤空行
- 过滤过短/过长的内容
- 去重（如果启用）

## 监控和调试

脚本会实时显示：
- 处理进度条
- 每 10 万行的内存使用情况
- 最终统计信息：
  - 总行数
  - 去重行数（如果启用）
  - 过滤行数
  - 保留行数
  - 输出文件路径

## 常见问题

### Q: 处理 50GB 数据需要多长时间？

A: 取决于：
- 硬件性能（CPU、磁盘 I/O）
- 是否启用去重
- parquet 文件的数量和大小

通常在现代 SSD 上，不去重模式可能需要 30 分钟到 2 小时。

### Q: 内存不足怎么办？

A:
1. 不要使用 `--deduplicate`
2. 减小 `--batch-size` 到 50000 或更小
3. 分批处理数据

### Q: 如何验证处理结果？

A: 使用以下命令检查输出文件：

```bash
# 查看文件大小
ls -lh data/train_corpus.txt

# 查看行数
wc -l data/train_corpus.txt

# 查看前几行
head -20 data/train_corpus.txt

# 查看统计信息
wc -l data/train_corpus.txt
du -h data/train_corpus.txt
```

## 后续步骤

处理完数据后，可以使用处理好的文本文件训练 BPE tokenizer：

```bash
python scripts/train_tokenizer.py \
    --train-file data/train_corpus.txt \
    --vocab-size 32000 \
    --output-dir output/
```
