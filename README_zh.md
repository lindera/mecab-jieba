# mecab-jieba

基于 [jieba][jieba] 词频词典，并融合 [CC-CEDICT][cedict] 数据（拼音、繁体/简体、英文释义）的 MeCab/[lindera][lindera] CSV 格式中文（普通话）词典。可选通过在 [UD Chinese GSD][ud-gsd] 树库上进行 CRF 训练来优化连接代价。

## 输出文件

### 基础词典

| 文件 | 生成命令 | 说明 |
| ---- | -------- | ---- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | 基础词典 CSV（584K 条目） |

### CRF 训练词典源（`lindera build` 的输入）

运行 `bash scripts/run_experiment.sh <experiment_name>`，在 `work/experiments/<experiment_name>/export/` 下生成以下文件：

| 文件 | 说明 |
| ---- | ---- |
| `export/lex.csv` | 带 CRF 训练代价的词条 |
| `export/matrix.def` | 词性连接代价矩阵 |
| `export/char.def` | 字符类别定义 |
| `export/metadata.json` | 词典元数据 |

这些文件作为源文件，在另一个仓库中传递给 `lindera build`。

## 环境要求

- Python 3.10+
- 进行 CRF 训练时需要 [lindera][lindera] 2.3.2+（启用 `train` 特性）：
  `cargo install --path lindera-cli --features train`

## 构建 jieba.csv

此步骤为 MeCab/lindera 词典准备词汇文件（词表）。
脚本下载两个外部数据源，合并后根据词频计算代价值，转换为 MeCab CSV 格式。

```bash
python3 scripts/build_jieba_csv.py
```

1. 将 jieba 的 `dict.txt.big`（表层形式、词频、词性）和 CC-CEDICT（拼音、繁体/简体、英文释义）下载到 `work/`。
2. 根据 `int(-log10(freq / total_freq) * 100)` 计算词代价。
3. 使用 CC-CEDICT 数据扩充各条目（覆盖率约 22%）。
4. 以 MeCab CSV 格式输出 `jieba.csv`（584K 条目，约 25MB）。

## CRF 训练（可选）

CRF 训练从分词语料库中学习连接代价，替代 `jieba.csv` 中基于词频的代价。

### 第 1 步：准备训练数据

```bash
# 下载 UD Chinese GSD
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD

# 生成种子词典：jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# 将 UD GSD 训练集转换为 lindera 语料格式
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --split train
```

### 第 2 步：训练与导出

```bash
bash scripts/run_experiment.sh baseline
```

**输出：`work/experiments/baseline/export/`** — 包含用于 lindera 的 CRF 训练词典源文件（`lex.csv`、`matrix.def` 等）。

### 训练参数

通过环境变量控制 `run_experiment.sh` 的行为：

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `CORPUS` | `work/train/corpus.txt` | 训练语料路径 |
| `FEATURE_DEF` | `feature.def` | 特征模板文件 |
| `CHAR_DEF` | `char.def` | 字符类别定义文件 |
| `UNK_DEF` | `unk.def` | 未知词定义文件 |
| `LAMBDA` | `0.01` | 正则化系数 |
| `MAX_ITER` | `100` | 最大训练迭代次数 |
| `REGULARIZATION` | `l2` | `l1`、`l2` 或 `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | Elastic Net 的 L1 比率（0.0–1.0） |

示例（最优已知配置，F1=0.7732）：

```bash
LAMBDA=0.01 MAX_ITER=100 REGULARIZATION=l2 bash scripts/run_experiment.sh baseline
```

### 特征模板

特征模板在 `feature.def` 中定义。默认（最优已知）配置：

```text
UNIGRAM:%F[0]          # 词性标签（PKU 标签集）
UNIGRAM U01:%t         # 字符类型（char.def 类别）
UNIGRAM U06:%F[6]      # 字符数（1、2、3、4+）
UNIGRAM U07:%F[7]      # 表层形式的首字符
UNIGRAM U08:%F[8]      # 表层形式的末字符
UNIGRAM U09:%F[9]      # 频率段（high/mid/low/rare）
BIGRAM B00:%L[0]/%R[0] # 词性间转移
```

特征字段索引参考：

| 索引 | 字段 | 示例值 | 可用范围 |
| ---- | ---- | ------ | -------- |
| `F[0]` | 词性标签 | n, v, a, d, ... | seed + corpus |
| `F[1]` | 字符类型 | CHINESE, ALPHA, NUMERIC, ... | seed + corpus |
| `F[2]` | 拼音 | wu3han4, ... | 仅 seed |
| `F[3]` | 繁体字 | 武漢, ... | 仅 seed |
| `F[4]` | 简体字 | 武汉, ... | 仅 seed |
| `F[5]` | 英文释义 | Wuhan, ... | 仅 seed |
| `F[6]` | 字符数 | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | 首字符 | 武, 不, 中, ... | seed + corpus |
| `F[8]` | 末字符 | 市, 的, 了, ... | seed + corpus |
| `F[9]` | 频率段 | high, mid, low, rare | 仅 seed |

> **注意：** 当多个 unigram 特征同时启用时，L1 正则化会将所有 bigram 权重剪枝为零。请使用 `REGULARIZATION=l2`（默认）或 `elasticnet`。

### 最优已知结果

在 UD Chinese GSD 测试集（481 句）上的评估结果：

| 配置 | 正则化 | P | R | F1 |
| ---- | ------ | - | - | -- |
| `%F[0]` + B00 | L2, λ=0.01 | 0.7803 | 0.7464 | 0.7630 |
| **`%F[0]` + `%t` + `%F[6-9]` + B00** | **L2, λ=0.01** | **0.7822** | **0.7644** | **0.7732** |

使用默认的 `feature.def` 和 `REGULARIZATION=l2` 可复现最优结果。

## MeCab CSV 格式

`jieba.csv` 的每条记录：

```text
表层形,0,0,代价,词性,拼音,繁体字,简体字,英文释义
```

| 字段 | 说明 |
| ---- | ---- |
| surface | 表层形式 |
| left_id | 左上下文 ID（1×1 矩阵时为 0） |
| right_id | 右上下文 ID（1×1 矩阵时为 0） |
| cost | `-log10(freq / total) * 100` |
| pos | 词性标签（PKU 标签集） |
| pinyin | 拼音（CC-CEDICT；无数据时为 `*`） |
| traditional | 繁体字（无数据时为 `*`） |
| simplified | 简体字（无数据时为 `*`） |
| definition | 英文释义（无数据时为 `*`） |

CC-CEDICT 覆盖率：约 22% 的 jieba 条目附有 CC-CEDICT 数据。

### 使用 MeCab 直接加载 jieba.csv

```bash
mecab-dict-index -f utf-8 -t utf-8
echo "武汉市解除离汉离鄂通道管控措施" | mecab -d .
```

## PKU 词性标签集

| 标签 | 说明 | 标签 | 说明 |
| ---- | ---- | ---- | ---- |
| `n` | 名词 | `v` | 动词 |
| `ns` | 地名 | `a` | 形容词 |
| `nr` | 人名 | `d` | 副词 |
| `nt` | 机构名 | `r` | 代词 |
| `nz` | 其他专有名词 | `m` | 数词 |
| `q` | 量词 | `p` | 介词 |
| `c` | 连词 | `u` | 助词 |

## 仓库结构

```text
mecab-jieba/
├── jieba.csv              # MeCab/lindera 词典 CSV（由 build_jieba_csv.py 生成）
├── char.def               # 字符类别映射
├── matrix.def             # 连接代价矩阵（1x1 占位，仅静态使用）
├── unk.def                # 未知词定义
├── dicrc                  # MeCab 词典配置
├── dict-src/              # CRF 训练词典源（lindera build 的输入）
├── scripts/
│   ├── build_jieba_csv.py # 下载 jieba + CC-CEDICT，生成 jieba.csv
│   ├── build_seed.py      # 生成 CRF 训练用 seed.csv
│   ├── convert_conllu.py  # UD CoNLL-U → 训练语料转换
│   ├── convert_sighan.py  # SIGHAN bakeoff 语料转换（可选）
│   ├── evaluate.py        # 在 UD GSD 测试集上评估分词 F1
│   └── run_experiment.sh  # 训练/导出/评估一体化流水线
└── work/                  # 生成物（不提交）
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # 各实验的结果
        └── <name>/
            ├── export/    # lex.csv, matrix.def, char.def, metadata.json
            ├── dict/      # 编译后的 lindera 词典
            └── result.txt # 评估结果
```

## 许可证

本仓库中的构建脚本和词典定义文件采用 [MIT 许可证](LICENSE) 发布。

`jieba.csv` 已预构建并提交到本仓库。源文件（`dict.txt.big` 和 CC-CEDICT）在构建时下载到 `work/`，不包含在仓库中。

## 致谢

- [jieba][jieba] — 中文分词
- [CC-CEDICT][cedict] — 中英词典数据
- [CC-CEDICT-MeCab][cedict-mecab] — 结构参考
- [MeCab][mecab] — 形态素分析器
- [lindera][lindera] — 支持 CRF 训练的形态素分析器
- [UD Chinese GSD][ud-gsd] — 训练与评估语料

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
