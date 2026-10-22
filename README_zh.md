# mecab-jieba

基于 [jieba][jieba] 词频词典，并融合 [CC-CEDICT][cedict] 数据
（拼音、繁体/简体、英文释义）的 MeCab/[lindera][lindera] CSV 格式中文（普通话）词典。
[lindera][lindera] 用来构建 Jieba 词典的 `dict-src/` 词典源，其词代价和连接代价
是在 [UD Chinese GSD][ud-gsd] 树库及其简体版 [UD Chinese GSDSimp][ud-gsdsimp]
（相同的句子）上通过 CRF 训练得到的。

## 输出文件

### 基础词典

| 文件 | 生成命令 | 说明 |
| ---- | -------- | ---- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | 基础词典 CSV（584K 条目） |

### CRF 训练词典源（`lindera build` 的输入）

运行 `bash scripts/run_experiment.sh <experiment_name>`，
在 `work/experiments/<experiment_name>/export/` 下生成以下文件：

| 文件 | 说明 |
| ---- | ---- |
| `export/lex.csv` | 带 CRF 训练代价和上下文 ID 的词条 |
| `export/matrix.def` | 连接代价矩阵（每个词性一个上下文 ID） |
| `export/unk.def` | 带 CRF 训练代价的未知词条目 |
| `export/char.def` | 字符类别定义 |
| `export/feature.def` | 训练时使用的特征模板 |
| `export/rewrite.def` | 训练时使用的改写规则 |
| `export/left-id.def` | 左上下文 ID |
| `export/right-id.def` | 右上下文 ID |
| `export/metadata.json` | 词典元数据 |

这些文件是传递给 `lindera build` 的源文件。
`dict-src/` 保存的是所发布版本的 `baseline` 实验的这些文件。

## 环境要求

- Python 3.10+
- 进行 CRF 训练时需要 commit `150db680` 及以后的 [lindera][lindera]
  （启用 `train` 特性）：`cargo install --path lindera-cli --features train`。
  更早的版本在训练时没有词典词候选，并为没有训练权重的连接导出 32767
  （[lindera#1250][lindera-1250]）。

## 构建 jieba.csv

此步骤为 MeCab/lindera 词典准备词汇文件（词表）。
脚本下载两个外部数据源，合并后根据词频计算代价值，转换为 MeCab CSV 格式。

```bash
python3 scripts/build_jieba_csv.py
```

1. 将 jieba 的 `dict.txt.big`（表层形式、词频、词性）和 CC-CEDICT（拼音、繁体/简体、英文释义）下载到 `work/`。
2. 根据 `int(-log10(freq / total_freq) * 100)` 计算词代价。
3. 在末尾补充 `dict.txt.big` 中缺少的 `〇`（U+3007），使用与 `零` 相同的词频和词性。
   `total_freq` 仍为 `dict.txt.big` 的值，因此其他代价不变。
4. 使用 CC-CEDICT 数据扩充各条目（覆盖率约 22%）。
5. 以 MeCab CSV 格式输出 `jieba.csv`（584K 条目，约 25MB）。

## CRF 训练（可选）

CRF 训练从分词语料库中学习连接代价，替代 `jieba.csv` 中基于词频的代价。

### 第 1 步：准备训练数据

`make prepare` 会运行 `scripts/build_jieba_csv.py` 和以下命令。

```bash
# 下载 UD Chinese GSD 和 GSDSimp（dict-src 训练时使用的 commit）
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD
git -C work/ud-chinese/UD_Chinese-GSD checkout 22f73e87f7ddc530bb131a6b7587971a03e1a712
git clone https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git \
  work/ud-chinese/UD_Chinese-GSDSimp
git -C work/ud-chinese/UD_Chinese-GSDSimp checkout 4231dfd59866fa5999ad4a6bc1fdecd7985b3b59

# 生成种子词典：jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# 将两者的训练集转换为 lindera 语料格式
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ work/ud-chinese/UD_Chinese-GSDSimp/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --seed work/train/seed.csv \
  --split train
```

与 MeCab 的学习器一样，只有当种子词典中有一行的表层形式和整个特征都与之相同时，
`lindera train` 才把一个正确词视为词典词。
因此，对于表层形式在种子词典中的正确词，`convert_conllu.py` 会赋予其中一行的特征：
与正确词性相同的第一行；若没有，则词性首字母相同的第一行
（jieba 的词性是 PKU 词性的细分：`u` 对应 `uj`，`v` 对应 `vn`）；
若仍没有，则该表层形式的第一行。
表层形式不在种子词典中的词赋予
`pos,chartype,*,*,*,*,char_count,first_char,last_char,*`。
否则，没有任何正确词成为词典词，词典词在训练中始终只是错误候选
（[lindera#1250][lindera-1250]）。

### 第 2 步：训练与导出

```bash
bash scripts/run_experiment.sh baseline
```

**输出：`work/experiments/baseline/export/`** — 包含用于 lindera 的 CRF 训练词典源文件
（`lex.csv`、`matrix.def` 等），`dict-src/` 是其副本。
在 UD Chinese GSD 和 GSDSimp 测试集上的评估结果写入
`work/experiments/baseline/result.txt`。

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

示例（最优已知配置，即默认值；见[最优已知结果](#最优已知结果)）：

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
| `F[2]` | 拼音 | wu3han4, ... | seed + 种子词 |
| `F[3]` | 繁体字 | 武漢, ... | seed + 种子词 |
| `F[4]` | 简体字 | 武汉, ... | seed + 种子词 |
| `F[5]` | 英文释义 | Wuhan, ... | seed + 种子词 |
| `F[6]` | 字符数 | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | 首字符 | 武, 不, 中, ... | seed + corpus |
| `F[8]` | 末字符 | 市, 的, 了, ... | seed + corpus |
| `F[9]` | 频率段 | high, mid, low, rare | seed + 种子词 |

“种子词”是表层形式在种子词典中的语料词，带有种子词典某一行的特征；
其他语料词的这些字段为 `*`。

> **注意：** 默认为 L2。仅用 GSD 训练集训练时，L1 在 GSD 的 dev 和 test 上的
> F1 略低于 L2（[lindera#1250][lindera-1250]）。

### 最优已知结果

使用 `scripts/evaluate.py` 在 UD Chinese GSD 和 GSDSimp 测试集（各 500 句）上的评估结果：

| 版本 | GSD P | GSD R | GSD F1 | GSDSimp F1 |
| ---- | ----- | ----- | ------ | ---------- |
| 0.1.2 | 0.7853 | 0.7668 | 0.7759 | 0.7798 |
| **0.1.3** | **0.8623** | **0.8987** | **0.8801** | **0.8906** |

0.1.2 由 lindera 2.3.3 在 GSD 上训练，没有任何正确词成为词典词。
0.1.3 按上述步骤在 GSD 和 GSDSimp 上训练。
使用默认的 `feature.def` 和 `REGULARIZATION=l2` 可复现 0.1.3 的结果，
其 `export/` 与 `dict-src/` 逐字节相同。
在正确切分的词上，GSD 的词性准确率为 0.6180（0.1.2 为 0.6288）：
词典的词性是 jieba 的词性，而 `evaluate.py` 将其与由 UPOS 映射得到的 PKU 词性比较。

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
├── jieba.csv              # 词典 CSV（由 build_jieba_csv.py 生成）
├── char.def               # 字符类别映射
├── matrix.def             # 连接代价矩阵（1x1 占位，仅静态使用）
├── unk.def                # 未知词定义
├── dicrc                  # MeCab 词典配置
├── dict-src/              # CRF 训练词典源（export/ 的副本）
├── scripts/
│   ├── build_jieba_csv.py # 下载 jieba + CC-CEDICT，生成 jieba.csv
│   ├── build_seed.py      # 生成 CRF 训练用 seed.csv
│   ├── convert_conllu.py  # UD CoNLL-U → 训练语料转换
│   ├── convert_sighan.py  # SIGHAN bakeoff 语料转换（可选）
│   ├── evaluate.py        # 在 UD 测试集上评估分词 F1
│   └── run_experiment.sh  # 训练/导出/评估一体化流水线
└── work/                  # 生成物（不提交）
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # 各实验的结果
        └── <name>/
            ├── export/    # lex.csv, matrix.def, unk.def, ...（9 个文件）
            ├── dict/      # 编译后的 lindera 词典
            └── result.txt # 评估结果
```

## 许可证

本仓库中的构建脚本和词典定义文件采用 [MIT 许可证](LICENSE) 发布。

`jieba.csv` 已预构建并提交到本仓库。
源文件（`dict.txt.big` 和 CC-CEDICT）在构建时下载到 `work/`，不包含在仓库中。

## 致谢

- [jieba][jieba] — 中文分词
- [CC-CEDICT][cedict] — 中英词典数据
- [CC-CEDICT-MeCab][cedict-mecab] — 结构参考
- [MeCab][mecab] — 形态素分析器
- [lindera][lindera] — 支持 CRF 训练的形态素分析器
- [UD Chinese GSD][ud-gsd] 和 [UD Chinese GSDSimp][ud-gsdsimp] — 训练与评估语料

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
[ud-gsdsimp]: https://github.com/UniversalDependencies/UD_Chinese-GSDSimp
[lindera-1250]: https://github.com/lindera/lindera/issues/1250
