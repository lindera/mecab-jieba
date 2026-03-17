# mecab-jieba

[jieba][jieba] の単語頻度辞書をベースに、[CC-CEDICT][cedict] データ（ピンイン・繁体字/簡体字・英語定義）で拡張した、MeCab/[lindera][lindera] CSV 形式の中国語（普通話）辞書です。オプションとして、[UD Chinese GSD][ud-gsd] ツリーバンクを用いた CRF 学習により接続コストを改善できます。

## 出力ファイル

### ベース辞書

| ファイル | 生成コマンド | 説明 |
| -------- | ------------ | ---- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | ベース辞書 CSV（584K エントリ） |

### CRF 学習済み辞書ソース（`lindera build` への入力）

`bash scripts/run_experiment.sh <experiment_name>` を実行すると、`work/experiments/<experiment_name>/dict-src/` 以下に次のファイルが生成されます。

| ファイル | 説明 |
| -------- | ---- |
| `dict-src/lex.csv` | CRF 学習済みコスト付き語彙エントリ |
| `dict-src/matrix.def` | 品詞間接続コスト行列 |
| `dict-src/char.def` | 文字種定義 |
| `dict-src/metadata.json` | 辞書メタデータ |

これらのファイルが、別リポジトリで `lindera build` に渡すソースです。

## 要件

- Python 3.10+
- CRF 学習を行う場合は [lindera][lindera] 2.3.2+（`train` フィーチャー付き）:
  `cargo install --path lindera-cli --features train`

## jieba.csv のビルド

MeCab/lindera 辞書のための語彙ファイル（レキシコン）を準備するステップです。
外部データソース 2 つをダウンロード・マージし、単語頻度からコストを算出して
MeCab CSV 形式に変換します。

```bash
python3 scripts/build_jieba_csv.py
```

1. jieba の `dict.txt.big`（表層形・頻度・品詞）と CC-CEDICT（ピンイン・繁体字/簡体字・英語定義）を `work/` にダウンロード。
2. 単語コストを `int(-log10(freq / total_freq) * 100)` で算出。
3. CC-CEDICT データで各エントリを拡充（カバレッジ約 22%）。
4. `jieba.csv`（584K エントリ、約 25MB）を MeCab CSV 形式で出力。

## CRF 学習（オプション）

CRF 学習はセグメンテーションコーパスから接続コストを学習し、`jieba.csv` の頻度ベースのコストを置き換えます。

### Step 1: 学習データの準備

```bash
# UD Chinese GSD をダウンロード
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD

# シード辞書を生成: jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# UD GSD 訓練データを lindera コーパス形式に変換
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --split train
```

### Step 2: 学習と辞書ソース生成

```bash
bash scripts/run_experiment.sh baseline
```

**出力: `work/experiments/baseline/dict-src/`** — lindera で使用する CRF 学習済み辞書ソースファイル（`lex.csv`、`matrix.def` 等）が格納されます。

### 学習パラメータ

環境変数で `run_experiment.sh` の動作を制御できます。

| 変数 | デフォルト | 説明 |
| ---- | ---------- | ---- |
| `CORPUS` | `work/train/corpus.txt` | 学習コーパスのパス |
| `FEATURE_DEF` | `work/train/feature.def` | 素性テンプレートファイル |
| `CHAR_DEF` | `work/train/char.def` | 文字種定義ファイル |
| `UNK_DEF` | `work/train/unk.def` | 未知語定義ファイル |
| `LAMBDA` | `0.01` | 正則化係数 |
| `MAX_ITER` | `100` | 最大学習イテレーション数 |
| `REGULARIZATION` | `l2` | `l1`、`l2`、または `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | Elastic Net の L1 比率（0.0–1.0） |

実行例（最良既知の設定、F1=0.7732）:

```bash
LAMBDA=0.01 MAX_ITER=100 REGULARIZATION=l2 bash scripts/run_experiment.sh baseline
```

### 素性テンプレート

素性テンプレートは `work/train/feature.def` で定義します。デフォルト（最良既知）の設定:

```text
UNIGRAM:%F[0]          # 品詞タグ（PKU タグセット）
UNIGRAM U01:%t         # 文字種（char.def カテゴリ）
UNIGRAM U06:%F[6]      # 文字数（1, 2, 3, 4+）
UNIGRAM U07:%F[7]      # 表層形の先頭文字
UNIGRAM U08:%F[8]      # 表層形の末尾文字
UNIGRAM U09:%F[9]      # 頻度帯（high/mid/low/rare）
BIGRAM B00:%L[0]/%R[0] # 品詞間遷移
```

素性フィールドインデックスの参照表:

| インデックス | フィールド | 値の例 | 利用可能な場所 |
| ------------ | ---------- | ------ | -------------- |
| `F[0]` | 品詞タグ | n, v, a, d, ... | seed + corpus |
| `F[1]` | 文字種 | CHINESE, ALPHA, NUMERIC, ... | seed + corpus |
| `F[2]` | ピンイン | wu3han4, ... | seed のみ |
| `F[3]` | 繁体字 | 武漢, ... | seed のみ |
| `F[4]` | 簡体字 | 武汉, ... | seed のみ |
| `F[5]` | 英語定義 | Wuhan, ... | seed のみ |
| `F[6]` | 文字数 | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | 先頭文字 | 武, 不, 中, ... | seed + corpus |
| `F[8]` | 末尾文字 | 市, 的, 了, ... | seed + corpus |
| `F[9]` | 頻度帯 | high, mid, low, rare | seed のみ |

> **注意:** L1 正則化は複数のユニグラム素性が有効な場合にすべてのバイグラム重みをゼロに刈り込みます。`REGULARIZATION=l2`（デフォルト）または `elasticnet` を使用してください。

### 最良既知スコア

UD Chinese GSD テストセット（481 文）での評価結果:

| 設定 | 正則化 | P | R | F1 |
| ---- | ------ | - | - | -- |
| `%F[0]` + B00 | L2, λ=0.01 | 0.7803 | 0.7464 | 0.7630 |
| **`%F[0]` + `%t` + `%F[6-9]` + B00** | **L2, λ=0.01** | **0.7822** | **0.7644** | **0.7732** |

デフォルトの `work/train/feature.def` と `REGULARIZATION=l2` で最良結果を再現できます。

## MeCab CSV フォーマット

`jieba.csv` の各エントリ:

```text
表層形,0,0,コスト,品詞,ピンイン,繁体字,簡体字,英語定義
```

| フィールド | 説明 |
| ---------- | ---- |
| surface | 表層形 |
| left_id | 左文脈 ID（1×1 行列では 0） |
| right_id | 右文脈 ID（1×1 行列では 0） |
| cost | `-log10(freq / total) * 100` |
| pos | 品詞タグ（PKU タグセット） |
| pinyin | ピンイン（CC-CEDICT; 未収録の場合 `*`） |
| traditional | 繁体字（未収録の場合 `*`） |
| simplified | 簡体字（未収録の場合 `*`） |
| definition | 英語定義（未収録の場合 `*`） |

CC-CEDICT のカバレッジ: jieba エントリの約 22% に CC-CEDICT データが付与されています。

### jieba.csv を MeCab で直接使用する

```bash
mecab-dict-index -f utf-8 -t utf-8
echo "武汉市解除离汉离鄂通道管控措施" | mecab -d .
```

## PKU 品詞タグセット

| タグ | 説明 | タグ | 説明 |
| ---- | ---- | ---- | ---- |
| `n` | 名詞 | `v` | 動詞 |
| `ns` | 地名 | `a` | 形容詞 |
| `nr` | 人名 | `d` | 副詞 |
| `nt` | 組織名 | `r` | 代名詞 |
| `nz` | その他固有名詞 | `m` | 数詞 |
| `q` | 量詞 | `p` | 前置詞 |
| `c` | 接続詞 | `u` | 助詞 |

## リポジトリ構成

```text
mecab-jieba/
├── jieba.csv              # MeCab/lindera 辞書 CSV（build_jieba_csv.py で生成）
├── char.def               # 文字種マッピング
├── matrix.def             # 接続コスト行列（1x1 ダミー、静的利用向け）
├── unk.def                # 未知語定義
├── dicrc                  # MeCab 辞書設定
├── dict-src/              # CRF 学習済み辞書ソース（lindera build への入力）
├── scripts/
│   ├── build_jieba_csv.py # jieba + CC-CEDICT ダウンロード → jieba.csv 生成
│   ├── build_seed.py      # CRF 学習用 seed.csv 生成
│   ├── convert_conllu.py  # UD CoNLL-U → 学習コーパス変換
│   ├── convert_sighan.py  # SIGHAN bakeoff コーパス変換（オプション）
│   ├── evaluate.py        # UD GSD テストセットで F1 評価
│   └── run_experiment.sh  # 学習/辞書ソース生成/ビルド/評価の一括パイプライン
└── work/                  # 生成物（コミット対象外）
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # 実験ごとの結果
        └── <name>/
            ├── dict-src/  # lex.csv, matrix.def, char.def, metadata.json
            ├── dict/      # コンパイル済み lindera 辞書
            └── result.txt # 評価結果
```

## ライセンス

このリポジトリのビルドスクリプトおよび辞書定義ファイルは [MIT ライセンス](LICENSE) で公開されています。

`jieba.csv` はビルド済みのものがリポジトリにコミットされています。ソースファイル（`dict.txt.big` および CC-CEDICT）はビルド時に `work/` にダウンロードされるため、リポジトリには含まれていません。

## 謝辞

- [jieba][jieba] — 中国語テキスト分割
- [CC-CEDICT][cedict] — 中英辞書データ
- [CC-CEDICT-MeCab][cedict-mecab] — 構造の参考
- [MeCab][mecab] — 形態素解析器
- [lindera][lindera] — CRF 学習対応形態素解析器
- [UD Chinese GSD][ud-gsd] — 学習・評価コーパス

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
