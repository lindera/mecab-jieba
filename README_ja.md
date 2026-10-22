# mecab-jieba

[jieba][jieba] の単語頻度辞書をベースに、[CC-CEDICT][cedict] データ
（ピンイン・繁体字/簡体字・英語定義）で拡張した、MeCab/[lindera][lindera]
CSV 形式の中国語（普通話）辞書です。
[lindera][lindera] が Jieba 辞書をビルドする `dict-src/` の辞書ソースは、
単語コストと接続コストを、[UD Chinese GSD][ud-gsd] ツリーバンクと、
同じ文の簡体字版 [UD Chinese GSDSimp][ud-gsdsimp] で CRF 学習したものです。

## 出力ファイル

### ベース辞書

| ファイル | 生成コマンド | 説明 |
| -------- | ------------ | ---- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | ベース辞書 CSV（584K エントリ） |

### CRF 学習済み辞書ソース（`lindera build` への入力）

`bash scripts/run_experiment.sh <experiment_name>` を実行すると、
`work/experiments/<experiment_name>/export/` 以下に次のファイルが生成されます。

| ファイル | 説明 |
| -------- | ---- |
| `export/lex.csv` | CRF 学習済みコストと文脈 ID 付きの語彙エントリ |
| `export/matrix.def` | 接続コスト行列（品詞ごとに 1 つの文脈 ID） |
| `export/unk.def` | CRF 学習済みコスト付きの未知語エントリ |
| `export/char.def` | 文字種定義 |
| `export/feature.def` | 学習に使った素性テンプレート |
| `export/rewrite.def` | 学習に使った書き換え規則 |
| `export/left-id.def` | 左文脈 ID |
| `export/right-id.def` | 右文脈 ID |
| `export/metadata.json` | 辞書メタデータ |

これらのファイルが `lindera build` に渡すソースです。
`dict-src/` は、リリースした版の `baseline` 実験のこれらのファイルです。

## 要件

- Python 3.10+
- CRF 学習を行う場合は、commit `150db680` 以降の [lindera][lindera]
  （`train` フィーチャー付き）: `cargo install --path lindera-cli --features train`。
  それより前の版は、辞書語の候補なしで学習し、学習した重みのない接続に
  32767 を書き出します（[lindera#1250][lindera-1250]）。

## jieba.csv のビルド

MeCab/lindera 辞書のための語彙ファイル（レキシコン）を準備するステップです。
外部データソース 2 つをダウンロード・マージし、単語頻度からコストを算出して
MeCab CSV 形式に変換します。

```bash
python3 scripts/build_jieba_csv.py
```

1. jieba の `dict.txt.big`（表層形・頻度・品詞）と CC-CEDICT（ピンイン・繁体字/簡体字・英語定義）を `work/` にダウンロード。
2. 単語コストを `int(-log10(freq / total_freq) * 100)` で算出。
3. `dict.txt.big` にない `〇`（U+3007）を、`零` と同じ頻度・品詞で末尾に追加。
   `total_freq` は `dict.txt.big` のままなので、ほかのコストは変わらない。
4. CC-CEDICT データで各エントリを拡充（カバレッジ約 22%）。
5. `jieba.csv`（584K エントリ、約 25MB）を MeCab CSV 形式で出力。

## CRF 学習（オプション）

CRF 学習はセグメンテーションコーパスから接続コストを学習し、
`jieba.csv` の頻度ベースのコストを置き換えます。

### Step 1: 学習データの準備

`make prepare` は `scripts/build_jieba_csv.py` と次のコマンドを実行します。

```bash
# UD Chinese GSD と GSDSimp を、dict-src の学習に使った commit でダウンロード
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD
git -C work/ud-chinese/UD_Chinese-GSD checkout 22f73e87f7ddc530bb131a6b7587971a03e1a712
git clone https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git \
  work/ud-chinese/UD_Chinese-GSDSimp
git -C work/ud-chinese/UD_Chinese-GSDSimp checkout 4231dfd59866fa5999ad4a6bc1fdecd7985b3b59

# シード辞書を生成: jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# 両方の訓練データを lindera コーパス形式に変換
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ work/ud-chinese/UD_Chinese-GSDSimp/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --seed work/train/seed.csv \
  --split train
```

`lindera train` は MeCab の学習器と同じく、シードの行が表層形と素性全体の
両方で一致するときだけ、正解の語を辞書の語として扱います。
そのため `convert_conllu.py` は、表層形がシードにある正解の語に、
その行のどれかの素性を付けます。
正解の品詞と同じ最初の行、なければ品詞の頭文字が同じ最初の行
（jieba の品詞は PKU の品詞を細分したもの: `u` に対する `uj`、`v` に対する `vn`）、
なければその表層形の最初の行です。
表層形がシードにない語には
`pos,chartype,*,*,*,*,char_count,first_char,last_char,*` を付けます。
これがないと、正解の語は 1 つも辞書の語にならず、辞書の語は学習で
常に不正解の候補になります（[lindera#1250][lindera-1250]）。

### Step 2: 学習・辞書ソース生成・ビルド・評価

```bash
bash scripts/run_experiment.sh baseline
```

以下の 4 ステップを一括実行します:

1. **Train** — CRF モデルを学習
2. **Export** — 辞書ソースファイルを生成 → `work/experiments/baseline/export/`
3. **Build** — コンパイル済み辞書を生成 → `work/experiments/baseline/dict/`
4. **Evaluate** — UD Chinese GSD と GSDSimp のテストセットで F1 評価 →
   `work/experiments/baseline/result.txt`

`export/` ディレクトリ内の辞書ソースファイル（`lex.csv`、`matrix.def` 等）が
`lindera build` に渡す入力で、`dict-src/` はその写しです。

### 学習パラメータ

環境変数で `run_experiment.sh` の動作を制御できます。

| 変数 | デフォルト | 説明 |
| ---- | ---------- | ---- |
| `CORPUS` | `work/train/corpus.txt` | 学習コーパスのパス |
| `FEATURE_DEF` | `feature.def` | 素性テンプレートファイル |
| `CHAR_DEF` | `char.def` | 文字種定義ファイル |
| `UNK_DEF` | `unk.def` | 未知語定義ファイル |
| `LAMBDA` | `0.01` | 正則化係数 |
| `MAX_ITER` | `100` | 最大学習イテレーション数 |
| `REGULARIZATION` | `l2` | `l1`、`l2`、または `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | Elastic Net の L1 比率（0.0–1.0） |

実行例（最良既知の設定。デフォルトと同じ。[最良既知スコア](#最良既知スコア) を参照）:

```bash
LAMBDA=0.01 MAX_ITER=100 REGULARIZATION=l2 bash scripts/run_experiment.sh baseline
```

### 素性テンプレート

素性テンプレートは `feature.def` で定義します。デフォルト（最良既知）の設定:

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
| `F[2]` | ピンイン | wu3han4, ... | seed + シードの語 |
| `F[3]` | 繁体字 | 武漢, ... | seed + シードの語 |
| `F[4]` | 簡体字 | 武汉, ... | seed + シードの語 |
| `F[5]` | 英語定義 | Wuhan, ... | seed + シードの語 |
| `F[6]` | 文字数 | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | 先頭文字 | 武, 不, 中, ... | seed + corpus |
| `F[8]` | 末尾文字 | 市, 的, 了, ... | seed + corpus |
| `F[9]` | 頻度帯 | high, mid, low, rare | seed + シードの語 |

「シードの語」は、表層形がシードにあるコーパスの語で、シードの行の素性を持ちます。
それ以外のコーパスの語では、これらのフィールドは `*` です。

> **注意:** デフォルトは L2 です。GSD の訓練データだけで学習したとき、
> L1 は GSD の dev と test で L2 より F1 が少し低くなりました
> （[lindera#1250][lindera-1250]）。

### 最良既知スコア

UD Chinese GSD と GSDSimp のテストセット（各 500 文）での
`scripts/evaluate.py` の評価結果:

| 版 | GSD P | GSD R | GSD F1 | GSDSimp F1 |
| -- | ----- | ----- | ------ | ---------- |
| 0.1.2 | 0.7853 | 0.7668 | 0.7759 | 0.7798 |
| **0.1.3** | **0.8623** | **0.8987** | **0.8801** | **0.8906** |

0.1.2 は lindera 2.3.3 で GSD から学習したもので、正解の語は 1 つも辞書の語に
なっていませんでした。0.1.3 は上の手順で GSD と GSDSimp から学習したものです。
デフォルトの `feature.def` と `REGULARIZATION=l2` で 0.1.3 の結果を再現でき、
その `export/` は `dict-src/` とバイト単位で一致します。
正しく分割した語の品詞の正解率は、GSD で 0.6180 です（0.1.2 では 0.6288）。
辞書の品詞は jieba の品詞で、`evaluate.py` はそれを UPOS から写した
PKU の品詞と比べるためです。

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
├── jieba.csv              # 辞書 CSV（build_jieba_csv.py で生成）
├── char.def               # 文字種マッピング
├── matrix.def             # 接続コスト行列（1x1 ダミー、静的利用向け）
├── unk.def                # 未知語定義
├── dicrc                  # MeCab 辞書設定
├── dict-src/              # CRF 学習済み辞書ソース（export/ の写し）
├── scripts/
│   ├── build_jieba_csv.py # jieba + CC-CEDICT ダウンロード → jieba.csv 生成
│   ├── build_seed.py      # CRF 学習用 seed.csv 生成
│   ├── convert_conllu.py  # UD CoNLL-U → 学習コーパス変換
│   ├── convert_sighan.py  # SIGHAN bakeoff コーパス変換（オプション）
│   ├── evaluate.py        # UD のテストセットで F1 評価
│   └── run_experiment.sh  # 学習/辞書ソース生成/ビルド/評価の一括パイプライン
└── work/                  # 生成物（コミット対象外）
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # 実験ごとの結果
        └── <name>/
            ├── export/    # lex.csv, matrix.def, unk.def, ...（9 ファイル）
            ├── dict/      # コンパイル済み lindera 辞書
            └── result.txt # 評価結果
```

## ライセンス

このリポジトリのビルドスクリプトおよび辞書定義ファイルは [MIT ライセンス](LICENSE) で公開されています。

`jieba.csv` はビルド済みのものがリポジトリにコミットされています。
ソースファイル（`dict.txt.big` および CC-CEDICT）はビルド時に `work/` に
ダウンロードされるため、リポジトリには含まれていません。

## 謝辞

- [jieba][jieba] — 中国語テキスト分割
- [CC-CEDICT][cedict] — 中英辞書データ
- [CC-CEDICT-MeCab][cedict-mecab] — 構造の参考
- [MeCab][mecab] — 形態素解析器
- [lindera][lindera] — CRF 学習対応形態素解析器
- [UD Chinese GSD][ud-gsd] と [UD Chinese GSDSimp][ud-gsdsimp] — 学習・評価コーパス

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
[ud-gsdsimp]: https://github.com/UniversalDependencies/UD_Chinese-GSDSimp
[lindera-1250]: https://github.com/lindera/lindera/issues/1250
