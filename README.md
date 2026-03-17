# mecab-jieba

A Chinese (Mandarin) dictionary in MeCab/[lindera][lindera] CSV format,
built from [jieba][jieba]'s word frequency dictionary and enriched with
[CC-CEDICT][cedict] data (pinyin, traditional/simplified forms, English
definitions). Optionally, connection costs can be improved via CRF training
on the [UD Chinese GSD][ud-gsd] treebank.

## Outputs

| File | How to generate | Description |
| ---- | --------------- | ----------- |
| `jieba.csv` | `python3 scripts/build_csv.py` | Base dictionary CSV (584K entries) |
| `work/export/lex.csv` | `bash scripts/run_experiment.sh baseline` | CRF-trained dictionary CSV |
| `work/export/matrix.def` | `bash scripts/run_experiment.sh baseline` | CRF-trained connection cost matrix |

## Requirements

- Python 3.10+
- [lindera][lindera] 2.3.2+ with `train` feature (for CRF training only):
  `cargo install --path lindera-cli --features train`

## Building jieba.csv

```bash
python3 scripts/build_csv.py
```

Downloads `dict.txt.big` from jieba and CC-CEDICT from MDBG, then merges
them into `jieba.csv` (584K entries, ~25MB).

## CRF Training (Optional)

CRF training learns better connection costs from a segmentation corpus,
replacing the frequency-based costs in `jieba.csv`.

### Step 1: Prepare training data

```bash
# Download UD Chinese GSD
mkdir -p .tmp/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  .tmp/ud-chinese/UD_Chinese-GSD

# Generate seed dictionary: jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# Convert UD GSD training split to lindera corpus format
python3 scripts/convert_conllu.py \
  --input .tmp/ud-chinese/UD_Chinese-GSD/ \
  --output work/train/corpus.txt \
  --jieba-dict dict.txt.big \
  --split train
```

### Step 2: Train and export

```bash
bash scripts/run_experiment.sh baseline
```

**Output: `work/export/lex.csv` and `work/export/matrix.def`** — these are
the CRF-trained dictionary files to use with lindera.

### Training parameters

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `CORPUS` | `work/train/corpus.txt` | Training corpus path |
| `FEATURE_DEF` | `work/train/feature.def` | Feature template file |
| `CHAR_DEF` | `work/train/char.def` | Character category definitions |
| `UNK_DEF` | `work/train/unk.def` | Unknown word definitions |
| `LAMBDA` | `0.01` | Regularization coefficient |
| `MAX_ITER` | `100` | Maximum training iterations |
| `REGULARIZATION` | `l2` | `l1`, `l2`, or `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | L1 ratio for Elastic Net (0.0–1.0) |

Example:

```bash
LAMBDA=0.005 MAX_ITER=200 bash scripts/run_experiment.sh my_experiment
```

### Feature templates

Feature templates are defined in `work/train/feature.def`.
The default (best known) configuration:

```text
UNIGRAM:%F[0]          # POS tag (PKU tagset)
UNIGRAM U01:%t         # Character type (char.def category)
UNIGRAM U06:%F[6]      # Character count (1, 2, 3, 4+)
UNIGRAM U07:%F[7]      # First character of surface form
UNIGRAM U08:%F[8]      # Last character of surface form
UNIGRAM U09:%F[9]      # Frequency band (high/mid/low/rare)
BIGRAM B00:%L[0]/%R[0] # POS-to-POS transition
```

Feature field index reference:

| Index | Field | Values | Available in |
| ----- | ----- | ------ | ------------ |
| `F[0]` | POS tag | n, v, a, d, ... | seed + corpus |
| `F[1]` | Character type | CHINESE, ALPHA, NUMERIC, ... | seed + corpus |
| `F[2]` | Pinyin | wu3han4, ... | seed only |
| `F[3]` | Traditional form | 武漢, ... | seed only |
| `F[4]` | Simplified form | 武汉, ... | seed only |
| `F[5]` | Definition | Wuhan, ... | seed only |
| `F[6]` | Character count | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | First character | 武, 不, 中, ... | seed + corpus |
| `F[8]` | Last character | 市, 的, 了, ... | seed + corpus |
| `F[9]` | Frequency band | high, mid, low, rare | seed only |

> **Note:** L1 regularization prunes all bigram weights to zero when
> multiple unigram features are active. Use `REGULARIZATION=l2`
> (the default) or `elasticnet`.

### Best known results

Evaluated on UD Chinese GSD test set (481 sentences):

| Configuration | Regularization | P | R | F1 |
| ------------- | -------------- | - | - | -- |
| `%F[0]` + B00 | L2, λ=0.01 | 0.7803 | 0.7464 | 0.7630 |
| **`%F[0]` + `%t` + `%F[6-9]` + B00** | **L2, λ=0.01** | **0.7822** | **0.7644** | **0.7732** |

The default `work/train/feature.def` and `REGULARIZATION=l2` reproduce
the best result.

## MeCab CSV Format

Each entry in `jieba.csv`:

```text
surface,0,0,cost,pos,pinyin,traditional,simplified,definition
```

| Field | Description |
| ----- | ----------- |
| surface | Word surface form |
| left_id | Left context ID (0 for 1×1 matrix) |
| right_id | Right context ID (0 for 1×1 matrix) |
| cost | `-log10(freq / total) * 100` |
| pos | Part-of-speech tag (PKU tagset) |
| pinyin | Pinyin (CC-CEDICT; `*` if unavailable) |
| traditional | Traditional form (`*` if unavailable) |
| simplified | Simplified form (`*` if unavailable) |
| definition | English definition (`*` if unavailable) |

CC-CEDICT coverage: approximately 22% of jieba entries have CC-CEDICT data.

### Using jieba.csv with MeCab directly

```bash
mecab-dict-index -f utf-8 -t utf-8
echo "武汉市解除离汉离鄂通道管控措施" | mecab -d .
```

## PKU Part-of-Speech Tagset

| Tag | Description | Tag | Description |
| --- | ----------- | --- | ----------- |
| `n` | Noun | `v` | Verb |
| `ns` | Place name | `a` | Adjective |
| `nr` | Person name | `d` | Adverb |
| `nt` | Organization | `r` | Pronoun |
| `nz` | Other proper noun | `m` | Numeral |
| `q` | Measure word | `p` | Preposition |
| `c` | Conjunction | `u` | Auxiliary |

## Repository Structure

```text
mecab-jieba/
├── jieba.csv              # MeCab/lindera dictionary CSV (generated by build_csv.py)
├── char.def               # Character category mapping
├── matrix.def             # Connection cost matrix (1x1 dummy, static use only)
├── unk.def                # Unknown word definitions
├── dicrc                  # MeCab dictionary configuration
├── scripts/
│   ├── build_csv.py       # Download jieba + CC-CEDICT, generate jieba.csv
│   ├── build_seed.py      # Generate seed.csv for CRF training
│   ├── convert_conllu.py  # Convert UD CoNLL-U to training corpus
│   ├── convert_sighan.py  # Convert SIGHAN bakeoff corpus (optional)
│   ├── evaluate.py        # Evaluate segmentation F1 on UD GSD test
│   └── run_experiment.sh  # Full train/export/evaluate pipeline
└── work/                  # Generated artifacts (not committed)
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    ├── export/            # lex.csv, matrix.def (output)
    └── experiments/       # Per-experiment configs and results
```

## License

The build scripts and dictionary definition files in this repository are
licensed under the [MIT License](LICENSE).

`jieba.csv` is pre-built and committed to this repository. The source files
(`dict.txt.big` and CC-CEDICT) are downloaded at build time and are not
included.

## Acknowledgments

- [jieba][jieba] — Chinese text segmentation
- [CC-CEDICT][cedict] — Chinese-English dictionary data
- [CC-CEDICT-MeCab][cedict-mecab] — Reference for structure
- [MeCab][mecab] — Morphological analyzer
- [lindera][lindera] — Morphological analyzer with CRF training
- [UD Chinese GSD][ud-gsd] — Training and evaluation corpus

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
