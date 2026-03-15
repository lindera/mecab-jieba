# mecab-jieba

A MeCab dictionary for Chinese (Mandarin) text segmentation
based on [jieba][jieba]'s word frequency dictionary,
enriched with [CC-CEDICT][cedict] data
(pinyin, traditional/simplified forms, English definitions).

Supports both static dictionary usage (MeCab/lindera)
and CRF-based training with [lindera][lindera] for
learned connection costs and POS transitions.

## Requirements

- Python 3.10+
- [lindera][lindera] 2.3.1+ (for CRF training and tokenization)
- [UD Chinese GSD][ud-gsd] (for training corpus and evaluation)

## Repository Structure

```text
mecab-jieba/
├── README.md
├── LICENSE
├── jieba.csv              # MeCab dictionary CSV (pre-built)
├── char.def               # Character category mapping
├── matrix.def             # Connection cost matrix (1x1 dummy)
├── unk.def                # Unknown word definitions
├── dicrc                  # MeCab dictionary configuration
├── dict.txt.big           # jieba frequency dictionary (downloaded)
├── scripts/
│   ├── build_csv.py       # Download jieba + CC-CEDICT, generate jieba.csv
│   ├── build_seed.py      # Generate seed.csv for CRF training
│   ├── convert_conllu.py  # Convert UD CoNLL-U to training corpus
│   ├── convert_sighan.py  # Convert SIGHAN bakeoff corpus (optional)
│   ├── evaluate.py        # Evaluate segmentation F1 on UD GSD test
│   └── run_experiment.sh  # Full train/export/build/evaluate pipeline
└── work/
    ├── train/             # Training data and CRF model
    ├── export/            # Exported dictionary (intermediate)
    ├── build/             # Compiled lindera dictionary
    └── experiments/       # Experiment results and configs
```

## Build (Static Dictionary)

### 1. Generate jieba.csv

Run the build script to download jieba's `dict.txt.big`
and CC-CEDICT, then merge and convert to MeCab CSV format:

```bash
python3 scripts/build_csv.py
```

This will:

1. Download `dict.txt.big` from the jieba repository
2. Download `cedict_1_0_ts_utf-8_mdbg.txt.gz` from MDBG
3. Merge CC-CEDICT data into jieba entries
4. Write `jieba.csv`

Options:

```bash
# Use local files instead of downloading
python3 scripts/build_csv.py \
  --jieba-file /path/to/dict.txt \
  --cedict-file /path/to/cedict_ts.u8

# Specify output file name
python3 scripts/build_csv.py --output my_dict.csv
```

### 2. Build MeCab dictionary

Compile the dictionary using `mecab-dict-index`:

```bash
mecab-dict-index -f utf-8 -t utf-8
```

### 3. Test with MeCab

```bash
echo "武汉市解除离汉离鄂通道管控措施" | mecab -d .
```

## CRF Training (lindera)

CRF training learns connection costs (POS transition weights)
from annotated corpora, improving segmentation accuracy over
the static frequency-based dictionary.

### Prerequisites

Install lindera with training support:

```bash
cargo install --path /path/to/lindera/lindera-cli --features train
```

### Step 1: Prepare Training Data

Download UD Chinese GSD and convert to training corpus format:

```bash
# Clone UD Chinese GSD
mkdir -p .tmp/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  .tmp/ud-chinese/UD_Chinese-GSD

# Convert to lindera corpus format
python3 scripts/convert_conllu.py \
  --input .tmp/ud-chinese/UD_Chinese-GSD/zh_gsd-ud-train.conllu \
  --output work/train/corpus.txt \
  --jieba-dict dict.txt.big \
  --split train

# Generate seed dictionary from jieba.csv
python3 scripts/build_seed.py
```

### Step 2: Train and Build

Run the full pipeline (train CRF model, export, build compiled dictionary):

```bash
bash scripts/run_experiment.sh baseline
```

The script executes four steps:

1. **Train** - CRF model training with lindera
2. **Export** - Export model to dictionary files (lex.csv, matrix.def, etc.)
3. **Build** - Compile dictionary to lindera binary format
4. **Evaluate** - Measure F1 score on UD Chinese GSD test set

### Step 3: Tokenize

```bash
lindera tokenize -d work/dict "武汉市解除离汉离鄂通道管控措施"
```

### Training Parameters

Parameters are configured via environment variables:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `CORPUS` | `work/train/corpus.txt` | Training corpus path |
| `FEATURE_DEF` | `work/train/feature.def` | Feature template definitions |
| `CHAR_DEF` | `work/train/char.def` | Character category definitions |
| `UNK_DEF` | `work/train/unk.def` | Unknown word definitions |
| `LAMBDA` | `0.01` | Regularization coefficient |
| `MAX_ITER` | `100` | Maximum training iterations |
| `REGULARIZATION` | `l1` | Regularization type: `l1`, `l2`, or `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | L1 ratio for Elastic Net (0.0-1.0) |

Example with custom parameters:

```bash
REGULARIZATION=elasticnet ELASTIC_NET_L1_RATIO=0.8 LAMBDA=0.01 \
  bash scripts/run_experiment.sh my_experiment
```

### Feature Templates

Feature templates are defined in `work/train/feature.def`:

```text
UNIGRAM:%F[0]              # POS tag unigram
UNIGRAM U01:%t             # Character type unigram (optional)
BIGRAM B00:%L[0]/%R[0]     # POS-to-POS transition bigram
```

| Template | Description | Notes |
| -------- | ----------- | ----- |
| `%F[0]` | POS tag (PKU tagset) | Always used |
| `%t` | Character type category | Requires Elastic Net regularization |
| `%L[0]/%R[0]` | Left/right POS transition | Core bigram feature |

> **Note:** Adding `%t` with L1 regularization causes all bigram
> weights to be pruned. Use `REGULARIZATION=elasticnet` with
> `ELASTIC_NET_L1_RATIO=0.8` when using `%t` features.

### Evaluation

Evaluate a trained dictionary against UD Chinese GSD test set:

```bash
python3 scripts/evaluate.py \
  --test-file .tmp/ud-chinese/UD_Chinese-GSD/zh_gsd-ud-test.conllu \
  --dict-dir work/dict \
  --jieba-dict dict.txt.big
```

Metrics: span-based micro-averaged Precision, Recall, and F1.

### Best Known Results

Evaluated on UD Chinese GSD test set (481 sentences):

| Configuration | P | R | F1 |
| ------------- | - | - | -- |
| L1, %F[0]+B00 (baseline) | 0.7188 | 0.7355 | 0.7271 |
| Elastic Net (r=0.8), %F[0]+%t+B00 | 0.7255 | 0.7320 | 0.7287 |

## MeCab CSV Format

Each entry in `jieba.csv` follows this format:

```text
surface,0,0,cost,pos,pinyin,traditional,simplified,definition
```

| Field | Description |
| ----- | ----------- |
| surface | Word surface form |
| left_id | Left context ID (0 for 1x1) |
| right_id | Right context ID (0 for 1x1) |
| cost | -log10(freq / total) * 100 |
| pos | Part-of-speech tag (PKU tagset) |
| pinyin | Pinyin (CC-CEDICT, * if N/A) |
| traditional | Traditional form (CC-CEDICT) |
| simplified | Simplified form (CC-CEDICT) |
| definition | English definition (CC-CEDICT) |

CC-CEDICT coverage: approximately 22% of jieba entries
have CC-CEDICT data. Unmatched entries use `*` for
pinyin, traditional, simplified, and definition fields.

### Cost Estimation

Word cost is derived from jieba's frequency data:

```text
cost = int(-log10(freq / total_freq) * 100)
```

High-frequency words get lower costs, making them
preferred by MeCab's Viterbi algorithm:

| Word | Frequency | Cost |
| ---- | --------- | ---- |
| 是 | 51,365,674 | 203 |
| 的 | 49,577,040 | 243 |
| 中国 | 45,459,016 | 282 |

### PKU Part-of-Speech Tagset

| Tag | Description | Tag | Description |
| --- | ----------- | --- | ----------- |
| `n` | Noun | `v` | Verb |
| `ns` | Place name | `a` | Adjective |
| `nr` | Person name | `d` | Adverb |
| `nt` | Organization | `r` | Pronoun |
| `nz` | Other proper noun | `m` | Numeral |
| `q` | Measure word | `p` | Preposition |
| `c` | Conjunction | `u` | Auxiliary |

## License

The build script and dictionary definition files in this
repository are licensed under the [MIT License](LICENSE).

Note: `jieba.csv` is pre-built and committed to this
repository. The source files (`dict.txt.big` and CC-CEDICT)
are downloaded at build time and are not included.

## Acknowledgments

- [jieba][jieba] - Chinese text segmentation
- [CC-CEDICT][cedict] - Chinese-English dictionary data
- [CC-CEDICT-MeCab][cedict-mecab] - Reference for structure
- [MeCab][mecab] - Morphological analyzer
- [lindera][lindera] - Morphological analyzer with CRF training
- [UD Chinese GSD][ud-gsd] - Training and evaluation corpus

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
