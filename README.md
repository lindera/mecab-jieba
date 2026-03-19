# mecab-jieba

A Chinese (Mandarin) dictionary in MeCab/[lindera][lindera] CSV format,
built from [jieba][jieba]'s word frequency dictionary and enriched with
[CC-CEDICT][cedict] data (pinyin, traditional/simplified forms, English
definitions). Optionally, connection costs can be improved via CRF training
on the [UD Chinese GSD][ud-gsd] treebank.

## Outputs

### Base dictionary

| File | How to generate | Description |
| ---- | --------------- | ----------- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | Base dictionary CSV (584K entries) |

### CRF-trained dictionary source (input for `lindera build`)

Run `bash scripts/run_experiment.sh <experiment_name>` to produce the
following files under `work/experiments/<experiment_name>/export/`:

| File | Description |
| ---- | ----------- |
| `export/lex.csv` | Lexicon entries with CRF-trained costs |
| `export/matrix.def` | Part-of-speech connection cost matrix |
| `export/char.def` | Character category definitions |
| `export/metadata.json` | Dictionary metadata |

These files are the source for `lindera build` in a separate repository.

## Requirements

- Python 3.10+
- [lindera][lindera] 2.3.2+ with `train` feature (for CRF training only):
  `cargo install --path lindera-cli --features train`

## Building jieba.csv

This step prepares the vocabulary file (lexicon) for the MeCab/lindera dictionary.
The script downloads two external data sources, merges them, and converts
each entry into MeCab CSV format with cost values computed from word frequencies.

```bash
python3 scripts/build_jieba_csv.py
```

1. Downloads jieba's `dict.txt.big` (word surface, frequency, POS) and
   CC-CEDICT (pinyin, traditional/simplified forms, English definitions)
   into `work/`.
2. Computes word cost as `int(-log10(freq / total_freq) * 100)`.
3. Enriches each entry with CC-CEDICT data where available (~22% coverage).
4. Outputs `jieba.csv` (584K entries, ~25MB) in MeCab CSV format.

## CRF Training (Optional)

CRF training learns better connection costs from a segmentation corpus,
replacing the frequency-based costs in `jieba.csv`.

### Step 1: Prepare training data

```bash
# Download UD Chinese GSD
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD

# Generate seed dictionary: jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# Convert UD GSD training split to lindera corpus format
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --split train
```

### Step 2: Train, export, build, and evaluate

```bash
bash scripts/run_experiment.sh baseline
```

Runs the full pipeline in four steps:

1. **Train** — Learn CRF model from corpus
2. **Export** — Generate dictionary source files → `work/experiments/baseline/export/`
3. **Build** — Compile dictionary → `work/experiments/baseline/dict/`
4. **Evaluate** — Score on UD Chinese GSD test set → `work/experiments/baseline/result.txt`

The dictionary source files in `export/` (`lex.csv`, `matrix.def`, etc.) are the input for `lindera build` in a separate repository.

### Training parameters

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `CORPUS` | `work/train/corpus.txt` | Training corpus path |
| `FEATURE_DEF` | `feature.def` | Feature template file |
| `CHAR_DEF` | `char.def` | Character category definitions |
| `UNK_DEF` | `unk.def` | Unknown word definitions |
| `LAMBDA` | `0.01` | Regularization coefficient |
| `MAX_ITER` | `100` | Maximum training iterations |
| `REGULARIZATION` | `l2` | `l1`, `l2`, or `elasticnet` |
| `ELASTIC_NET_L1_RATIO` | `0.5` | L1 ratio for Elastic Net (0.0–1.0) |

Example (best known configuration, F1=0.7732):

```bash
LAMBDA=0.01 MAX_ITER=100 REGULARIZATION=l2 bash scripts/run_experiment.sh baseline
```

### Feature templates

Feature templates are defined in `feature.def`.
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

The default `feature.def` and `REGULARIZATION=l2` reproduce
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
├── jieba.csv              # MeCab/lindera dictionary CSV (generated by build_jieba_csv.py)
├── char.def               # Character category mapping
├── matrix.def             # Connection cost matrix (1x1 dummy, static use only)
├── unk.def                # Unknown word definitions
├── dicrc                  # MeCab dictionary configuration
├── dict-src/              # CRF-trained dictionary source (copied from export/, input for lindera build)
├── scripts/
│   ├── build_jieba_csv.py # Download jieba + CC-CEDICT, generate jieba.csv
│   ├── build_seed.py      # Generate seed.csv for CRF training
│   ├── convert_conllu.py  # Convert UD CoNLL-U to training corpus
│   ├── convert_sighan.py  # Convert SIGHAN bakeoff corpus (optional)
│   ├── evaluate.py        # Evaluate segmentation F1 on UD GSD test
│   └── run_experiment.sh  # Full train/export/build/evaluate pipeline
└── work/                  # Generated artifacts (not committed)
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # Per-experiment results
        └── <name>/
            ├── export/    # lex.csv, matrix.def, char.def, metadata.json
            ├── dict/      # Compiled lindera dictionary
            └── result.txt # Evaluation results
```

## License

The build scripts and dictionary definition files in this repository are
licensed under the [MIT License](LICENSE).

`jieba.csv` is pre-built and committed to this repository. The source files
(`dict.txt.big` and CC-CEDICT) are downloaded into `work/` at build time
and are not included.

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
