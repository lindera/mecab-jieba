# mecab-jieba

A Chinese (Mandarin) dictionary in MeCab/[lindera][lindera] CSV format,
built from [jieba][jieba]'s word frequency dictionary and enriched with
[CC-CEDICT][cedict] data (pinyin, traditional/simplified forms, English
definitions). The dictionary source in `dict-src/`, which [lindera][lindera]
builds its Jieba dictionary from, has word and connection costs trained with
a CRF on the [UD Chinese GSD][ud-gsd] treebank and on the same sentences in
simplified script, [UD Chinese GSDSimp][ud-gsdsimp].

## Outputs

### Base dictionary

| File | How to generate | Description |
| ---- | --------------- | ----------- |
| `jieba.csv` | `python3 scripts/build_jieba_csv.py` | Base CSV, 584K entries |

### CRF-trained dictionary source (input for `lindera build`)

Run `bash scripts/run_experiment.sh <experiment_name>` to produce the
following files under `work/experiments/<experiment_name>/export/`:

| File | Description |
| ---- | ----------- |
| `export/lex.csv` | Lexicon entries with CRF-trained costs and context IDs |
| `export/matrix.def` | Connection cost matrix (one context ID per POS) |
| `export/unk.def` | Unknown word entries with CRF-trained costs |
| `export/char.def` | Character category definitions |
| `export/feature.def` | Feature templates the model was trained with |
| `export/rewrite.def` | Rewrite rules the model was trained with |
| `export/left-id.def` | Left context IDs |
| `export/right-id.def` | Right context IDs |
| `export/metadata.json` | Dictionary metadata |

These files are the source for `lindera build`. `dict-src/` holds them for
the `baseline` experiment of the released version.

## Requirements

- Python 3.10+
- [lindera][lindera] with the `train` feature, from commit `150db680` or
  later (for CRF training only):
  `cargo install --path lindera-cli --features train`. Earlier versions
  train without dictionary-word candidates and export 32767 for a
  connection without a trained weight ([lindera#1250][lindera-1250]).

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
3. Adds `〇` (U+3007), which `dict.txt.big` lacks, after the other entries,
   with the frequency and POS of `零`. `total_freq` stays that of
   `dict.txt.big`, so no other cost changes.
4. Enriches each entry with CC-CEDICT data where available (~22% coverage).
5. Outputs `jieba.csv` (584K entries, ~25MB) in MeCab CSV format.

## CRF Training (Optional)

CRF training learns better connection costs from a segmentation corpus,
replacing the frequency-based costs in `jieba.csv`.

### Step 1: Prepare training data

`make prepare` runs `scripts/build_jieba_csv.py` and the commands below.

```bash
# Download UD Chinese GSD and GSDSimp, at the commits dict-src is trained on
mkdir -p work/ud-chinese
git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git \
  work/ud-chinese/UD_Chinese-GSD
git -C work/ud-chinese/UD_Chinese-GSD checkout 22f73e87f7ddc530bb131a6b7587971a03e1a712
git clone https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git \
  work/ud-chinese/UD_Chinese-GSDSimp
git -C work/ud-chinese/UD_Chinese-GSDSimp checkout 4231dfd59866fa5999ad4a6bc1fdecd7985b3b59

# Generate seed dictionary: jieba.csv → work/train/seed.csv
python3 scripts/build_seed.py

# Convert the training splits of both to lindera corpus format
python3 scripts/convert_conllu.py \
  --input work/ud-chinese/UD_Chinese-GSD/ work/ud-chinese/UD_Chinese-GSDSimp/ \
  --output work/train/corpus.txt \
  --jieba-dict work/dict.txt.big \
  --seed work/train/seed.csv \
  --split train
```

`lindera train`, like MeCab's learner, makes a gold token a dictionary word
only when a seed row has its surface and the same whole feature. So
`convert_conllu.py` gives a gold token whose surface has seed rows the
feature of one of them: the first row with the gold POS, else the first
whose POS starts with the same letter (jieba's tags refine the PKU tags:
`uj` for `u`, `vn` for `v`), else the first row of the surface. A token
whose surface is not in the seed gets
`pos,chartype,*,*,*,*,char_count,first_char,last_char,*`. Without this, no
gold token is a dictionary word, and every dictionary word is only ever a
wrong candidate in training ([lindera#1250][lindera-1250]).

### Step 2: Train, export, build, and evaluate

```bash
bash scripts/run_experiment.sh baseline
```

Runs the full pipeline in four steps:

1. **Train** — Learn CRF model from corpus
2. **Export** — Generate dictionary source files →
   `work/experiments/baseline/export/`
3. **Build** — Compile dictionary → `work/experiments/baseline/dict/`
4. **Evaluate** — Score on the UD Chinese GSD and GSDSimp test sets →
   `work/experiments/baseline/result.txt`

The dictionary source files in `export/` (`lex.csv`, `matrix.def`, etc.)
are the input for `lindera build`; `dict-src/` is a copy of them.

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

Example (best known configuration, the defaults; see
[Best known results](#best-known-results)):

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
| `F[2]` | Pinyin | wu3han4, ... | seed + seed words |
| `F[3]` | Traditional form | 武漢, ... | seed + seed words |
| `F[4]` | Simplified form | 武汉, ... | seed + seed words |
| `F[5]` | Definition | Wuhan, ... | seed + seed words |
| `F[6]` | Character count | 1, 2, 3, 4+ | seed + corpus |
| `F[7]` | First character | 武, 不, 中, ... | seed + corpus |
| `F[8]` | Last character | 市, 的, 了, ... | seed + corpus |
| `F[9]` | Frequency band | high, mid, low, rare | seed + seed words |

"Seed words" are the corpus tokens whose surface is in the seed, which carry
a seed row's feature; the other corpus tokens have `*` in those fields.

> **Note:** The default is L2. Trained on the GSD training split alone, L1
> gave a slightly lower F1 than L2 on the GSD dev and test splits
> ([lindera#1250][lindera-1250]).

### Best known results

Evaluated with `scripts/evaluate.py` on the test splits of UD Chinese GSD
and GSDSimp (500 sentences each):

| Version | GSD P | GSD R | GSD F1 | GSDSimp F1 |
| ------- | ----- | ----- | ------ | ---------- |
| 0.1.2 | 0.7853 | 0.7668 | 0.7759 | 0.7798 |
| **0.1.3** | **0.8623** | **0.8987** | **0.8801** | **0.8906** |

0.1.2 was trained with lindera 2.3.3 on GSD, where no gold token was a
dictionary word. 0.1.3 is trained as above, on GSD and GSDSimp. The
default `feature.def` and `REGULARIZATION=l2` reproduce the 0.1.3 result;
its `export/` equals `dict-src/` byte for byte. POS accuracy on
correctly segmented words is 0.6180 on GSD (0.6288 with 0.1.2): the
dictionary's POS are jieba's tags, which `evaluate.py` compares with the
PKU tags it maps UPOS to.

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
├── jieba.csv              # Dictionary CSV (generated by build_jieba_csv.py)
├── char.def               # Character category mapping
├── matrix.def             # Connection cost matrix (1x1 dummy, static use only)
├── unk.def                # Unknown word definitions
├── dicrc                  # MeCab dictionary configuration
├── dict-src/              # CRF-trained source for lindera build (export/)
├── scripts/
│   ├── build_jieba_csv.py # Download jieba + CC-CEDICT, generate jieba.csv
│   ├── build_seed.py      # Generate seed.csv for CRF training
│   ├── convert_conllu.py  # Convert UD CoNLL-U to training corpus
│   ├── convert_sighan.py  # Convert SIGHAN bakeoff corpus (optional)
│   ├── evaluate.py        # Evaluate segmentation F1 on a UD test split
│   └── run_experiment.sh  # Full train/export/build/evaluate pipeline
└── work/                  # Generated artifacts (not committed)
    ├── train/             # seed.csv, corpus.txt, model.dat, feature.def, ...
    └── experiments/       # Per-experiment results
        └── <name>/
            ├── export/    # lex.csv, matrix.def, unk.def, ... (9 files)
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
- [UD Chinese GSD][ud-gsd] and [UD Chinese GSDSimp][ud-gsdsimp] — Training
  and evaluation corpora

[jieba]: https://github.com/fxsjy/jieba
[cedict]: https://cc-cedict.org/
[cedict-mecab]: https://github.com/lindera/CC-CEDICT-MeCab
[mecab]: https://taku910.github.io/mecab/
[lindera]: https://github.com/lindera/lindera
[ud-gsd]: https://github.com/UniversalDependencies/UD_Chinese-GSD
[ud-gsdsimp]: https://github.com/UniversalDependencies/UD_Chinese-GSDSimp
[lindera-1250]: https://github.com/lindera/lindera/issues/1250
