# mecab-jieba

A MeCab dictionary for Chinese (Mandarin) text segmentation based on [jieba](https://github.com/fxsjy/jieba)'s word frequency dictionary, enriched with [CC-CEDICT](https://cc-cedict.org/) data (pinyin, traditional/simplified forms, English definitions).

## Requirements

- Python 3.10+

## Repository Structure

```text
mecab-jieba/
├── README.md
├── LICENSE
├── .gitignore
├── scripts/
│   └── build_csv.py     # Download jieba + CC-CEDICT and convert to MeCab CSV
├── jieba.csv             # MeCab dictionary CSV (pre-built)
├── char.def              # Character category mapping for Chinese text
├── matrix.def            # 1x1 dummy connection cost matrix
├── unk.def               # Unknown word definitions
└── dicrc                 # MeCab dictionary configuration
```

## Build

### 1. Generate jieba.csv

Run the build script to download jieba's `dict.txt.big` and CC-CEDICT, then merge and convert to MeCab CSV format:

```bash
python3 scripts/build_csv.py
```

This will:

1. Download `dict.txt.big` from the jieba repository
2. Download `cedict_1_0_ts_utf-8_mdbg.txt.gz` from MDBG
3. Merge CC-CEDICT data (pinyin, traditional/simplified, definition) into jieba entries
4. Write `jieba.csv`

Options:

```bash
# Use local files instead of downloading
python3 scripts/build_csv.py --jieba-file /path/to/dict.txt --cedict-file /path/to/cedict_ts.u8

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

## MeCab CSV Format

Each entry in `jieba.csv` follows this format:

```text
surface,left_id,right_id,cost,pos,pinyin,traditional,simplified,definition
```

| Field         | Description                                            |
| ------------- | ------------------------------------------------------ |
| `surface`     | Word surface form                                      |
| `left_id`     | Left context ID (always `0` for 1x1 matrix)           |
| `right_id`    | Right context ID (always `0` for 1x1 matrix)          |
| `cost`        | Word cost: `int(-log10(freq / total_freq) * 100)`     |
| `pos`         | Part-of-speech tag (PKU tagset)                        |
| `pinyin`      | Pinyin pronunciation (from CC-CEDICT, `*` if N/A)     |
| `traditional` | Traditional Chinese form (from CC-CEDICT, `*` if N/A)  |
| `simplified`  | Simplified Chinese form (from CC-CEDICT, `*` if N/A)  |
| `definition`  | English definition (from CC-CEDICT, `*` if N/A)       |

CC-CEDICT coverage: approximately 22% of jieba entries have CC-CEDICT data. Unmatched entries use `*` for pinyin, traditional, simplified, and definition fields.

### Cost Estimation

Word cost is derived from jieba's frequency data:

```text
cost = int(-log10(freq / total_freq) * 100)
```

High-frequency words get lower costs, making them preferred by MeCab's Viterbi algorithm:

| Word                     | Frequency  | Cost |
| ------------------------ | ---------- | ---- |
| 是 (to be)               | 51,365,674 | 203  |
| 的 (possessive particle) | 49,577,040 | 243  |
| 中国 (China)             | 45,459,016 | 282  |

### PKU Part-of-Speech Tagset

| Tag  | Description        | Tag  | Description |
| ---- | ------------------ | ---- | ----------- |
| `n`  | Noun               | `v`  | Verb        |
| `ns` | Place name         | `a`  | Adjective   |
| `nr` | Person name        | `d`  | Adverb      |
| `nt` | Organization       | `r`  | Pronoun     |
| `nz` | Other proper noun  | `m`  | Numeral     |
| `q`  | Measure word       | `p`  | Preposition |
| `c`  | Conjunction        | `u`  | Auxiliary   |

## License

The build script and dictionary definition files in this repository are licensed under the [MIT License](LICENSE).

Note: `jieba.csv` is pre-built and committed to this repository. The source files (`dict.txt.big` and CC-CEDICT) are downloaded at build time and are not included.

## Acknowledgments

- [jieba](https://github.com/fxsjy/jieba) - Chinese text segmentation
- [CC-CEDICT](https://cc-cedict.org/) - Chinese-English dictionary data
- [CC-CEDICT-MeCab](https://github.com/lindera/CC-CEDICT-MeCab) - Reference for MeCab dictionary structure
- [MeCab](https://taku910.github.io/mecab/) - Morphological analyzer
