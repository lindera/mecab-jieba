#!/usr/bin/env python3
"""Generate seed.csv for lindera training from jieba.csv.

Usage:
    python scripts/build_seed.py [--input FILE] [--output FILE]

Reads jieba.csv and produces a seed dictionary with cost=0 (left_id=0,
right_id=0).  The CRF training step will learn proper costs and connection
IDs; this script merely prepares the surface/feature inventory.

Output format (same columns as jieba.csv, but cost forced to 0):

    surface,0,0,0,pos,pinyin,traditional,simplified,definition
"""

import argparse
import csv
import os
import sys
from collections import Counter


def get_char_type(surface: str) -> str:
    """Return character type of the first character, matching char.def categories."""
    if not surface:
        return "DEFAULT"
    cp = ord(surface[0])
    if cp in (0x0020, 0x00D0, 0x0009, 0x000B, 0x000A):
        return "SPACE"
    if (0x0030 <= cp <= 0x0039) or (0xFF10 <= cp <= 0xFF19) or (0x2070 <= cp <= 0x209F) or (0x2150 <= cp <= 0x218F):
        return "NUMERIC"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0xFF21 <= cp <= 0xFF3A) or (0xFF41 <= cp <= 0xFF5A) or (0x00C0 <= cp <= 0x00FF) or (0x0100 <= cp <= 0x017F) or (0x0180 <= cp <= 0x0236) or (0x1E00 <= cp <= 0x1EF9):
        return "ALPHA"
    if (0x4E00 <= cp <= 0x9FA5) or (0x3400 <= cp <= 0x4DB5) or (0x2E80 <= cp <= 0x2EF3) or (0x2F00 <= cp <= 0x2FD5) or cp == 0x3005 or (0xF900 <= cp <= 0xFA2D) or (0xFA30 <= cp <= 0xFA6A):
        return "CHINESE"
    if 0x3041 <= cp <= 0x309F:
        return "HIRAGANA"
    if (0x30A1 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF) or cp == 0x30FC or (0xFF66 <= cp <= 0xFF9F):
        return "KATAKANA"
    if 0x0400 <= cp <= 0x050F:
        return "CYRILLIC"
    if 0x0374 <= cp <= 0x03FB:
        return "GREEK"
    return "SYMBOL"


OUTPUT_FILE = "work/train/seed.csv"
INPUT_FILE = "jieba.csv"


def escape_csv_field(value: str) -> str:
    """Escape a CSV field value if it contains commas or quotes."""
    if "," in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def build_seed(input_path: str, output_path: str) -> None:
    """Read *input_path* (jieba.csv) and write seed.csv to *output_path*."""
    pos_counter: Counter[str] = Counter()
    total = 0

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with (
        open(input_path, encoding="utf-8", newline="") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 5:
                continue

            surface = row[0]
            # row[1] = left_id, row[2] = right_id, row[3] = cost (all ignored)
            pos = row[4]
            char_type = get_char_type(surface)
            # Insert char_type as second feature field: pos, chartype, pinyin, ...
            features = [pos, char_type] + row[5:]

            pos_counter[pos] += 1
            total += 1

            # Escape fields that may contain commas or quotes
            escaped_features = [escape_csv_field(f) for f in features]

            fout.write(f"{surface},0,0,0,{','.join(escaped_features)}\n")

    # Print statistics
    print(f"Total entries: {total}")
    print(f"Output: {output_path}")
    print(f"\nPOS distribution ({len(pos_counter)} tags):")
    for pos, count in pos_counter.most_common():
        pct = count * 100 / total if total else 0
        print(f"  {pos:8s}  {count:>8d}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate seed.csv for lindera training from jieba.csv"
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"Input jieba.csv file (default: {INPUT_FILE} in repo root)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output seed.csv file (default: {OUTPUT_FILE} in repo root)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repository root (parent of scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = args.input if args.input else os.path.join(repo_root, INPUT_FILE)
    output_path = args.output if args.output else os.path.join(repo_root, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    build_seed(input_path, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
