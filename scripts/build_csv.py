#!/usr/bin/env python3
"""Download jieba dict.txt.big and CC-CEDICT, then convert to MeCab CSV format.

Usage:
    python scripts/build_csv.py [--output FILE]

The script downloads jieba's dict.txt.big and CC-CEDICT (cedict_ts.u8) from
GitHub, merges them, and converts each entry to MeCab CSV format:

    surface,left_id,right_id,cost,pos,pinyin,traditional,simplified,definition

Cost is computed as: int(-log10(freq / total_freq) * 100)
"""

import argparse
import math
import os
import re
import sys
import urllib.request

JIEBA_DICT_URL = (
    "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"
)
CEDICT_URL = (
    "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
)
OUTPUT_FILE = "jieba.csv"


def download(url: str, dest: str) -> None:
    """Download a file from *url* to *dest*."""
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def parse_jieba_dict(path: str) -> list[tuple[str, int, str]]:
    """Parse jieba dict.txt and return list of (surface, freq, pos)."""
    entries: list[tuple[str, int, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            surface = parts[0]
            freq = int(parts[1])
            pos = parts[2] if len(parts) >= 3 else "*"
            entries.append((surface, freq, pos))
    return entries


def parse_cedict(path: str) -> dict[str, tuple[str, str, str, str]]:
    """Parse CC-CEDICT and return a dict keyed by simplified form.

    Returns:
        {simplified: (pinyin, traditional, simplified, definition)}

    When multiple entries exist for the same simplified form,
    definitions are joined with " / ".
    """
    import gzip

    pattern = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/$")
    result: dict[str, tuple[str, str, str, str]] = {}

    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue
            traditional = match.group(1)
            simplified = match.group(2)
            pinyin = match.group(3)
            definition = match.group(4)

            if simplified in result:
                # Merge multiple entries: join definitions
                prev = result[simplified]
                merged_def = prev[3] + " / " + definition
                result[simplified] = (prev[0], prev[1], prev[2], merged_def)
            else:
                result[simplified] = (pinyin, traditional, simplified, definition)

            # Also index by traditional form if different
            if traditional != simplified and traditional not in result:
                result[traditional] = (pinyin, traditional, simplified, definition)

    return result


def compute_cost(freq: int, total_freq: int) -> int:
    """Compute MeCab word cost from jieba frequency.

    cost = int(-log10(freq / total_freq) * 100)

    Lower cost means the word is preferred by the Viterbi algorithm.
    For freq=0, use a high penalty cost.
    """
    if freq <= 0 or total_freq <= 0:
        return 10000
    return int(-math.log10(freq / total_freq) * 100)


def escape_csv_field(value: str) -> str:
    """Escape a CSV field value if it contains commas or quotes."""
    if "," in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def convert_to_csv(
    entries: list[tuple[str, int, str]],
    cedict: dict[str, tuple[str, str, str, str]],
    output: str,
) -> None:
    """Convert parsed entries to MeCab CSV and write to *output*."""
    total_freq = sum(freq for _, freq, _ in entries)
    matched = 0

    print(f"Total jieba entries: {len(entries)}, total frequency: {total_freq}")
    print(f"CC-CEDICT entries loaded: {len(cedict)}")

    with open(output, "w", encoding="utf-8") as f:
        for surface, freq, pos in entries:
            cost = compute_cost(freq, total_freq)
            cedict_entry = cedict.get(surface)
            if cedict_entry:
                pinyin, traditional, simplified, definition = cedict_entry
                definition = escape_csv_field(definition)
                matched += 1
            else:
                pinyin = "*"
                traditional = "*"
                simplified = "*"
                definition = "*"
            # MeCab CSV: surface,left_id,right_id,cost,pos,pinyin,traditional,simplified,definition
            f.write(
                f"{surface},0,0,{cost},{pos},{pinyin},{traditional},{simplified},{definition}\n"
            )

    print(f"Wrote {len(entries)} entries to {output}")
    print(
        f"CC-CEDICT matched: {matched}/{len(entries)} "
        f"({matched * 100 // len(entries)}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert jieba dict.txt.big to MeCab CSV format with CC-CEDICT data"
    )
    parser.add_argument(
        "--jieba-url",
        default=JIEBA_DICT_URL,
        help="URL to download jieba dict.txt.big",
    )
    parser.add_argument(
        "--jieba-file",
        default=None,
        help="Path to a local jieba dict.txt file (skip download)",
    )
    parser.add_argument(
        "--cedict-url",
        default=CEDICT_URL,
        help="URL to download CC-CEDICT",
    )
    parser.add_argument(
        "--cedict-file",
        default=None,
        help="Path to a local cedict_ts.u8 file (skip download)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output CSV file (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    # Resolve paths relative to repository root (parent of scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # jieba dict
    if args.jieba_file:
        jieba_path = args.jieba_file
    else:
        jieba_path = os.path.join(repo_root, "dict.txt.big")
        if not os.path.exists(jieba_path):
            download(args.jieba_url, jieba_path)
        else:
            print(f"Using existing {jieba_path}")

    # CC-CEDICT
    if args.cedict_file:
        cedict_path = args.cedict_file
    else:
        cedict_path = os.path.join(repo_root, "cedict_1_0_ts_utf-8_mdbg.txt.gz")
        if not os.path.exists(cedict_path):
            download(args.cedict_url, cedict_path)
        else:
            print(f"Using existing {cedict_path}")

    output_path = os.path.join(repo_root, args.output)

    # Parse
    entries = parse_jieba_dict(jieba_path)
    if not entries:
        print("Error: no entries parsed from jieba dict file", file=sys.stderr)
        sys.exit(1)

    cedict = parse_cedict(cedict_path)

    # Convert
    convert_to_csv(entries, cedict, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
