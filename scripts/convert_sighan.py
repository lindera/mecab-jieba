#!/usr/bin/env python3
"""Convert SIGHAN 2005 bakeoff corpus to lindera/MeCab training corpus format.

Usage:
    python scripts/convert_sighan.py \
        --input .tmp/sighan2005/icwb2-data/training/pku_training.utf8 \
        --output work/train/corpus_pku.txt \
        --jieba-dict work/dict.txt.big

SIGHAN format: one sentence per line, words separated by double spaces.
Output format: surface\\tpos,chartype,*,*,*,*  (one token per line, EOS between sentences)

POS tags are assigned by looking up words in the jieba dictionary.
Words not found get a default POS based on character type heuristics.
"""

import argparse
import os
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Character type classifier (aligned with char.def categories)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# POS assignment heuristics
# ---------------------------------------------------------------------------

# Common Chinese punctuation → POS "w"
PUNCTUATION = set("，。！？、；：""''（）《》【】〈〉「」『』〔〕…—·－ー．,!?;:()[]{}\"'")


def guess_pos(surface: str) -> str:
    """Guess POS tag for a word not found in jieba dictionary."""
    if not surface:
        return "x"
    if all(c in PUNCTUATION for c in surface):
        return "w"
    char_type = get_char_type(surface)
    if char_type == "NUMERIC":
        return "m"
    if char_type == "ALPHA":
        return "x"
    # Default for Chinese characters: noun
    return "n"


def load_jieba_dict(path: str) -> dict[str, str]:
    """Load jieba dict.txt.big and return a mapping from surface to POS."""
    surface_to_pos: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            surface = parts[0]
            pos = parts[2]
            if surface not in surface_to_pos:
                surface_to_pos[surface] = pos
    return surface_to_pos


# ---------------------------------------------------------------------------
# SIGHAN corpus parser
# ---------------------------------------------------------------------------

def parse_sighan(
    path: str,
    jieba_pos: dict[str, str],
) -> tuple[list[list[tuple[str, str, str]]], Counter, Counter]:
    """Parse a SIGHAN bakeoff training file.

    Returns (sentences, pos_counter, lookup_stats) where each sentence is
    a list of (surface, pos, char_type) tuples.
    """
    sentences: list[list[tuple[str, str, str]]] = []
    pos_counter: Counter = Counter()
    lookup_stats: Counter = Counter()  # "found" / "guessed"

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # SIGHAN format: words separated by double spaces
            words = line.split()
            # Filter empty strings from split
            words = [w.strip() for w in words if w.strip()]

            if not words:
                continue

            current: list[tuple[str, str, str]] = []
            for word in words:
                # Look up POS in jieba dictionary
                pos = jieba_pos.get(word)
                if pos is not None:
                    lookup_stats["found"] += 1
                else:
                    pos = guess_pos(word)
                    lookup_stats["guessed"] += 1

                char_type = get_char_type(word)
                pos_counter[pos] += 1
                current.append((word, pos, char_type))

            if current:
                sentences.append(current)

    return sentences, pos_counter, lookup_stats


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def get_char_count_label(surface: str) -> str:
    """Return character count label: '1', '2', '3', or '4+' for 4 or more."""
    n = len(surface)
    if n >= 4:
        return "4+"
    return str(n)


def write_corpus(
    sentences: list[list[tuple[str, str, str]]], output: str
) -> None:
    """Write sentences in lindera training corpus format."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sent in sentences:
            for surface, pos, char_type in sent:
                char_count = get_char_count_label(surface)
                first_char = surface[0] if surface else "*"
                last_char = surface[-1] if surface else "*"
                # F[0]=pos, F[1]=chartype, F[2..5]=*, F[6]=char_count,
                # F[7]=first_char, F[8]=last_char, F[9]=freq_band(*)
                f.write(f"{surface}\t{pos},{char_type},*,*,*,*,{char_count},{first_char},{last_char},*\n")
            f.write("EOS\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SIGHAN 2005 bakeoff corpus to lindera training format"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Input SIGHAN training file(s) (UTF-8)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Output corpus file",
    )
    parser.add_argument(
        "--jieba-dict",
        required=True,
        metavar="FILE",
        help="Path to jieba dict.txt.big for POS assignment",
    )
    args = parser.parse_args()

    # Validate jieba dict
    if not os.path.isfile(args.jieba_dict):
        print(f"Error: jieba dict not found: {args.jieba_dict}", file=sys.stderr)
        sys.exit(1)

    # Load jieba dict for POS assignment
    print(f"Loading jieba dict: {args.jieba_dict} ...")
    jieba_pos = load_jieba_dict(args.jieba_dict)
    print(f"  Loaded {len(jieba_pos)} entries with POS tags")

    # Parse all input files
    all_sentences: list[list[tuple[str, str, str]]] = []
    total_pos_counter: Counter = Counter()
    total_lookup_stats: Counter = Counter()

    for filepath in args.input:
        if not os.path.isfile(filepath):
            print(f"Warning: {filepath} not found, skipping", file=sys.stderr)
            continue

        print(f"Processing {filepath} ...")
        sentences, pos_counter, lookup_stats = parse_sighan(filepath, jieba_pos)
        all_sentences.extend(sentences)
        total_pos_counter.update(pos_counter)
        total_lookup_stats.update(lookup_stats)

    total_sentences = len(all_sentences)
    total_tokens = sum(len(s) for s in all_sentences)

    if total_tokens == 0:
        print("Error: no tokens parsed from input files", file=sys.stderr)
        sys.exit(1)

    # Write output
    write_corpus(all_sentences, args.output)
    print(f"\nWrote {total_sentences} sentences ({total_tokens} tokens) to {args.output}")

    # Print statistics
    found = total_lookup_stats["found"]
    guessed = total_lookup_stats["guessed"]
    total = found + guessed
    print(f"\nPOS lookup statistics:")
    print(f"  Found in jieba dict: {found:>8d} ({found*100/total:.1f}%)")
    print(f"  Guessed (heuristic): {guessed:>8d} ({guessed*100/total:.1f}%)")

    print(f"\nPOS distribution ({len(total_pos_counter)} tags):")
    for tag, count in total_pos_counter.most_common(20):
        pct = count * 100 / total
        print(f"  {tag:8s}  {count:>8d}  ({pct:5.1f}%)")
    if len(total_pos_counter) > 20:
        print(f"  ... and {len(total_pos_counter) - 20} more tags")

    print("Done.")


if __name__ == "__main__":
    main()
