#!/usr/bin/env python3
"""Convert UD Chinese CoNLL-U files to lindera/MeCab training corpus format.

Usage:
    python scripts/convert_conllu.py \\
        --input dir_or_file [dir_or_file ...] \\
        --output corpus.txt \\
        --jieba-dict /path/to/dict.txt.big \\
        --split train

The script reads CoNLL-U formatted files from Universal Dependencies (UD)
Chinese treebanks and converts them into lindera training corpus format:

    surface\\tpos,*,*,*,*
    surface\\tpos,*,*,*,*
    EOS

UPOS tags are mapped to PKU tagset.  For PROPN, the jieba dict.txt.big is
consulted to distinguish nr (person name), ns (place name), nt (organization),
and nz (other proper noun).
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# UPOS → PKU tagset mapping
# ---------------------------------------------------------------------------

UPOS_TO_PKU: dict[str, str] = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "d",
    "PRON": "r",
    "NUM": "m",
    "CLASSIFIER": "q",
    "PART": "u",
    "ADP": "p",
    "CCONJ": "c",
    "SCONJ": "c",
    "PUNCT": "w",
    "AUX": "v",
    "DET": "r",
    "X": "x",
    "SYM": "w",
    "INTJ": "e",
    # PROPN is handled separately via jieba dict lookup.
}

# jieba POS tags that map to specific PROPN sub-types
PROPN_POS_MAP: dict[str, str] = {
    "nr": "nr",
    "ns": "ns",
    "nt": "nt",
    "nz": "nz",
}

DEFAULT_PROPN_TAG = "nz"


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
# jieba dict loader
# ---------------------------------------------------------------------------


def load_jieba_dict(path: str) -> dict[str, str]:
    """Load jieba dict.txt.big and return a mapping from surface to POS.

    The dict format is: ``surface freq pos`` (space-separated).
    Only entries with a POS field are stored.  When duplicate surfaces appear,
    the first occurrence wins.
    """
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


def resolve_propn(surface: str, jieba_pos: dict[str, str]) -> str:
    """Resolve a PROPN surface to a PKU sub-tag using jieba dict.

    Returns one of: nr, ns, nt, nz.
    """
    pos = jieba_pos.get(surface)
    if pos is None:
        return DEFAULT_PROPN_TAG
    return PROPN_POS_MAP.get(pos, DEFAULT_PROPN_TAG)


# ---------------------------------------------------------------------------
# CoNLL-U parser
# ---------------------------------------------------------------------------


def is_token_id(token_id: str) -> bool:
    """Return True if *token_id* is a simple integer (not MWT or empty node)."""
    # Multi-word tokens have IDs like "1-2", empty nodes like "0.1".
    return token_id.isdigit()


def parse_conllu(
    path: str,
    jieba_pos: dict[str, str],
    upos_counter: Counter,
    propn_stats: Counter,
) -> list[list[tuple[str, str, str]]]:
    """Parse a CoNLL-U file and return sentences as lists of (surface, pku_pos, char_type).

    Updates *upos_counter* and *propn_stats* in place.
    """
    sentences: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Comment line
            if line.startswith("#"):
                continue

            # Blank line → sentence boundary
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                # Malformed line; skip silently.
                continue

            token_id = fields[0]
            if not is_token_id(token_id):
                # Skip multi-word tokens (e.g. "1-2") and empty nodes (e.g. "0.1")
                continue

            surface = fields[1]
            upos = fields[3]

            upos_counter[upos] += 1

            # Map UPOS → PKU
            if upos == "PROPN":
                pku_tag = resolve_propn(surface, jieba_pos)
                propn_stats[pku_tag] += 1
            else:
                pku_tag = UPOS_TO_PKU.get(upos, "x")

            char_type = get_char_type(surface)
            current.append((surface, pku_tag, char_type))

    # Flush last sentence if file doesn't end with blank line
    if current:
        sentences.append(current)

    return sentences


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def collect_conllu_files(
    inputs: list[str], split: str | None
) -> list[str]:
    """Collect CoNLL-U file paths from the given inputs.

    Each input may be a file or a directory.  If *split* is given (e.g.
    ``train``), only files whose name contains ``-{split}.conllu`` are
    included.
    """
    files: list[str] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if split and f"-{split}.conllu" not in p.name:
                continue
            files.append(str(p))
        elif p.is_dir():
            for child in sorted(p.glob("*.conllu")):
                if split and f"-{split}.conllu" not in child.name:
                    continue
                files.append(str(child))
        else:
            print(f"Warning: {inp} is not a file or directory, skipping", file=sys.stderr)
    return files


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_corpus(
    sentences: list[list[tuple[str, str, str]]], output: str
) -> None:
    """Write sentences in lindera training corpus format."""
    with open(output, "w", encoding="utf-8") as f:
        for sent in sentences:
            for surface, pos, char_type in sent:
                # pos,chartype,*,*,*,*  (6 feature columns, matching seed.csv)
                f.write(f"{surface}\t{pos},{char_type},*,*,*,*\n")
            f.write("EOS\n")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_statistics(
    total_sentences: int,
    total_tokens: int,
    upos_counter: Counter,
    propn_stats: Counter,
) -> None:
    """Print conversion statistics to stdout."""
    print(f"\n{'='*60}")
    print("Conversion Statistics")
    print(f"{'='*60}")
    print(f"  Total sentences : {total_sentences}")
    print(f"  Total tokens    : {total_tokens}")

    print(f"\n  UPOS distribution:")
    for tag, count in upos_counter.most_common():
        pct = count * 100 / total_tokens if total_tokens else 0
        pku = UPOS_TO_PKU.get(tag, "nz/nr/ns/nt" if tag == "PROPN" else "x")
        print(f"    {tag:12s} → {pku:6s}  {count:>8d}  ({pct:5.1f}%)")

    if propn_stats:
        propn_total = sum(propn_stats.values())
        print(f"\n  PROPN resolution ({propn_total} total):")
        for tag, count in propn_stats.most_common():
            pct = count * 100 / propn_total if propn_total else 0
            print(f"    {tag:6s}  {count:>8d}  ({pct:5.1f}%)")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UD Chinese CoNLL-U files to lindera/MeCab training corpus format"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Input CoNLL-U file(s) or directory(ies)",
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
        help="Path to jieba dict.txt.big for PROPN resolution",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "dev", "test"],
        help="Only process files matching this split name (e.g. train → *-train.conllu)",
    )
    args = parser.parse_args()

    # Validate jieba dict
    if not os.path.isfile(args.jieba_dict):
        print(f"Error: jieba dict not found: {args.jieba_dict}", file=sys.stderr)
        sys.exit(1)

    # Collect input files
    conllu_files = collect_conllu_files(args.input, args.split)
    if not conllu_files:
        print("Error: no CoNLL-U files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(conllu_files)} CoNLL-U file(s):")
    for f in conllu_files:
        print(f"  {f}")

    # Load jieba dict for PROPN resolution
    print(f"\nLoading jieba dict: {args.jieba_dict} ...")
    jieba_pos = load_jieba_dict(args.jieba_dict)
    print(f"  Loaded {len(jieba_pos)} entries with POS tags")

    # Parse all files
    upos_counter: Counter = Counter()
    propn_stats: Counter = Counter()
    all_sentences: list[list[tuple[str, str, str]]] = []

    for filepath in conllu_files:
        print(f"Processing {filepath} ...")
        sentences = parse_conllu(filepath, jieba_pos, upos_counter, propn_stats)
        all_sentences.extend(sentences)

    total_sentences = len(all_sentences)
    total_tokens = sum(len(s) for s in all_sentences)

    if total_tokens == 0:
        print("Error: no tokens parsed from input files", file=sys.stderr)
        sys.exit(1)

    # Write output
    write_corpus(all_sentences, args.output)
    print(f"\nWrote {total_sentences} sentences ({total_tokens} tokens) to {args.output}")

    # Print statistics
    print_statistics(total_sentences, total_tokens, upos_counter, propn_stats)
    print("Done.")


if __name__ == "__main__":
    main()
