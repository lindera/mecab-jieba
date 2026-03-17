#!/usr/bin/env python3
"""Evaluate segmentation and POS tagging accuracy against UD Chinese test data.

Usage:
    python scripts/evaluate.py \\
        --test-file .tmp/ud-chinese/UD_Chinese-GSD/zh_gsd-ud-test.conllu \\
        --dict-dir work/dict \\
        --jieba-dict work/dict.txt.big

The script parses the UD Chinese CoNLL-U test file to obtain gold-standard
segmentation and POS tags, runs lindera tokenize on each sentence's raw text,
and computes:

    - Segmentation F1 (span-based, micro-averaged)
    - POS accuracy on correctly segmented words
"""

import argparse
import subprocess
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# UPOS -> PKU tagset mapping (same as convert_conllu.py)
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
# CoNLL-U parser (with text extraction)
# ---------------------------------------------------------------------------


def is_token_id(token_id: str) -> bool:
    """Return True if *token_id* is a simple integer (not MWT or empty node)."""
    return token_id.isdigit()


def parse_conllu_for_eval(
    path: str,
    jieba_pos: dict[str, str],
) -> list[dict]:
    """Parse a CoNLL-U file and return sentences with raw text and gold tokens.

    Each returned dict has:
        - "text": raw text from ``# text = ...`` comment
        - "tokens": list of (surface, pku_pos) tuples
    """
    sentences: list[dict] = []
    current_tokens: list[tuple[str, str]] = []
    current_text: str | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Comment line — extract text if present
            if line.startswith("#"):
                if line.startswith("# text = "):
                    current_text = line[len("# text = "):]
                continue

            # Blank line -> sentence boundary
            if not line.strip():
                if current_tokens:
                    sentences.append({
                        "text": current_text,
                        "tokens": current_tokens,
                    })
                    current_tokens = []
                    current_text = None
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                continue

            token_id = fields[0]
            if not is_token_id(token_id):
                continue

            surface = fields[1]
            upos = fields[3]

            # Map UPOS -> PKU
            if upos == "PROPN":
                pku_tag = resolve_propn(surface, jieba_pos)
            else:
                pku_tag = UPOS_TO_PKU.get(upos, "x")

            current_tokens.append((surface, pku_tag))

    # Flush last sentence if file doesn't end with blank line
    if current_tokens:
        sentences.append({
            "text": current_text,
            "tokens": current_tokens,
        })

    return sentences


# ---------------------------------------------------------------------------
# Span computation
# ---------------------------------------------------------------------------


def tokens_to_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Convert a list of token surfaces to character-level (start, end) spans."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for token in tokens:
        end = offset + len(token)
        spans.append((offset, end))
        offset = end
    return spans


# ---------------------------------------------------------------------------
# lindera tokenize
# ---------------------------------------------------------------------------


def run_lindera(text: str, dict_dir: str) -> list[tuple[str, str]] | None:
    """Run lindera tokenize on *text* and return list of (surface, pos).

    Returns None if the subprocess fails.
    """
    try:
        result = subprocess.run(
            ["lindera", "tokenize", "-d", dict_dir],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error running lindera: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"lindera error: {result.stderr.strip()}", file=sys.stderr)
        return None

    tokens: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line == "EOS" or not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        surface = parts[0]
        features = parts[1]
        pos = features.split(",", 1)[0]
        tokens.append((surface, pos))

    return tokens


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    sentences: list[dict],
    dict_dir: str,
    verbose: bool = False,
) -> None:
    """Evaluate segmentation F1 and POS accuracy."""
    total_tp = 0
    total_gold = 0
    total_sys = 0
    pos_match = 0
    pos_total = 0

    evaluated = 0
    skipped = 0
    seg_errors: list[dict] = []

    for sent in sentences:
        text = sent["text"]
        gold_tokens = sent["tokens"]

        if text is None:
            skipped += 1
            continue

        # Verify that concatenation of gold surfaces matches raw text
        gold_concat = "".join(s for s, _ in gold_tokens)
        if gold_concat != text:
            if verbose:
                print(
                    f"Warning: text mismatch, skipping sentence: "
                    f"{text!r} vs {gold_concat!r}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        # Run lindera
        sys_tokens = run_lindera(text, dict_dir)
        if sys_tokens is None:
            skipped += 1
            continue

        # Verify system concatenation matches text
        sys_concat = "".join(s for s, _ in sys_tokens)
        if sys_concat != text:
            if verbose:
                print(
                    f"Warning: system output mismatch, skipping: "
                    f"{text!r} vs {sys_concat!r}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        evaluated += 1

        # Compute spans
        gold_surfaces = [s for s, _ in gold_tokens]
        sys_surfaces = [s for s, _ in sys_tokens]
        gold_spans = set(tokens_to_spans(gold_surfaces))
        sys_spans = set(tokens_to_spans(sys_surfaces))

        tp_spans = gold_spans & sys_spans
        total_tp += len(tp_spans)
        total_gold += len(gold_spans)
        total_sys += len(sys_spans)

        # POS accuracy on matching spans
        gold_span_to_pos = {
            span: pos
            for span, (_, pos) in zip(
                tokens_to_spans(gold_surfaces), gold_tokens
            )
        }
        sys_span_to_pos = {
            span: pos
            for span, (_, pos) in zip(
                tokens_to_spans(sys_surfaces), sys_tokens
            )
        }
        for span in tp_spans:
            pos_total += 1
            if gold_span_to_pos[span] == sys_span_to_pos[span]:
                pos_match += 1

        # Collect segmentation error examples
        if gold_spans != sys_spans and len(seg_errors) < 10:
            gold_only = gold_spans - sys_spans
            sys_only = sys_spans - gold_spans
            seg_errors.append({
                "text": text,
                "gold_only": sorted(gold_only),
                "sys_only": sorted(sys_only),
            })

    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total sentences  : {len(sentences)}")
    print(f"  Evaluated        : {evaluated}")
    print(f"  Skipped          : {skipped}")

    # Segmentation F1 (micro)
    print(f"\n  Segmentation (micro-averaged, span-based):")
    if total_sys > 0:
        precision = total_tp / total_sys
    else:
        precision = 0.0
    if total_gold > 0:
        recall = total_tp / total_gold
    else:
        recall = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f"    Gold spans     : {total_gold}")
    print(f"    System spans   : {total_sys}")
    print(f"    True positives : {total_tp}")
    print(f"    Precision      : {precision:.4f}")
    print(f"    Recall         : {recall:.4f}")
    print(f"    F1             : {f1:.4f}")

    # POS accuracy
    print(f"\n  POS accuracy (on correctly segmented words):")
    if pos_total > 0:
        pos_acc = pos_match / pos_total
    else:
        pos_acc = 0.0
    print(f"    Matching spans : {pos_total}")
    print(f"    POS correct    : {pos_match}")
    print(f"    Accuracy       : {pos_acc:.4f}")

    # Segmentation error examples
    if seg_errors:
        print(f"\n  Segmentation error examples (first {len(seg_errors)}):")
        for i, err in enumerate(seg_errors, 1):
            text = err["text"]
            display_text = text if len(text) <= 60 else text[:57] + "..."
            print(f"\n    [{i}] {display_text}")
            for span in err["gold_only"][:5]:
                word = text[span[0]:span[1]]
                print(f"         gold only: ({span[0]},{span[1]}) {word!r}")
            for span in err["sys_only"][:5]:
                word = text[span[0]:span[1]]
                print(f"         sys  only: ({span[0]},{span[1]}) {word!r}")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation and POS tagging accuracy against UD Chinese test data"
    )
    parser.add_argument(
        "--test-file",
        required=True,
        metavar="FILE",
        help="UD Chinese test CoNLL-U file",
    )
    parser.add_argument(
        "--dict-dir",
        default="work/dict",
        metavar="DIR",
        help="lindera dictionary directory (default: work/dict)",
    )
    parser.add_argument(
        "--jieba-dict",
        required=True,
        metavar="FILE",
        help="Path to jieba dict.txt.big for PROPN resolution in gold tags",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sentence details for errors",
    )
    args = parser.parse_args()

    # Load jieba dict for PROPN resolution
    print(f"Loading jieba dict: {args.jieba_dict} ...")
    jieba_pos = load_jieba_dict(args.jieba_dict)
    print(f"  Loaded {len(jieba_pos)} entries with POS tags")

    # Parse test file
    print(f"Parsing test file: {args.test_file} ...")
    sentences = parse_conllu_for_eval(args.test_file, jieba_pos)
    print(f"  Parsed {len(sentences)} sentences")

    if not sentences:
        print("Error: no sentences parsed from test file", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    print(f"\nRunning lindera tokenize with dict: {args.dict_dir} ...")
    evaluate(sentences, args.dict_dir, verbose=args.verbose)
    print("Done.")


if __name__ == "__main__":
    main()
