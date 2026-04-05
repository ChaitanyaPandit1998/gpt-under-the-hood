"""
build_pretraining_corpus.py

Assembles the pre-training corpus directly from raw Wikipedia JSON files.
No Claude API required.

Each play contributes:  summary + plot (where available)
Format:  === PLAY: <Title> ===\n<summary>\n\n<plot>

Reads from:  data/shakespeare_pipeline/raw_wikipedia/<slug>.json
Writes to:   data/shakespeare_pretraining_corpus.txt
             data/shakespeare_pipeline/corpus_stats.json

Usage:
    python scripts/build_pretraining_corpus.py
"""

import json
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
PIPELINE_DIR = DATA_DIR / "shakespeare_pipeline"
RAW_DIR      = PIPELINE_DIR / "raw_wikipedia"
OUT_CORPUS   = DATA_DIR / "shakespeare_pretraining_corpus.txt"
STATS_PATH   = PIPELINE_DIR / "corpus_stats.json"

# Tier order for curriculum-style sorting (major plays first)
TIER_ORDER = {"major": 0, "secondary": 1, "minor": 2}


def clean_inline(text: str) -> str:
    """Collapse runs of spaces within a line but leave newlines intact."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_play_block(data: dict) -> str:
    """
    Build the text block for a single play.

    Uses full_text (structured Wikipedia article) if available, preserving
    section headings and paragraph breaks.  Falls back to summary + plot for
    any cached files that pre-date the full_text field.

    Format:
        === PLAY: <Title> ===

        <paragraph>

        <section heading>

        <paragraph>
        ...
    """
    title     = data["title"]
    full_text = clean_inline(data.get("full_text", ""))
    summary   = clean_inline(data.get("summary", ""))
    plot      = clean_inline(data.get("plot", ""))

    parts = [f"=== PLAY: {title} ==="]
    if full_text:
        parts.append(full_text)
    else:
        # Fallback for older cached files without full_text
        if summary:
            parts.append(summary)
        if plot:
            parts.append(plot)

    return "\n\n".join(parts)


def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print("ERROR: No raw Wikipedia files found. Run fetch_wikipedia_shakespeare.py first.")
        return

    print(f"Building corpus from {len(raw_files)} raw Wikipedia files...")

    plays = []
    for f in raw_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        plays.append(data)

    # Sort: major first, then secondary, then minor
    plays.sort(key=lambda d: (TIER_ORDER.get(d.get("tier", "minor"), 2), d["title"]))

    blocks      = []
    stats       = []
    total_words = 0
    total_chars = 0

    for data in plays:
        block = build_play_block(data)
        blocks.append(block)

        words = len(block.split())
        chars = len(block)
        total_words += words
        total_chars += chars

        has_full = bool(data.get("full_text"))
        stats.append({
            "title":    data["title"],
            "tier":     data.get("tier", "?"),
            "genre":    data.get("genre", "?"),
            "has_full": has_full,
            "words":    words,
            "chars":    chars,
        })
        marker = "full" if has_full else "summary+plot"
        print(f"  {data['title']:42s}  {words:6d} words  ({marker})")

    # Triple newline between plays so the tokeniser sees clear document boundaries
    corpus = "\n\n\n".join(blocks)

    OUT_CORPUS.write_text(corpus, encoding="utf-8")

    STATS_PATH.write_text(json.dumps({
        "total_plays":  len(blocks),
        "total_words":  total_words,
        "total_chars":  total_chars,
        "plays":        stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    plays_with_full = sum(1 for s in stats if s["has_full"])
    print(f"\n--- Corpus Stats ---")
    print(f"  Plays:          {len(blocks)} ({plays_with_full} full Wikipedia text, {len(blocks)-plays_with_full} fallback)")
    print(f"  Total words:    {total_words:,}")
    print(f"  Total chars:    {total_chars:,}")
    print(f"  Corpus saved:   {OUT_CORPUS}")
    print(f"  Stats saved:    {STATS_PATH}")
    print(f"\nReady for pre-training in notebook 06.")


if __name__ == "__main__":
    main()
