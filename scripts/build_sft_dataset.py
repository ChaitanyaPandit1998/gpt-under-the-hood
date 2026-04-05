"""
build_sft_dataset.py

Merges all per-play Q&A files into a single SFT dataset.

Steps:
  1. Load all qa/<slug>.json files
  2. Merge + deduplicate (by canonical question)
  3. Validate (answer length, non-empty)
  4. Shuffle with fixed seed
  5. Split into train (90%) / val (10%)
  6. Write output files

Reads from:  data/shakespeare_pipeline/qa/<slug>.json
Writes to:   data/shakespeare_sft_train.json
             data/shakespeare_sft_val.json
             data/shakespeare_pipeline/sft_stats.json

Format of output JSON:  [[question, answer], ...]

Usage:
    python scripts/build_sft_dataset.py

    # Preview stats only:
    python scripts/build_sft_dataset.py --stats-only
"""

import argparse
import json
import random
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
PIPELINE_DIR = DATA_DIR / "shakespeare_pipeline"
QA_DIR       = PIPELINE_DIR / "qa"
OUT_TRAIN    = DATA_DIR / "shakespeare_sft_train.json"
OUT_VAL      = DATA_DIR / "shakespeare_sft_val.json"
STATS_PATH   = PIPELINE_DIR / "sft_stats.json"

MAX_ANSWER_CHARS   = 150
MAX_QUESTION_CHARS = 200
VAL_FRACTION       = 0.10
RANDOM_SEED        = 42


def validate_pair(q: str, a: str) -> bool:
    if not q or not a:
        return False
    if len(q) > MAX_QUESTION_CHARS:
        return False
    if len(a) < 10 or len(a) > MAX_ANSWER_CHARS:
        return False
    if q.isspace() or a.isspace():
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats without writing output files")
    args = parser.parse_args()

    index_path = QA_DIR / "qa_index.json"
    if not index_path.exists():
        print("ERROR: qa_index.json not found. Run generate_shakespeare_qa.py first.")
        return

    index = json.loads(index_path.read_text(encoding="utf-8"))
    plays = [p for p in index if p["status"] in ("ok", "cached") and p["file"]]

    print(f"Loading Q&A from {len(plays)} plays...")

    all_pairs   = []
    per_play    = {}

    for entry in plays:
        path = QA_DIR / entry["file"]
        if not path.exists():
            print(f"  WARNING: Missing qa file for '{entry['title']}' — skipping")
            continue

        data   = json.loads(path.read_text(encoding="utf-8"))
        pairs  = data.get("pairs", [])
        title  = data["title"]

        valid_for_play = 0
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                q, a = str(item[0]).strip(), str(item[1]).strip()
                if validate_pair(q, a):
                    all_pairs.append([q, a])
                    valid_for_play += 1

        per_play[title] = valid_for_play
        print(f"  {title:40s}  {valid_for_play:3d} valid pairs")

    print(f"\nTotal before dedup: {len(all_pairs)}")

    # Deduplicate by canonical question
    seen   = set()
    unique = []
    for q, a in all_pairs:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append([q, a])

    removed = len(all_pairs) - len(unique)
    print(f"After dedup: {len(unique)} pairs ({removed} duplicates removed)")

    # Shuffle
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique)

    # Train/val split
    n_val   = max(1, int(len(unique) * VAL_FRACTION))
    n_train = len(unique) - n_val
    train   = unique[:n_train]
    val     = unique[n_train:]

    print(f"\n--- SFT Dataset Stats ---")
    print(f"  Total pairs: {len(unique)}")
    print(f"  Train:       {len(train)} ({len(train)/len(unique)*100:.1f}%)")
    print(f"  Val:         {len(val)} ({len(val)/len(unique)*100:.1f}%)")
    print(f"\n  Per-play breakdown:")
    for title, n in sorted(per_play.items(), key=lambda x: -x[1]):
        print(f"    {title:40s}  {n:3d}")

    # Save stats
    STATS_PATH.write_text(json.dumps({
        "total_pairs":  len(unique),
        "train_pairs":  len(train),
        "val_pairs":    len(val),
        "per_play":     per_play,
        "val_fraction": VAL_FRACTION,
        "random_seed":  RANDOM_SEED,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Stats saved: {STATS_PATH}")

    if args.stats_only:
        print("\n(--stats-only: output files not written)")
        return

    OUT_TRAIN.write_text(json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_VAL.write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Train saved: {OUT_TRAIN}")
    print(f"  Val saved:   {OUT_VAL}")
    print(f"\nReady for SFT fine-tuning in notebook 07.")


if __name__ == "__main__":
    main()
