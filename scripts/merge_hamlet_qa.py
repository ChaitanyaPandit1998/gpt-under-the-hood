"""
merge_hamlet_qa.py

Merges hamlet_qa.json (existing), hamlet_qa_generated.json, and hamlet_qa_handcrafted.json
into a single deduplicated, validated file: hamlet_qa_expanded.json.

Usage:
    python scripts/merge_hamlet_qa.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

SOURCES = {
    "existing":     DATA_DIR / "hamlet_qa.json",
    "generated":    DATA_DIR / "hamlet_qa_generated.json",
    "handcrafted":  DATA_DIR / "hamlet_qa_handcrafted.json",
}
OUTPUT = DATA_DIR / "hamlet_qa_expanded.json"

MAX_ANSWER_CHARS = 150
MAX_QUESTION_CHARS = 150


def load(path):
    if not path.exists():
        print(f"  WARNING: {path.name} not found — skipping")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [[str(q), str(a)] for q, a in data]


def validate(q, a):
    return (
        0 < len(q) <= MAX_QUESTION_CHARS
        and 10 <= len(a) <= MAX_ANSWER_CHARS
        and not q.isspace()
        and not a.isspace()
    )


def main():
    print("Loading sources...")
    counts = {}
    all_pairs = []

    for name, path in SOURCES.items():
        pairs = load(path)
        counts[name] = len(pairs)
        print(f"  {name}: {len(pairs)} pairs from {path.name}")
        all_pairs.extend(pairs)

    print(f"\nTotal before dedup: {len(all_pairs)}")

    # Deduplicate by canonical (lowercased, stripped) question — preserve order
    seen = set()
    unique = []
    for q, a in all_pairs:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append([q, a])

    print(f"After deduplication: {len(unique)} pairs ({len(all_pairs) - len(unique)} removed)")

    # Validate answer + question length
    valid = [[q, a] for q, a in unique if validate(q, a)]
    invalid_count = len(unique) - len(valid)
    if invalid_count:
        print(f"Removed {invalid_count} pairs with invalid length (answer >{MAX_ANSWER_CHARS} chars or question >{MAX_QUESTION_CHARS} chars)")
    print(f"After validation: {len(valid)} pairs")

    # Shuffle with fixed seed for reproducibility
    random.Random(42).shuffle(valid)

    # Write output
    OUTPUT.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nOutput written to: {OUTPUT.name}")
    print(f"\n--- Summary ---")
    print(f"  existing (hamlet_qa.json):         {counts.get('existing', 0):>4} pairs")
    print(f"  generated (hamlet_qa_generated):   {counts.get('generated', 0):>4} pairs")
    print(f"  handcrafted (hamlet_qa_handcrafted): {counts.get('handcrafted', 0):>4} pairs")
    print(f"  -------------------------------------")
    print(f"  Total unique, valid (hamlet_qa_expanded): {len(valid):>4} pairs")
    print(f"\nReady for fine-tuning in notebook 07.")


if __name__ == "__main__":
    main()
