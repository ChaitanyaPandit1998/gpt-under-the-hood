"""
generate_enriched_summaries.py

Uses Claude API to enrich raw Wikipedia data into clean 4-section summaries:
  1. Overview (3-4 sentences, modern English)
  2. Characters (1 sentence each, key cast only)
  3. Plot (flowing prose, 6-10 sentences)
  4. Themes (2-4 sentences)

Reads from:  data/shakespeare_pipeline/raw_wikipedia/<slug>.json
Writes to:   data/shakespeare_pipeline/enriched/<slug>.json
Writes to:   data/shakespeare_pipeline/enriched_index.json

Usage:
    ANTHROPIC_API_KEY=<key> python scripts/generate_enriched_summaries.py

    # Process a single play (for testing):
    ANTHROPIC_API_KEY=<key> python scripts/generate_enriched_summaries.py --play Hamlet

Requires:
    pip install anthropic
"""

import argparse
import json
import time
import re
from pathlib import Path

import anthropic

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data" / "shakespeare_pipeline"
RAW_DIR      = DATA_DIR / "raw_wikipedia"
ENRICHED_DIR = DATA_DIR / "enriched"
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

MODEL        = "claude-haiku-4-5-20251001"   # fast + cheap for enrichment
MAX_TOKENS   = 1200
INTER_DELAY  = 2.0   # seconds between API calls

SYSTEM_PROMPT = """\
You are an expert Shakespeare scholar writing clear modern English summaries \
for an educational language model training dataset. \
Write in fluent, natural prose — no bullet points, no markdown formatting, no headers. \
Every sentence must be factually accurate and complete.\
"""

ENRICHMENT_TEMPLATE = """\
Using the Wikipedia text below about Shakespeare's "{title}", write a structured \
4-section summary. Output ONLY a JSON object with these exact keys:
  "overview"   - 3-4 sentences introducing the play (genre, setting, central conflict).
  "characters" - one sentence per major character describing their role; \
list the 4-6 most important characters, one per line inside this string.
  "plot"       - 6-10 sentences in chronological order covering the full arc: \
setup, rising action, climax, resolution. Modern English only, no archaic phrasing.
  "themes"     - 2-4 sentences covering the main thematic concerns of the play.

Wikipedia text:
---
{wiki_text}
---

Respond with valid JSON only. No markdown, no explanation outside the JSON.
"""


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def build_wiki_text(data: dict) -> str:
    """Combine summary + plot, truncated to ~3000 chars to stay within context."""
    parts = []
    if data.get("summary"):
        parts.append("SUMMARY:\n" + data["summary"])
    if data.get("plot"):
        parts.append("PLOT:\n" + data["plot"])
    text = "\n\n".join(parts)
    return text[:3000]


def enrich_play(client: anthropic.Anthropic, data: dict) -> dict | None:
    """Call Claude to produce the 4-section enrichment. Returns parsed dict or None."""
    wiki_text = build_wiki_text(data)
    if len(wiki_text) < 100:
        print(f"  WARNING: Very short Wikipedia text for '{data['title']}' ({len(wiki_text)} chars)")

    prompt = ENRICHMENT_TEMPLATE.format(title=data["title"], wiki_text=wiki_text)

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        enriched = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ERROR: JSON parse failed for '{data['title']}': {e}")
        print(f"  Raw response (first 300 chars): {raw[:300]}")
        return None

    for key in ("overview", "characters", "plot", "themes"):
        if key not in enriched:
            print(f"  WARNING: Missing key '{key}' in enrichment for '{data['title']}'")
            enriched[key] = ""

    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", help="Process only this play title (exact match)")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # Load fetch index to get all play files
    index_path = DATA_DIR / "fetch_index.json"
    if not index_path.exists():
        print("ERROR: fetch_index.json not found. Run fetch_wikipedia_shakespeare.py first.")
        return

    index = json.loads(index_path.read_text(encoding="utf-8"))
    plays = [p for p in index if p["status"] in ("ok", "cached") and p["file"]]

    if args.play:
        plays = [p for p in plays if p["title"] == args.play]
        if not plays:
            print(f"ERROR: Play '{args.play}' not found in fetch index.")
            return

    enriched_index = []
    success = 0
    failed  = []

    for i, entry in enumerate(plays):
        title    = entry["title"]
        out_path = ENRICHED_DIR / entry["file"]

        if out_path.exists():
            print(f"[{i+1:02d}/{len(plays)}] SKIP  {title} (already enriched)")
            enriched_index.append({"title": title, "file": entry["file"], "status": "cached"})
            continue

        raw_path = RAW_DIR / entry["file"]
        if not raw_path.exists():
            print(f"[{i+1:02d}/{len(plays)}] SKIP  {title} (raw file missing)")
            failed.append(title)
            continue

        raw_data = json.loads(raw_path.read_text(encoding="utf-8"))

        print(f"[{i+1:02d}/{len(plays)}] ENRICH {title} ...", end=" ", flush=True)
        enriched = enrich_play(client, raw_data)

        if enriched is None:
            print("FAILED")
            failed.append(title)
            enriched_index.append({"title": title, "file": None, "status": "failed"})
            time.sleep(INTER_DELAY)
            continue

        result = {
            "title":  title,
            "tier":   raw_data.get("tier", "secondary"),
            "genre":  raw_data.get("genre", "unknown"),
            **enriched,
        }
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        overview_words = len(enriched["overview"].split())
        plot_words     = len(enriched["plot"].split())
        print(f"OK  overview={overview_words}w  plot={plot_words}w")
        success += 1
        enriched_index.append({"title": title, "file": entry["file"], "status": "ok"})
        time.sleep(INTER_DELAY)

    # Save enriched index
    out_idx = ENRICHED_DIR / "enriched_index.json"
    out_idx.write_text(json.dumps(enriched_index, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- Summary ---")
    print(f"  Enriched: {success}/{len(plays)} plays")
    print(f"  Index:    {out_idx}")
    if failed:
        print(f"  FAILED ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
