"""
fetch_wikipedia_shakespeare.py

Fetches Wikipedia summaries and plot sections for all 37 Shakespeare plays.
Saves raw text to data/shakespeare_pipeline/raw_wikipedia/

Usage:
    python scripts/fetch_wikipedia_shakespeare.py

Output:
    data/shakespeare_pipeline/raw_wikipedia/<slug>.json  (one file per play)
    data/shakespeare_pipeline/fetch_index.json           (manifest)

Requires:
    pip install wikipedia-api
"""

import json
import time
import re
from pathlib import Path

import wikipediaapi

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "shakespeare_pipeline"
RAW_DIR  = DATA_DIR / "raw_wikipedia"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── All 37 canonical Shakespeare plays ───────────────────────────────────────
# tier: "major" → ~65 Q&A pairs, "secondary" → ~35, "minor" → ~25
PLAYS = [
    # Tragedies — major (most studied)
    {"title": "Hamlet",                      "wiki": "Hamlet",                           "tier": "major",     "genre": "tragedy"},
    {"title": "Macbeth",                     "wiki": "Macbeth",                          "tier": "major",     "genre": "tragedy"},
    {"title": "Othello",                     "wiki": "Othello",                          "tier": "major",     "genre": "tragedy"},
    {"title": "King Lear",                   "wiki": "King_Lear",                        "tier": "major",     "genre": "tragedy"},
    {"title": "Romeo and Juliet",            "wiki": "Romeo_and_Juliet",                 "tier": "major",     "genre": "tragedy"},

    # Tragedies — secondary
    {"title": "Julius Caesar",               "wiki": "Julius_Caesar_(play)",             "tier": "secondary", "genre": "tragedy"},
    {"title": "Antony and Cleopatra",        "wiki": "Antony_and_Cleopatra",             "tier": "secondary", "genre": "tragedy"},
    {"title": "Coriolanus",                  "wiki": "Coriolanus_(play)",                "tier": "secondary", "genre": "tragedy"},
    {"title": "Timon of Athens",             "wiki": "Timon_of_Athens",                  "tier": "minor",     "genre": "tragedy"},
    {"title": "Titus Andronicus",            "wiki": "Titus_Andronicus",                 "tier": "minor",     "genre": "tragedy"},

    # Comedies — major
    {"title": "A Midsummer Night's Dream",   "wiki": "A Midsummer Night's Dream",        "tier": "major",     "genre": "comedy"},
    {"title": "Much Ado About Nothing",      "wiki": "Much_Ado_About_Nothing",           "tier": "major",     "genre": "comedy"},
    {"title": "As You Like It",              "wiki": "As_You_Like_It",                   "tier": "major",     "genre": "comedy"},
    {"title": "The Merchant of Venice",      "wiki": "The_Merchant_of_Venice",           "tier": "major",     "genre": "comedy"},
    {"title": "Twelfth Night",               "wiki": "Twelfth_Night",                    "tier": "major",     "genre": "comedy"},
    {"title": "The Tempest",                 "wiki": "The_Tempest",                      "tier": "major",     "genre": "comedy"},

    # Comedies — secondary
    {"title": "The Taming of the Shrew",     "wiki": "The_Taming_of_the_Shrew",          "tier": "secondary", "genre": "comedy"},
    {"title": "Measure for Measure",         "wiki": "Measure_for_Measure",              "tier": "secondary", "genre": "comedy"},
    {"title": "The Winter's Tale",           "wiki": "The Winter's Tale",                "tier": "secondary", "genre": "comedy"},
    {"title": "The Merry Wives of Windsor",  "wiki": "The_Merry_Wives_of_Windsor",       "tier": "minor",     "genre": "comedy"},
    {"title": "The Comedy of Errors",        "wiki": "The_Comedy_of_Errors",             "tier": "minor",     "genre": "comedy"},
    {"title": "Love's Labour's Lost",        "wiki": "Love's Labour's Lost",             "tier": "minor",     "genre": "comedy"},
    {"title": "The Two Gentlemen of Verona", "wiki": "The_Two_Gentlemen_of_Verona",      "tier": "minor",     "genre": "comedy"},

    # Histories — major
    {"title": "Henry V",                     "wiki": "Henry_V_(play)",                   "tier": "major",     "genre": "history"},
    {"title": "Richard III",                 "wiki": "Richard_III_(play)",               "tier": "major",     "genre": "history"},
    {"title": "Richard II",                  "wiki": "Richard_II_(play)",                "tier": "secondary", "genre": "history"},

    # Histories — secondary
    {"title": "Henry IV, Part 1",            "wiki": "Henry_IV,_Part_1",                 "tier": "secondary", "genre": "history"},
    {"title": "Henry IV, Part 2",            "wiki": "Henry_IV,_Part_2",                 "tier": "secondary", "genre": "history"},
    {"title": "Henry VI, Part 1",            "wiki": "Henry_VI,_Part_1",                 "tier": "minor",     "genre": "history"},
    {"title": "Henry VI, Part 2",            "wiki": "Henry_VI,_Part_2",                 "tier": "minor",     "genre": "history"},
    {"title": "Henry VI, Part 3",            "wiki": "Henry_VI,_Part_3",                 "tier": "minor",     "genre": "history"},
    {"title": "Henry VIII",                  "wiki": "Henry_VIII_(play)",                "tier": "minor",     "genre": "history"},
    {"title": "King John",                   "wiki": "King_John_(play)",                 "tier": "minor",     "genre": "history"},

    # Problem plays / Romances
    {"title": "All's Well That Ends Well",   "wiki": "All's Well That Ends Well",        "tier": "secondary", "genre": "comedy"},
    {"title": "Troilus and Cressida",        "wiki": "Troilus_and_Cressida",             "tier": "secondary", "genre": "tragedy"},
    {"title": "Cymbeline",                   "wiki": "Cymbeline",                        "tier": "minor",     "genre": "comedy"},
    {"title": "Pericles",                    "wiki": "Pericles,_Prince_of_Tyre",         "tier": "minor",     "genre": "comedy"},
]

# Sections to extract (first match wins; fallback to summary)
PLOT_SECTION_NAMES = ["Plot", "Synopsis", "Plot synopsis", "Summary", "Plot summary"]


def slug(title: str) -> str:
    """Convert play title to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def extract_plot_section(page) -> str:
    """Return the first matching plot section, or '' if none found."""
    for section_name in PLOT_SECTION_NAMES:
        for section in page.sections:
            if section.title.lower() == section_name.lower():
                return section.text.strip()
            # Check subsections one level deep
            for sub in section.sections:
                if sub.title.lower() == section_name.lower():
                    return sub.text.strip()
    return ""


def clean_text(text: str) -> str:
    """Remove Wikipedia markers and collapse all whitespace to single spaces."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note \d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_structured(text: str) -> str:
    """Remove Wikipedia markers while preserving paragraph and section breaks."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note \d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text)
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple spaces on a single line (but leave newlines intact)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to 2 (one blank line max)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_play(wiki: wikipediaapi.Wikipedia, play: dict) -> dict | None:
    """Fetch a single play. Returns structured dict or None on failure."""
    page = wiki.page(play["wiki"])
    if not page.exists():
        print(f"  WARNING: Page not found for '{play['title']}' (wiki='{play['wiki']}')")
        return None

    summary   = clean_text(page.summary)
    plot      = clean_text(extract_plot_section(page))
    full_text = clean_text_structured(page.text)

    result = {
        "title":      play["title"],
        "tier":       play["tier"],
        "genre":      play["genre"],
        "wiki_title": page.title,
        "summary":    summary,
        "plot":       plot,
        "full_text":  full_text,
        "summary_chars":   len(summary),
        "plot_chars":      len(plot),
        "full_text_chars": len(full_text),
    }
    return result


def main():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="ShakespeareDataPipeline/1.0 (educational; gpt-under-the-hood)",
    )

    index = []
    success = 0
    failed  = []

    for i, play in enumerate(PLAYS):
        title = play["title"]
        out_path = RAW_DIR / f"{slug(title)}.json"

        if out_path.exists():
            print(f"[{i+1:02d}/{len(PLAYS)}] SKIP  {title} (already fetched)")
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            index.append({"title": title, "file": out_path.name, "status": "cached",
                           "summary_chars": data.get("summary_chars", 0),
                           "plot_chars": data.get("plot_chars", 0)})
            continue

        print(f"[{i+1:02d}/{len(PLAYS)}] FETCH {title} ...", end=" ", flush=True)
        data = fetch_play(wiki, play)

        if data is None:
            print("FAILED")
            failed.append(title)
            index.append({"title": title, "file": None, "status": "failed"})
            time.sleep(1.0)
            continue

        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK  full={data['full_text_chars']}ch  summary={data['summary_chars']}ch  plot={data['plot_chars']}ch")
        success += 1
        index.append({"title": title, "file": out_path.name, "status": "ok",
                      "summary_chars": data["summary_chars"],
                      "plot_chars": data["plot_chars"],
                      "full_text_chars": data["full_text_chars"]})
        time.sleep(1.5)  # Polite rate limit

    # Save manifest
    manifest_path = DATA_DIR / "fetch_index.json"
    manifest_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- Summary ---")
    print(f"  Fetched:  {success}/{len(PLAYS)} plays")
    print(f"  Manifest: {manifest_path}")
    if failed:
        print(f"  FAILED ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
