import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "shakespeare_qa.json"
CLEAN_PATH = ROOT / "data" / "shakespeare_qa_cleaned.json"
TRAIN_PATH = ROOT / "data" / "shakespeare_qa_train.json"
VAL_PATH = ROOT / "data" / "shakespeare_qa_val.json"
REPORT_PATH = ROOT / "data" / "shakespeare_qa_cleaning_report.json"

RANDOM_SEED = 42
MAX_WHO_SAID = 40
VAL_FRACTION = 0.15


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_question(text: str) -> str:
    return normalize_space(text).lower()


def canonical_answer(text: str) -> str:
    return normalize_space(text)


def is_malformed_pair(question: str, answer: str) -> tuple[bool, str | None]:
    q = normalize_space(question)
    a = normalize_space(answer)

    if not q or not a:
        return True, "empty"

    if "Of the whole body." in a:
        return True, "malformed_entity"

    if a.endswith(":"):
        return True, "unfinished_answer"

    if "plays an important role in Shakespeare's drama." in a:
        return True, "generic_role_answer"

    if "is a character in Shakespeare's works who appears in the plays." in a:
        return True, "generic_character_answer"

    if "is one of Shakespeare's characters, featured in the dramatic works." in a:
        return True, "generic_character_blurb"

    if q.lower().startswith("who said:") and not a.startswith("This was said by "):
        return True, "bad_quote_answer_format"

    if q.lower().startswith("who said:"):
        speaker = a.removeprefix("This was said by ").rstrip(".")
        if not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", speaker):
            return True, "suspicious_speaker_name"
        words = speaker.split()
        if len(words) > 4:
            return True, "overlong_speaker_name"
        if speaker in {"A pretty tale", "All"}:
            return True, "non_speaker_quote_text"

    return False, None


def question_bucket(question: str) -> str:
    q = canonical_question(question)
    if q.startswith("who said:"):
        return "who_said"
    if q.startswith("who is ") or q.startswith("what role does ") or q.startswith("tell me about "):
        return "character"
    if q.startswith("who wrote") or q.startswith("how many"):
        return "factoid"
    if q.startswith("what ") or q.startswith("why ") or q.startswith("explain "):
        return "explanatory"
    return "other"


def main() -> None:
    random.seed(RANDOM_SEED)

    raw_data = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    report: dict[str, object] = {
        "raw_count": len(raw_data),
        "removed": {},
    }

    deduped = []
    seen_pairs = set()
    exact_dupes = 0
    for question, answer in raw_data:
        key = (canonical_question(question), canonical_answer(answer))
        if key in seen_pairs:
            exact_dupes += 1
            continue
        seen_pairs.add(key)
        deduped.append((normalize_space(question), normalize_space(answer)))
    report["removed"]["exact_duplicate_pairs"] = exact_dupes

    by_question = {}
    question_dupes = 0
    for question, answer in deduped:
        key = canonical_question(question)
        if key in by_question:
            question_dupes += 1
            continue
        by_question[key] = (question, answer)
    unique_questions = list(by_question.values())
    report["removed"]["duplicate_questions"] = question_dupes

    cleaned = []
    filtered_counts: dict[str, int] = {}
    for question, answer in unique_questions:
        bad, reason = is_malformed_pair(question, answer)
        if bad:
            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
            continue
        cleaned.append((question, answer))
    report["removed"]["filtered_rows"] = filtered_counts

    who_said = [pair for pair in cleaned if question_bucket(pair[0]) == "who_said"]
    non_who_said = [pair for pair in cleaned if question_bucket(pair[0]) != "who_said"]
    random.shuffle(who_said)
    balanced = non_who_said + who_said[:MAX_WHO_SAID]
    report["removed"]["who_said_trimmed"] = max(0, len(who_said) - MAX_WHO_SAID)

    random.shuffle(balanced)

    val_size = max(1, int(len(balanced) * VAL_FRACTION))
    val_data = balanced[:val_size]
    train_data = balanced[val_size:]

    report["clean_count"] = len(balanced)
    report["train_count"] = len(train_data)
    report["val_count"] = len(val_data)

    bucket_counts: dict[str, int] = {}
    for question, _ in balanced:
        bucket = question_bucket(question)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    report["bucket_counts"] = bucket_counts

    CLEAN_PATH.write_text(json.dumps(balanced, indent=2), encoding="utf-8")
    TRAIN_PATH.write_text(json.dumps(train_data, indent=2), encoding="utf-8")
    VAL_PATH.write_text(json.dumps(val_data, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
