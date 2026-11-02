"""Check eval/questions.json: schema, counts and that every gold id exists.

Gold ids are article/annex level ("ai_act:art6", "ai_act:anxIII", "gdpr:art22"),
matched against the chunk ids in data/processed/chunks_en.jsonl.

Usage: python -m eval.validate_questions
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"direct", "multi_article", "unanswerable"}


def unit_id(chunk_id: str) -> str:
    """'ai_act:art5:p1:a' -> 'ai_act:art5'."""
    return ":".join(chunk_id.split(":")[:2])


def main() -> None:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]
    chunks = [json.loads(l) for l in (ROOT / "data/processed/chunks_en.jsonl").open()]
    valid_units = {unit_id(c["id"]) for c in chunks}
    errors = []
    ids = Counter(q["id"] for q in questions)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for q in questions:
        if q["type"] not in TYPES:
            errors.append(f"{q['id']}: unknown type {q['type']}")
        if not q["question"].get("en"):
            errors.append(f"{q['id']}: missing English question")
        if q["type"] == "unanswerable" and q["gold"]:
            errors.append(f"{q['id']}: unanswerable question with gold ids")
        if q["type"] != "unanswerable" and not q["gold"]:
            errors.append(f"{q['id']}: answerable question without gold ids")
        errors += [f"{q['id']}: unknown gold id {g}" for g in q["gold"] if g not in valid_units]
    types = Counter(q["type"] for q in questions)
    translated = {lang: sum(bool(q["question"].get(lang)) for q in questions) for lang in ("es", "de")}
    print(f"{len(questions)} questions | {dict(types)} | translated: {translated}")
    print("\n".join(errors) if errors else "OK")


if __name__ == "__main__":
    main()
