"""Evaluate the three systems on eval/questions.json, in English, Spanish and German.

Stages (run all of them, or one at a time):

  retrieval   Recall@5 of the sources each RAG system gives the model.
              Needs only the embedding model and the reranker (CPU is fine).
  generation  Asks every question to every system and stores the answers in
              results/answers.jsonl (resumable). Needs the LLM (vLLM on Colab).
  judge       LLM-as-a-judge faithfulness of every RAG answer to its sources,
              stored in results/judgements.jsonl (resumable).
  report      Aggregates everything into results/*.csv, results/summary.md
              and the charts.

Metrics
  recall@5            share of gold articles/annexes among the 5 sources
                      (answerable questions; 1/0 when gold_any)
  citation_precision  share of cited articles that are gold (answered, answerable)
  correct_citation    share of answerable questions whose answer cites a gold
                      article (abstaining counts as wrong)
  faithfulness        share of claims supported by the sources (RAG systems)
  abstention          share of unanswerable questions answered with "I don't know"
  false_abstention    share of answerable questions answered with "I don't know"

Usage:
  python -m eval.run_eval --stage retrieval
  python -m eval.run_eval --stage all          # on Colab, with vLLM running
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from rag.context import expand
from rag.generate import LLMClient
from rag.pipeline import Assistant

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUESTIONS = ROOT / "eval" / "questions.json"
ANSWERS = RESULTS / "answers.jsonl"
JUDGEMENTS = RESULTS / "judgements.jsonl"

SYSTEMS = ["llm_only", "dense", "hybrid_rerank"]
RETRIEVAL_SYSTEMS = ["dense", "bm25", "hybrid", "hybrid_rerank"]  # bm25/hybrid: ablation
LANGS = ["en", "es", "de"]
K = 5

SYSTEM_LABELS = {"llm_only": "LLM only", "dense": "RAG (vector)",
                 "bm25": "BM25 only", "hybrid": "Hybrid (no reranker)",
                 "hybrid_rerank": "RAG (hybrid + reranker)"}

JUDGE_PROMPT = """You check whether an answer is supported by source texts.

Split the ANSWER into its individual factual claims. For each claim decide whether \
it is supported by the SOURCES (paraphrases count as supported; claims that add \
information not in the sources are unsupported). Ignore citation markers like [1].

Reply with JSON only: {{"claims": <number of claims>, "supported": <number supported>}}

SOURCES:
{sources}

ANSWER:
{answer}"""


# --------------------------------------------------------------------------- #
def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text())["questions"]


def runs(questions: list[dict], langs: list[str]):
    """(question, lang, text) for every available translation."""
    for q in questions:
        for lang in langs:
            text = q["question"].get(lang)
            if text:
                yield q, lang, text


def unit(chunk_id: str) -> str:
    return ":".join(chunk_id.split(":")[:2])


def recall(gold: list[str], found: list[str], gold_any: bool) -> float:
    hits = [g in found for g in gold]
    return float(any(hits)) if gold_any else sum(hits) / len(gold)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
def stage_retrieval(questions, langs) -> pd.DataFrame:
    rows = []
    for system in RETRIEVAL_SYSTEMS:
        assistant = Assistant(system)
        for q, lang, text in runs(questions, langs):
            if q["type"] == "unanswerable":
                continue
            sources = expand(assistant.retrieve(text, lang), lang)[:K]
            units = [unit(s.chunk["id"]) for s in sources]
            rows.append({"system": system, "lang": lang, "id": q["id"], "type": q["type"],
                         "translated": bool(q["question"].get("es")),
                         "recall@5": recall(q["gold"], units, q["gold_any"]),
                         "sources": units})
        print(f"retrieval: {system} done")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_json(RESULTS / "retrieval_runs.jsonl", orient="records", lines=True, force_ascii=False)
    return df


def stage_generation(questions, langs) -> None:
    RESULTS.mkdir(exist_ok=True)
    done = {(r["system"], r["id"], r["lang"]) for r in read_jsonl(ANSWERS)}
    llm = LLMClient()
    for system in SYSTEMS:
        assistant = Assistant(system, llm=llm)
        for q, lang, text in runs(questions, langs):
            if (system, q["id"], lang) in done:
                continue
            answer = assistant.ask(text, lang)
            append_jsonl(ANSWERS, {
                "system": system, "id": q["id"], "lang": lang, "type": q["type"],
                "model": llm.model, "question": text, "answer": answer.text,
                "abstained": answer.abstained, "cited_units": answer.cited_units,
                "sources": [{"id": s.chunk["id"], "ref": s.chunk["ref_local"],
                             "text": s.chunk["text"]} for s in answer.sources],
            })
        print(f"generation: {system} done")


def parse_judgement(text: str) -> float | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        claims, supported = int(data["claims"]), int(data["supported"])
    except (ValueError, KeyError, TypeError):
        return None
    return None if claims <= 0 else min(supported, claims) / claims


def stage_judge() -> None:
    done = {(r["system"], r["id"], r["lang"]) for r in read_jsonl(JUDGEMENTS)}
    # Same LLM as the assistant unless JUDGE_BASE_URL / JUDGE_MODEL point elsewhere.
    judge = LLMClient(base_url=os.getenv("JUDGE_BASE_URL"), model=os.getenv("JUDGE_MODEL"),
                      max_tokens=50)
    for row in read_jsonl(ANSWERS):
        key = (row["system"], row["id"], row["lang"])
        if key in done or row["system"] == "llm_only" or row["abstained"]:
            continue
        sources = "\n\n".join(f"[{i}] {s['ref']}\n{s['text']}"
                              for i, s in enumerate(row["sources"], 1))
        prompt = JUDGE_PROMPT.format(sources=sources, answer=row["answer"])
        raw = judge.chat([{"role": "user", "content": prompt}])
        append_jsonl(JUDGEMENTS, {"system": row["system"], "id": row["id"], "lang": row["lang"],
                                  "judge_model": judge.model, "raw": raw,
                                  "faithfulness": parse_judgement(raw)})
    print("judge done")


# --------------------------------------------------------------------------- #
def generation_metrics(questions) -> pd.DataFrame:
    gold = {q["id"]: q for q in questions}
    judged = {(r["system"], r["id"], r["lang"]): r["faithfulness"] for r in read_jsonl(JUDGEMENTS)}
    rows = []
    for r in read_jsonl(ANSWERS):
        q = gold[r["id"]]
        answerable = q["type"] != "unanswerable"
        cited = set(r["cited_units"])
        row = {"system": r["system"], "lang": r["lang"], "id": r["id"], "type": q["type"],
               "translated": bool(q["question"].get("es")), "abstained": r["abstained"]}
        if answerable:
            row["false_abstention"] = float(r["abstained"])
            row["correct_citation"] = float(bool(cited & set(q["gold"])))
            if not r["abstained"] and cited:
                row["citation_precision"] = len(cited & set(q["gold"])) / len(cited)
            row["faithfulness"] = judged.get((r["system"], r["id"], r["lang"]))
        else:
            row["abstention"] = float(r["abstained"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean per system x language, plus 'en (subset)': English on the translated
    questions only, so the three languages are compared on the same questions."""
    parts = [df]
    subset = df[(df["lang"] == "en") & df["translated"]].copy()
    subset["lang"] = "en (subset)"
    parts.append(subset)
    full = pd.concat(parts)
    cols = [m for m in metrics if m in full.columns]
    table = full.groupby(["system", "lang"])[cols].mean().reset_index()
    counts = full.groupby(["system", "lang"]).size().rename("n").reset_index()
    return table.merge(counts, on=["system", "lang"])


def stage_report(questions) -> None:
    from eval.charts import plot_generation, plot_retrieval

    retrieval_path = RESULTS / "retrieval_runs.jsonl"
    lines = ["# Evaluation results", ""]
    if retrieval_path.exists():
        retrieval = summarise(pd.read_json(retrieval_path, lines=True), ["recall@5"])
        retrieval.to_csv(RESULTS / "retrieval_summary.csv", index=False)
        pivot = retrieval.pivot(index="system", columns="lang", values="recall@5")
        pivot = pivot.reindex([s for s in RETRIEVAL_SYSTEMS if s in pivot.index])
        pivot.index = [SYSTEM_LABELS[s] for s in pivot.index]
        lines += ["## Retrieval: Recall@5", "", (pivot * 100).round(1).to_markdown(), ""]
        plot_retrieval(retrieval, RESULTS / "recall_at_5.png")

    if ANSWERS.exists():
        gen = generation_metrics(questions)
        metrics = ["correct_citation", "citation_precision", "faithfulness",
                   "abstention", "false_abstention"]
        summary = summarise(gen, metrics)
        summary.to_csv(RESULTS / "generation_summary.csv", index=False)
        model = read_jsonl(ANSWERS)[0]["model"]
        lines += [f"## Answers ({model})", ""]
        for lang in ["en", "en (subset)", "es", "de"]:
            part = summary[summary["lang"] == lang].set_index("system")
            if part.empty:
                continue
            part = part.reindex([s for s in SYSTEMS if s in part.index])
            part.index = [SYSTEM_LABELS[s] for s in part.index]
            table = (part[[m for m in metrics if m in part.columns]] * 100).round(1)
            lines += [f"### {lang}", "", table.to_markdown(), ""]
        plot_generation(summary, RESULTS / "generation_metrics.png")

    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["retrieval", "generation", "judge", "report", "all"])
    parser.add_argument("--langs", nargs="+", default=LANGS)
    args = parser.parse_args()
    questions = load_questions()
    if args.stage in ("retrieval", "all"):
        stage_retrieval(questions, args.langs)
    if args.stage in ("generation", "all"):
        stage_generation(questions, args.langs)
    if args.stage in ("judge", "all"):
        stage_judge()
    if args.stage in ("report", "retrieval", "all"):
        stage_report(questions)


if __name__ == "__main__":
    main()
