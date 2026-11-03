"""The assistant: question -> retrieval -> generation -> answer with citations.

Modes compared in the evaluation:
  * "llm_only"       the model alone, no retrieval
  * "dense"          RAG with vector search only
  * "hybrid_rerank"  RAG with BM25 + vector search, fusion and reranker
("hybrid" without the reranker is also available for ablations.)

Usage:
  python -m rag.pipeline "Which AI practices are prohibited?"
  python -m rag.pipeline "¿Qué dice el artículo 22 del RGPD?" --mode dense
  python -m rag.pipeline "Was regelt Anhang III?" --retrieve-only
"""

import argparse

from rag.generate import (Answer, LLMClient, is_abstention, llm_only_messages,
                          parse_free_text_citations, parse_rag_answer, rag_messages)
from rag.context import expand
from rag.lang import detect_language
from rag.retriever import Retriever

MODES = ("llm_only", "dense", "hybrid", "hybrid_rerank")
TOP_K = 5


class Assistant:
    def __init__(self, mode: str = "hybrid_rerank", llm: LLMClient | None = None,
                 top_k: int = TOP_K, min_rerank_score: float | None = None):
        """min_rerank_score: if set (hybrid_rerank only), abstain without calling the
        LLM when the best reranker score is below it."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.top_k = top_k
        self.min_rerank_score = min_rerank_score
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def retrieve(self, question: str, lang: str):
        if self.mode == "llm_only":
            return []
        # Twice as many chunks as sources: chunks of the same paragraph are
        # merged later, so this still yields top_k distinct sources.
        return Retriever(lang, self.mode).search(question, 2 * self.top_k)

    def ask(self, question: str, lang: str | None = None) -> Answer:
        lang = lang or detect_language(question)
        if self.mode == "llm_only":
            text = self.llm.chat(llm_only_messages(question, lang))
            abstained = is_abstention(text)
            units = [] if abstained else parse_free_text_citations(text)
            return Answer(question, lang, self.mode, text, abstained, units, units)

        hits = self.retrieve(question, lang)
        top_score = hits[0].rerank_score if hits else None
        if (self.min_rerank_score is not None and top_score is not None
                and top_score < self.min_rerank_score):
            return Answer(question, lang, self.mode, "NO_ANSWER", True, [], [], hits)

        sources = expand(hits, lang)[:self.top_k]
        text = self.llm.chat(rag_messages(question, sources, lang))
        abstained = is_abstention(text)
        units, refs = ([], []) if abstained else parse_rag_answer(text, sources)
        return Answer(question, lang, self.mode, text, abstained, units, refs, sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    parser.add_argument("--lang", choices=["en", "es", "de"])
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    assistant = Assistant(args.mode)
    lang = args.lang or detect_language(args.question)
    if args.retrieve_only:
        for i, hit in enumerate(assistant.retrieve(args.question, lang), 1):
            print(f"[{i}] {hit.score:.3f}  {hit.chunk['ref_local']} — {hit.chunk['title']}")
        return
    answer = assistant.ask(args.question, lang)
    print(answer.display_text)
    if answer.cited_refs:
        print("\nSources:\n" + "\n".join(f"  - {r}" for r in answer.cited_refs))


if __name__ == "__main__":
    main()
