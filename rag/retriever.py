"""Dense, BM25 and hybrid retrieval with an optional cross-encoder reranker.

Retrieval modes (the ones compared in the evaluation):
  * "dense"          bge-m3 embeddings + FAISS inner-product search
  * "hybrid"         dense + BM25 merged with Reciprocal Rank Fusion
  * "hybrid_rerank"  hybrid candidates re-scored by bge-reranker-v2-m3

BM25 gets one extra trick: every chunk is indexed with reference tokens built
from its metadata ("ref_art_6", "ref_ai_act_art_6", "ref_anx_iii"), and explicit
references in the question ("Article 6", "artículo 22 del RGPD", "Anhang III")
are turned into the same tokens. That gives exact matches on article numbers,
which embeddings miss.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ingest.config import INDEX_DIR, PROCESSED_DIR
from rag.references import find_references

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60            # standard constant in Reciprocal Rank Fusion
CANDIDATES = 30       # chunks per retriever before fusion / reranking
REF_TOKEN_WEIGHT = 3  # how many times a reference token is repeated in a query
MAX_PINNED = 3        # explicitly referenced units guaranteed a place in the top k
# Recitals explain the law but are not the law: the reranker tends to prefer
# their prose, so their score is discounted to put articles and annexes first.
RECITAL_WEIGHT = 0.8

STOPWORDS = set("""
a an and are as at be by for from has have how in is it its of on or that the this to was
what when which who will with does do can shall should must may any under about
el la los las un una unos unas y o de del al en con por para que qué es son se su sus lo
como cómo cuál cuáles cuando cuándo debe deben puede pueden tiene tienen hay sobre si
der die das ein eine einer eines und oder von vom zu zum zur im in mit für ist sind wird werden
was wie wann welche welcher welches hat haben kann können muss müssen auf an bei des den dem
""".split())


@dataclass
class Hit:
    chunk: dict
    score: float           # score of the stage that produced the final ranking
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    @property
    def unit(self) -> str:
        """Article/annex/recital id used for citations: 'ai_act:art5'."""
        return ":".join(self.chunk["id"].split(":")[:2])


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def chunk_tokens(chunk: dict) -> list[str]:
    """Body + title words. The citation header is left out on purpose: its
    numbers ("Article 97(6)") would match any "6" in a query; article numbers
    are matched through the reference tokens instead."""
    words = " ".join(filter(None, [chunk["title"], chunk["chapter_title"], chunk["text"]]))
    return tokenize(words) + chunk_ref_tokens(chunk)


def query_tokens(query: str) -> list[str]:
    """Query words without the reference numbers, plus reference tokens."""
    from rag.references import ANNEX_RE, ARTICLE_RE, RECITAL_RE
    plain = query
    for regex in (ARTICLE_RE, ANNEX_RE, RECITAL_RE):
        plain = regex.sub(" ", plain)
    return tokenize(plain) + query_ref_tokens(query)


def chunk_ref_tokens(chunk: dict) -> list[str]:
    kind = {"article": "art", "annex": "anx", "recital": "rct"}[chunk["kind"]]
    number = chunk["number"].lower()
    return [f"ref_{kind}_{number}", f"ref_{chunk['law']}_{kind}_{number}"]


def query_ref_tokens(query: str) -> list[str]:
    tokens = []
    for law, unit in find_references(query):
        kind, number = unit[:3], unit[3:].lower()
        tokens.append(f"ref_{law}_{kind}_{number}" if law else f"ref_{kind}_{number}")
    return tokens * REF_TOKEN_WEIGHT


class Corpus:
    """Chunks, embeddings and BM25 index for one language."""

    def __init__(self, lang: str):
        self.lang = lang
        path = PROCESSED_DIR / f"chunks_{lang}.jsonl"
        self.chunks = [json.loads(line) for line in path.open(encoding="utf-8")]
        ids = json.loads((INDEX_DIR / f"ids_{lang}.json").read_text())
        if ids != [c["id"] for c in self.chunks]:
            raise RuntimeError(f"Embeddings for '{lang}' are stale: run python -m ingest.embed")
        vectors = np.load(INDEX_DIR / f"embeddings_{lang}.npy").astype(np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([chunk_tokens(c) for c in self.chunks])
        self.by_unit: dict[str, list[int]] = {}
        for row, chunk in enumerate(self.chunks):
            self.by_unit.setdefault(":".join(chunk["id"].split(":")[:2]), []).append(row)


@lru_cache(maxsize=None)
def get_corpus(lang: str) -> Corpus:
    return Corpus(lang)


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.half() if model.device.type == "cuda" else model  # share the GPU with vLLM


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(RERANKER_MODEL, max_length=1024)
    if model.device.type == "cuda":
        model.model.half()
    return model


class Retriever:
    def __init__(self, lang: str, mode: str = "hybrid_rerank"):
        if mode not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError(f"unknown retrieval mode {mode}")
        self.corpus = get_corpus(lang)
        self.mode = mode

    # -- single retrievers ------------------------------------------------- #
    def dense(self, query: str, k: int = CANDIDATES) -> list[Hit]:
        vector = embedding_model().encode([query], normalize_embeddings=True)
        scores, rows = self.corpus.index.search(np.asarray(vector, dtype=np.float32), k)
        return [Hit(self.corpus.chunks[r], float(s), dense_rank=i)
                for i, (r, s) in enumerate(zip(rows[0], scores[0]))]

    def bm25(self, query: str, k: int = CANDIDATES) -> list[Hit]:
        scores = self.corpus.bm25.get_scores(query_tokens(query))
        rows = np.argsort(-scores)[:k]
        return [Hit(self.corpus.chunks[r], float(scores[r]), bm25_rank=i)
                for i, r in enumerate(rows) if scores[r] > 0]

    # -- combinations ------------------------------------------------------ #
    def hybrid(self, query: str, k: int = CANDIDATES) -> list[Hit]:
        """Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank))."""
        fused: dict[str, Hit] = {}
        for hits, attr in ((self.dense(query), "dense_rank"), (self.bm25(query), "bm25_rank")):
            for rank, hit in enumerate(hits):
                entry = fused.setdefault(hit.chunk["id"], Hit(hit.chunk, 0.0))
                setattr(entry, attr, rank)
                entry.score += 1.0 / (RRF_K + rank + 1)
        return sorted(fused.values(), key=lambda h: -h.score)[:k]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        pairs = [(query, h.chunk["text_for_embedding"]) for h in hits]
        scores = reranker().predict(pairs, batch_size=8)
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
            hit.score = hit.rerank_score * (RECITAL_WEIGHT if hit.chunk["kind"] == "recital" else 1.0)
        return sorted(hits, key=lambda h: -h.score)

    def referenced_units(self, query: str) -> list[str]:
        """Units named in the query that exist in the corpus. A bare "Article 6"
        (no law named) resolves to the AI Act first, then the GDPR."""
        units = []
        for law, unit in find_references(query):
            for candidate_law in ([law] if law else ["ai_act", "gdpr"]):
                key = f"{candidate_law}:{unit}"
                if key in self.corpus.by_unit and key not in units:
                    units.append(key)
        return units[:MAX_PINNED]

    def pin_references(self, query: str, ranked: list[Hit], k: int) -> list[Hit]:
        """Make sure every article/annex named in the question is in the top k,
        using its best-ranked chunk (or its first chunk if it was not retrieved)."""
        pinned = []
        for unit in self.referenced_units(query):
            best = next((h for h in ranked if h.unit == unit), None)
            if best is None:
                best = Hit(self.corpus.chunks[self.corpus.by_unit[unit][0]], 0.0)
            pinned.append(best)
        rest = [h for h in ranked if all(h is not p for p in pinned)]
        return (pinned + rest)[:k] if pinned else ranked[:k]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self.mode == "dense":
            return self.dense(query, k)
        if self.mode == "bm25":
            return self.bm25(query, k)
        candidates = self.hybrid(query)
        if self.mode == "hybrid_rerank":
            candidates = self.rerank(query, candidates)
        return self.pin_references(query, candidates, k)
