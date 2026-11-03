"""Embed every chunk with BAAI/bge-m3 (dense, multilingual, 1024 dims).

Writes, per language:
  data/index/embeddings_<lang>.npy   float16, L2-normalised, shape (n_chunks, 1024)
  data/index/ids_<lang>.json         chunk ids in the same row order
  data/index/hashes_<lang>.json      hash of each embedded text, so a re-run only
                                     embeds chunks whose text changed

On a CPU this takes roughly half an hour for the three languages; on a Colab
GPU it takes a couple of minutes.

Usage: python -m ingest.embed [--langs en es de] [--batch-size 16]
"""

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ingest.config import INDEX_DIR, LANGUAGES, PROCESSED_DIR

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 1024  # longest chunk is ~1,100 tokens; keeps CPU memory in check


def load_chunks(lang: str) -> list[dict]:
    path = PROCESSED_DIR / f"chunks_{lang}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache(lang: str) -> dict:
    """hash -> vector from a previous run (empty if there is none)."""
    paths = [INDEX_DIR / f"{name}_{lang}.{ext}" for name, ext in
             (("embeddings", "npy"), ("hashes", "json"))]
    if not all(p.exists() for p in paths):
        return {}
    vectors = np.load(paths[0])
    hashes = json.loads(paths[1].read_text())
    return dict(zip(hashes, vectors)) if len(hashes) == len(vectors) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    if device == "cuda":
        model.half()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        chunks = load_chunks(lang)
        hashes = [text_hash(c["text_for_embedding"]) for c in chunks]
        cache = load_cache(lang)
        todo = [i for i, h in enumerate(hashes) if h not in cache]
        # Sort by length so batches have similar padding (much faster on CPU).
        todo.sort(key=lambda i: len(chunks[i]["text_for_embedding"]))
        start = time.time()
        embeddings = np.zeros((len(chunks), 1024), dtype=np.float16)
        if todo:
            vectors = model.encode([chunks[i]["text_for_embedding"] for i in todo],
                                   batch_size=args.batch_size, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_numpy=True)
            embeddings[todo] = vectors.astype(np.float16)
        for i, h in enumerate(hashes):
            if h in cache:
                embeddings[i] = cache[h]
        np.save(INDEX_DIR / f"embeddings_{lang}.npy", embeddings)
        (INDEX_DIR / f"ids_{lang}.json").write_text(json.dumps([c["id"] for c in chunks]))
        (INDEX_DIR / f"hashes_{lang}.json").write_text(json.dumps(hashes))
        print(f"{lang}: {len(chunks)} chunks, {len(todo)} embedded, "
              f"{len(chunks) - len(todo)} reused, {time.time() - start:.0f}s on {device}")


if __name__ == "__main__":
    main()
