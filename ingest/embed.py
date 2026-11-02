"""Embed every chunk with BAAI/bge-m3 (dense, multilingual, 1024 dims).

Writes, per language:
  data/index/embeddings_<lang>.npy   float16, L2-normalised, shape (n_chunks, 1024)
  data/index/ids_<lang>.json         chunk ids in the same row order

On a CPU this takes roughly half an hour for the three languages; on a Colab
GPU it takes a couple of minutes.

Usage: python -m ingest.embed [--langs en es de] [--batch-size 16]
"""

import argparse
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
        # Sort by length so batches have similar padding (much faster on CPU).
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["text_for_embedding"]))
        texts = [chunks[i]["text_for_embedding"] for i in order]
        start = time.time()
        vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.empty_like(vectors)
        embeddings[order] = vectors
        np.save(INDEX_DIR / f"embeddings_{lang}.npy", embeddings.astype(np.float16))
        (INDEX_DIR / f"ids_{lang}.json").write_text(json.dumps([c["id"] for c in chunks]))
        print(f"{lang}: {embeddings.shape} in {time.time() - start:.0f}s on {device}")


if __name__ == "__main__":
    main()
