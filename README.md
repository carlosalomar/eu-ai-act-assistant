# EU AI Act & GDPR Assistant

A retrieval-augmented assistant that answers questions about the EU AI Act
(Regulation (EU) 2024/1689, as amended by the Digital Omnibus on AI) and the
GDPR (Regulation (EU) 2016/679), citing the exact article, annex or recital
behind each answer, and an evaluation that measures how well it works in
English, Spanish and German.

> Work in progress: ingestion and the RAG pipeline are done; the evaluation
> and the demo follow.

## Pipeline so far

```bash
pip install -r requirements.txt
python -m ingest.download   # official texts from EUR-Lex + version check
python -m ingest.parse      # article/paragraph chunks with metadata
python -m ingest.embed      # bge-m3 embeddings (GPU recommended)
python -m eval.validate_questions
pytest tests
```

The LLM runs on Google Colab with vLLM: open
[notebooks/colab_rag.ipynb](notebooks/colab_rag.ipynb). Any OpenAI-compatible
endpoint works locally too (`LLM_BASE_URL`, `LLM_MODEL`):

```bash
python -m rag.pipeline "Which AI practices are prohibited?"            # hybrid + reranker
python -m rag.pipeline "¿Qué dice el artículo 22 del RGPD?" --mode dense
python -m rag.pipeline "Was regelt Anhang III?" --retrieve-only
```

## How it works

1. **Chunking by legal unit.** Articles are split by paragraph, long paragraphs
   by point, annexes by top-level point, and each recital is one chunk. Every
   chunk carries law, article, paragraph, title, chapter and section.
2. **Hybrid retrieval.** bge-m3 dense vectors (FAISS) and BM25 are merged with
   Reciprocal Rank Fusion. BM25 also indexes reference tokens, so "Article 6",
   "artículo 22 del RGPD" or "Anhang III" match exactly; articles named in the
   question are always kept among the sources.
3. **Reranking.** bge-reranker-v2-m3 re-scores the 30 fused candidates; recitals
   are slightly discounted so operative articles come first.
4. **Small-to-big context.** Chunks that are points of the same paragraph are
   merged back into the full paragraph before reaching the model.
5. **Cited answers.** The model (Qwen2.5-7B-Instruct served with vLLM) answers
   only from numbered sources, cites them as [n] (mapped back to "AI Act,
   Article 5(1)") and answers `NO_ANSWER` when the sources do not cover the
   question.

Three systems are compared in the evaluation: `llm_only` (no retrieval),
`dense` (vector search) and `hybrid_rerank`.

See [docs/data_sources.md](docs/data_sources.md) for the exact versions used.

## Repository layout

```
data/      raw EUR-Lex XHTML (git-ignored), processed chunks, embeddings
ingest/    download, article-level parsing, embeddings
rag/       references, retrieval, context building, generation, pipeline
notebooks/ Colab notebook (vLLM)
tests/     unit tests
eval/      questions.json + evaluation scripts
results/   tables and charts
```

Not legal advice.
