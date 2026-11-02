# EU AI Act & GDPR Assistant

A retrieval-augmented assistant that answers questions about the EU AI Act
(Regulation (EU) 2024/1689, as amended by the Digital Omnibus on AI) and the
GDPR (Regulation (EU) 2016/679), citing the exact article, annex or recital
behind each answer, and an evaluation that measures how well it works in
English, Spanish and German.

> Work in progress. Day 1 (ingestion) is done; retrieval, generation,
> evaluation and the demo follow.

## Pipeline so far

```bash
pip install -r requirements.txt
python -m ingest.download   # official texts from EUR-Lex + version check
python -m ingest.parse      # article/paragraph chunks with metadata
python -m ingest.embed      # bge-m3 embeddings (GPU recommended)
python -m eval.validate_questions
```

See [docs/data_sources.md](docs/data_sources.md) for the exact versions used.

## Repository layout

```
data/      raw EUR-Lex XHTML (git-ignored), processed chunks, embeddings
ingest/    download, article-level parsing, embeddings
rag/       hybrid retrieval, reranking, generation
eval/      questions.json + evaluation scripts
results/   tables and charts
```

Not legal advice.
