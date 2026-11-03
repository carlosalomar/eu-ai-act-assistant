# Evaluation results

## Retrieval: Recall@5

|                         |   de |   en |   en (subset) |   es |
|:------------------------|-----:|-----:|--------------:|-----:|
| RAG (vector)            | 81.6 | 90   |          86.8 | 78.9 |
| BM25 only               | 60.5 | 68.8 |          60.5 | 60.5 |
| Hybrid (no reranker)    | 73.7 | 87.5 |          81.6 | 78.9 |
| RAG (hybrid + reranker) | 94.7 | 92.5 |          92.1 | 92.1 |
