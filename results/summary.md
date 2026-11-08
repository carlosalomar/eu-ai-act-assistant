# Evaluation results

## Retrieval: Recall@5

|                         |   de |   en |   en (subset) |   es |
|:------------------------|-----:|-----:|--------------:|-----:|
| RAG (vector)            | 81.6 | 90   |          86.8 | 78.9 |
| BM25 only               | 60.5 | 68.8 |          60.5 | 60.5 |
| Hybrid (no reranker)    | 73.7 | 87.5 |          81.6 | 78.9 |
| RAG (hybrid + reranker) | 94.7 | 92.5 |          92.1 | 92.1 |

## Answers (Qwen/Qwen2.5-7B-Instruct-AWQ)

### en

|                         |   correct_citation |   citation_precision |   faithfulness |   abstention |   false_abstention |
|:------------------------|-------------------:|---------------------:|---------------:|-------------:|-------------------:|
| LLM only                |               27.5 |                 26.6 |          nan   |          100 |                 20 |
| RAG (vector)            |               92.5 |                 62   |           84   |           90 |                  0 |
| RAG (hybrid + reranker) |               92.5 |                 63.5 |           86.1 |           90 |                  0 |

### en (subset)

|                         |   correct_citation |   citation_precision |   faithfulness |   abstention |   false_abstention |
|:------------------------|-------------------:|---------------------:|---------------:|-------------:|-------------------:|
| LLM only                |               26.3 |                 21.9 |          nan   |          100 |               15.8 |
| RAG (vector)            |               89.5 |                 61.1 |           81.2 |          100 |                0   |
| RAG (hybrid + reranker) |               94.7 |                 62.3 |           82.8 |          100 |                0   |

### es

|                         |   correct_citation |   citation_precision |   faithfulness |   abstention |   false_abstention |
|:------------------------|-------------------:|---------------------:|---------------:|-------------:|-------------------:|
| LLM only                |               21.1 |                 23.6 |          nan   |        100   |               36.8 |
| RAG (vector)            |               68.4 |                 57.4 |           83.7 |         66.7 |                5.3 |
| RAG (hybrid + reranker) |               94.7 |                 76.1 |           84.9 |        100   |                0   |

### de

|                         |   correct_citation |   citation_precision |   faithfulness |   abstention |   false_abstention |
|:------------------------|-------------------:|---------------------:|---------------:|-------------:|-------------------:|
| LLM only                |               15.8 |                 12.5 |          nan   |        100   |               15.8 |
| RAG (vector)            |               78.9 |                 58.3 |           88.2 |         66.7 |                0   |
| RAG (hybrid + reranker) |               94.7 |                 80.7 |           82.3 |        100   |                0   |
