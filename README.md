# Enterprise Hybrid RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system operating over internal technical documentation. Combines dense semantic vector retrieval (ChromaDB) with lexical sparse search (BM25Plus), ranked via Reciprocal Rank Fusion (RRF), cross-encoder reranking, and citation verification via LLM-as-judge.

---

## Architecture Overview

[Raw Docs: PDF / MD / HTML / TXT]
│
▼
[Multi-Format Loader]
│
▼
[Configurable Chunker] ── (Fixed / Recursive / Semantic)
│
▼
[Deduplication Engine] ── (Cosine similarity > 0.95 skipped)
│
┌───────┴────────────────────────┐
▼                                ▼
[ChromaDB Dense Index]        [BM25Plus Sparse Index]
(nomic-embed-text)            (Exact Token Matching)
│                                │
└───────┬────────────────────────┘
▼
[Weighted RRF Fusion] ── (k = 60)
│ (Top 15 candidates)
▼
[Cross-Encoder Reranker] ── (ms-marco-MiniLM-L-6-v2)
│ (Top 3-4 chunks)
▼
[Grounded Generation] ── (llama3.2 with strict inline [n] citations)
│
┌───────┴────────────────────────┐
▼                                ▼
[Citation Verifier (Judge)]   [Confidence Scorer]
(Claim-by-claim entailment)   (Composite 0.0 - 1.0)

---

## Key Engineering Decisions

| Problem | Naive Solution | Production Solution Implemented |
| :--- | :--- | :--- |
| **Out-of-Vocabulary & Code Lookups** | Vector-only embeddings fail on exact IDs (`ERR_AUTH_001`). | **Hybrid Search**: BM25 keyword matching runs alongside ChromaDB dense vectors. |
| **Score Incompatibility** | Unstable Min-Max normalization between cosine `[-1, 1]` and BM25 `[0, ∞)`. | **Reciprocal Rank Fusion (RRF)**: Merges candidates based purely on positional rank. |
| **Context Window Noise** | Top-K bi-encoder chunks include partial topical overlap. | **Cross-Encoder Reranker**: Jointly models query-chunk interactions to retain high-precision context. |
| **Hallucinated Citations** | Trusting model-generated brackets without verification. | **LLM-as-Judge Auditor**: Extracts claim-citation pairs and asserts entailment against source chunks. |
| **Duplicate Bloat** | Re-indexing repeated corporate boilerplate across multiple docs. | **Vector Deduplication**: Skips chunks with cosine similarity $> 0.95$ prior to index commits. |

---

## Chunking Strategy Benchmark

Evaluated on the internal Golden Dataset (`src/evaluation/golden_dataset.json`):

| Strategy | Retrieval Hit Rate (%) | Answer Correctness (%) | Citation Accuracy (%) | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Fixed Size (500 chars)** | 60.0% | 60.0% | 80.0% | Splits structural codes across chunk boundaries. |
| **Semantic Paragraphs** | 80.0% | 80.0% | 85.0% | Variable chunk lengths dilute dense embeddings. |
| **Recursive Character (Ours)** | **100.0%** | **100.0%** | **100.0%** | Preserves section headings and keeps code lists intact. |

---

## Quickstart (Local Deployment)

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2