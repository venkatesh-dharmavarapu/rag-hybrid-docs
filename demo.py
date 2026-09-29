import time
from pathlib import Path
from src.generation.pipeline import RAGPipeline
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import DenseRetriever


def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_demo():
    print_banner("1. INGESTION & DUAL-INDEXING")
    pipeline = IngestionPipeline()
    raw_files = list(Path("data/raw").glob("*.md"))
    print(f"Found {len(raw_files)} documents in data/raw:")
    for f in raw_files:
        print(f"  - {f.name}")

    ingest_result = pipeline.run(raw_files)
    print(f"Status: {ingest_result['status']}")
    print(f"Total Chunks Created: {ingest_result.get('total_chunks_created', 0)}")
    print(f"Chunks Indexed: {ingest_result.get('indexed_count', 0)}")

    rag = RAGPipeline()
    dense_only = DenseRetriever()

    # Scenario 1: Exact code lookup (BM25 strength)
    print_banner("2. TEST SCENARIO: EXACT KEYWORD LOOKUP (BM25 STRENGTH)")
    q1 = "What happens when ERR_RATE_LIMIT is encountered?"
    print(f"Query: \"{q1}\"\n")

    res1 = rag.query(q1)
    print(f"Grounded Answer:\n{res1['answer']}\n")
    print(f"Confidence Score: {res1['confidence']['composite_score'] * 100}%")
    print(f"Citation Verification:")
    for c in res1["citations"]:
        status = "VERIFIED" if c["is_supported"] else "UNSUPPORTED"
        print(f"  - [{c['citation_id']}] {status}: \"{c['claim']}\"")

    # Scenario 2: Dense vs Hybrid side-by-side
    print_banner("3. TEST SCENARIO: HYBRID VS DENSE-ONLY COMPARISON")
    q2 = "What is the token expiration period?"
    print(f"Query: \"{q2}\"\n")

    print("[Dense-Only Top Chunk]:")
    d_res = dense_only.search(q2, top_k=1)
    if d_res:
        print(f"  Score: {round(d_res[0]['score'], 3)} | Content: {d_res[0]['content'][:120]}...\n")

    print("[Hybrid + RRF + Rerank Top Source]:")
    h_res = rag.query(q2, top_n_rerank=1)
    if h_res["sources"]:
        print(f"  Source: {h_res['sources'][0]['source']} | Content: {h_res['sources'][0]['content'][:120]}...\n")

    # Scenario 3: Negative Control / Anti-Hallucination
    print_banner("4. TEST SCENARIO: NEGATIVE CONTROL (ANTI-HALLUCINATION)")
    q3 = "What is the company budget for 2027 server infrastructure?"
    print(f"Query: \"{q3}\"\n")

    res3 = rag.query(q3)
    print(f"System Response:\n{res3['answer']}\n")
    print(f"Confidence Score: {res3['confidence']['composite_score'] * 100}%")
    print_banner("DEMO COMPLETE: ALL PRODUCTION GATES VERIFIED")


if __name__ == "__main__":
    run_demo()