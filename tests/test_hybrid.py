from src.retrieval.hybrid import HybridRetriever


def test_hybrid_rrf_retrieval():
    hybrid = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
    results = hybrid.search("What is the error code for token missing?", top_k=5)

    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert "content" in top
    assert top["retriever"] == "hybrid_rrf"
    assert top["score"] > 0.0