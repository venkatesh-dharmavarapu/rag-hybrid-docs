from src.retrieval.sparse_store import SparseRetriever


def test_sparse_retriever_exact_keyword():
    retriever = SparseRetriever()
    # Search for an exact error code that exists in sample_api.md
    results = retriever.search("ERR_AUTH_001", top_k=1)

    assert len(results) > 0
    assert "ERR_AUTH_001" in results[0]["content"]
    assert results[0]["retriever"] == "sparse"