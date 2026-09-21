from src.retrieval.vector_store import DenseRetriever


def test_dense_retriever():
    retriever = DenseRetriever()
    results = retriever.search("How do I authenticate with the API?", top_k=2)

    # Ensure results come back as a list
    assert isinstance(results, list)
    if len(results) > 0:
        first = results[0]
        assert "id" in first
        assert "content" in first
        assert "score" in first
        assert first["retriever"] == "dense"
        assert "Authentication" in first["content"] or "token" in first["content"].lower()