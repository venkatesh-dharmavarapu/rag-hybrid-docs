from src.retrieval.reranker import DocumentReranker


def test_reranker_relevance_ordering():
    query = "How long do auth tokens stay valid?"
    mock_candidates = [
        {
            "id": "doc_irrelevant",
            "content": "Our headquarters is located in San Francisco.",
            "score": 0.5,
            "metadata": {}
        },
        {
            "id": "doc_relevant",
            "content": "Tokens expire after 15 minutes of inactivity.",
            "score": 0.4,
            "metadata": {}
        }
    ]

    reranker = DocumentReranker()
    ranked = reranker.rerank(query, mock_candidates, top_n=2)

    assert len(ranked) == 2
    # The token document must be ranked #1 despite lower initial mock score
    assert ranked[0]["id"] == "doc_relevant"
    assert "15 minutes" in ranked[0]["content"]