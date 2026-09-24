from src.generation.pipeline import RAGPipeline


def test_end_to_end_rag():
    rag = RAGPipeline()
    result = rag.query("What header is required for payment API requests?")

    assert "answer" in result
    assert "confidence" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    # Must mention Authorization or Bearer from sample_api.md
    assert "Authorization" in result["answer"] or "Bearer" in result["answer"]