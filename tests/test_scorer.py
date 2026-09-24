from src.generation.scorer import ConfidenceScorer
from src.generation.verifier import CitationCheck


def test_confidence_scorer():
    checks = [
        CitationCheck(claim="A", citation_id=1, is_supported=True, reason="ok"),
        CitationCheck(claim="B", citation_id=2, is_supported=True, reason="ok"),
    ]
    chunks = [{"rerank_score": 2.5}]

    scores = ConfidenceScorer.calculate_score("Answer with [1] and [2].", chunks, checks)

    assert scores["citation_coverage"] == 1.0
    assert scores["composite_score"] > 0.7


def test_confidence_scorer_insufficient_context():
    scores = ConfidenceScorer.calculate_score("INSUFFICIENT_CONTEXT", [], [])
    assert scores["composite_score"] == 0.0