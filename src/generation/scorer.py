from typing import Any, Dict, List
from src.generation.verifier import CitationCheck


class ConfidenceScorer:
    """Calculates composite confidence scores for generated answers."""

    @staticmethod
    def calculate_score(
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        citation_checks: List[CitationCheck]
    ) -> Dict[str, float]:
        """
        Returns a dictionary with sub-scores and a final composite score (0.0 - 1.0).
        """
        # 1. Check for fallback response
        if "INSUFFICIENT_CONTEXT" in answer or not retrieved_chunks:
            return {
                "retrieval_confidence": 0.0,
                "citation_coverage": 0.0,
                "composite_score": 0.0
            }

        # 2. Retrieval Score (normalized average of top chunk rerank scores)
        # CrossEncoder scores typically range from -10 to +10; normalize to 0..1
        raw_scores = [c.get("rerank_score", c.get("score", 0.0)) for c in retrieved_chunks[:3]]
        avg_score = sum(raw_scores) / max(len(raw_scores), 1)
        # Bound between 0.1 and 1.0
        retrieval_confidence = max(0.1, min(1.0, (avg_score + 5.0) / 10.0))

        # 3. Citation Coverage (percentage of verified citations)
        if not citation_checks:
            # If no citations were made, confidence is reduced
            citation_coverage = 0.5
        else:
            supported_count = sum(1 for c in citation_checks if c.is_supported)
            citation_coverage = supported_count / len(citation_checks)

        # 4. Composite Score: 40% retrieval strength + 60% citation accuracy
        composite = (0.4 * retrieval_confidence) + (0.6 * citation_coverage)

        return {
            "retrieval_confidence": round(retrieval_confidence, 2),
            "citation_coverage": round(citation_coverage, 2),
            "composite_score": round(composite, 2)
        }