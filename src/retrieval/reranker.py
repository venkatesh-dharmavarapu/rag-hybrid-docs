from typing import Any, Dict, List
from sentence_transformers import CrossEncoder


class DocumentReranker:
    """
    Reranks candidate chunks using a lightweight cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scores each (query, chunk_content) pair and returns top_n candidates.
        """
        if not candidates:
            return []

        pairs = [[query, candidate["content"]] for candidate in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            item = candidate.copy()
            item["rerank_score"] = float(scores[idx])
            scored_candidates.append(item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_candidates[:top_n]