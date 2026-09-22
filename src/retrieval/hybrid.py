from typing import Any, Dict, List
from src.retrieval.sparse_store import SparseRetriever
from src.retrieval.vector_store import DenseRetriever


class HybridRetriever:
    """
    Combines Dense (vector) and Sparse (BM25) search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        sparse_retriever: SparseRetriever = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Runs both retrievers and merges their results via weighted RRF.
        """
        dense_results = self.dense.search(query, top_k=top_k)
        sparse_results = self.sparse.search(query, top_k=top_k)

        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Accumulate dense ranks
        for rank, item in enumerate(dense_results):
            chunk_id = item["id"]
            chunk_map[chunk_id] = item
            score = self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score

        # 2. Accumulate sparse ranks
        for rank, item in enumerate(sparse_results):
            chunk_id = item["id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = item
            score = self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score

        # 3. Sort chunks descending by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_ids[:top_k]:
            merged_item = chunk_map[cid].copy()
            merged_item["score"] = fused_scores[cid]
            merged_item["retriever"] = "hybrid_rrf"
            final_results.append(merged_item)

        return final_results