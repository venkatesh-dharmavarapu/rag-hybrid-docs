import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings


def tokenize(text: str) -> List[str]:
    """Extracts lowercase alphanumeric tokens, stripping punctuation."""
    return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """Retrieves context chunks using keyword matching via BM25."""

    def __init__(self, index_file: Path = None):
        self.index_file = index_file or (settings.DATA_PROCESSED_DIR / "bm25_corpus.pkl")
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        """Loads serialized BM25 model and chunk metadata."""
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_file}. Run Phase 1 ingestion first!"
            )

        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Pair scores with corresponding chunks
        scored_pairs = list(enumerate(scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        query_set = set(tokenized_query)

        for idx, score in scored_pairs[:top_k]:
            chunk_data = self.chunks[idx]
            chunk_tokens = set(tokenize(chunk_data["content"]))

            # Only return chunks that contain at least one query term
            if query_set.intersection(chunk_tokens):
                results.append({
                    "id": chunk_data["id"],
                    "content": chunk_data["content"],
                    "score": float(score),
                    "metadata": chunk_data["metadata"],
                    "retriever": "sparse"
                })

        return results