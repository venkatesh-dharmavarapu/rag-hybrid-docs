from typing import List, Tuple
import numpy as np
from src.ingestion.chunkers import TextChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_val = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_val / (norm_a * norm_b))


class ChunkDeduplicator:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._stored_embeddings: List[np.ndarray] = []

    def filter_duplicates(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]]
    ) -> Tuple[List[TextChunk], List[List[float]], int]:
        unique_chunks: List[TextChunk] = []
        unique_embeddings: List[List[float]] = []
        skipped_count = 0

        for chunk, emb in zip(chunks, embeddings):
            emb_vec = np.array(emb, dtype=np.float32)

            is_duplicate = False
            for existing_vec in self._stored_embeddings:
                sim = cosine_similarity(emb_vec, existing_vec)
                if sim >= self.threshold:
                    is_duplicate = True
                    skipped_count += 1
                    break

            if not is_duplicate:
                self._stored_embeddings.append(emb_vec)
                unique_chunks.append(chunk)
                unique_embeddings.append(emb)

        return unique_chunks, unique_embeddings, skipped_count