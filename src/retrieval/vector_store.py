from typing import Any, Dict, List
import chromadb
import ollama

from src.config import settings


class DenseRetriever:
    """Retrieves context chunks using vector cosine similarity via ChromaDB."""

    def __init__(self, collection_name: str = "internal_docs"):
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed_query(self, query: str) -> List[float]:
        """Embeds the query text using the configured Ollama embedding model."""
        resp = self.ollama_client.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=query
        )
        return resp["embedding"]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for the top-k chunks closest in meaning to the query.
        Returns a list of dicts with id, content, score, and metadata.
        """
        query_embedding = self._embed_query(query)

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # ChromaDB cosine distance is (1 - cosine_similarity).
            # Convert to similarity score between 0 and 1:
            similarity = 1.0 - distances[i]
            candidates.append({
                "id": ids[i],
                "content": documents[i],
                "score": float(similarity),
                "metadata": metadatas[i],
                "retriever": "dense"
            })

        return candidates