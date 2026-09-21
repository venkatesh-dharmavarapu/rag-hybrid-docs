import pickle
import re
from pathlib import Path
from typing import List
import chromadb
import ollama
from rank_bm25 import BM25Plus

from src.config import settings
from src.ingestion.chunkers import ChunkingEngine, ChunkingStrategy, TextChunk
from src.ingestion.deduplicator import ChunkDeduplicator
from src.ingestion.loaders import DocumentLoader


class IngestionPipeline:
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
        dedup_threshold: float = settings.DEDUPLICATION_THRESHOLD,
    ):
        self.loader = DocumentLoader()
        self.chunker = ChunkingEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.deduplicator = ChunkDeduplicator(threshold=dedup_threshold)
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)

        # ChromaDB setup
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name="internal_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # BM25 storage path
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.bm25_path = settings.DATA_PROCESSED_DIR / "bm25_corpus.pkl"

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            resp = self.ollama_client.embeddings(
                model=settings.EMBEDDING_MODEL,
                prompt=text
            )
            embeddings.append(resp["embedding"])
        return embeddings

    def run(
        self,
        input_paths: List[Path],
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> dict:
        all_chunks: List[TextChunk] = []

        # 1. Load and chunk
        for path in input_paths:
            docs = self.loader.load_file(path)
            for doc in docs:
                chunks = self.chunker.chunk_document(doc, strategy=strategy)
                all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "no_chunks_found", "indexed_count": 0}

        # 2. Embed
        chunk_texts = [c.content for c in all_chunks]
        embeddings = self._get_embeddings(chunk_texts)

        # 3. Deduplicate
        unique_chunks, unique_embeddings, skipped_count = self.deduplicator.filter_duplicates(
            all_chunks, embeddings
        )

        if not unique_chunks:
            return {"status": "all_duplicates_skipped", "skipped": skipped_count}

        # 4. Save to ChromaDB
        ids = [f"{c.source_doc}_{c.chunk_index}_{i}" for i, c in enumerate(unique_chunks)]
        metadatas = []
        for c in unique_chunks:
            meta = {
                "source": c.source_doc,
                "strategy": c.strategy,
                "char_count": c.char_count,
            }
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=unique_embeddings,
            documents=[c.content for c in unique_chunks],
            metadatas=metadatas
        )

        # 5. Build and save BM25 keyword index using BM25Plus and alphanumeric tokens
        tokenized_corpus = [re.findall(r"\w+", c.content.lower()) for c in unique_chunks]
        bm25 = BM25Plus(tokenized_corpus)

        bm25_payload = {
            "bm25": bm25,
            "chunks": [
                {
                    "content": c.content,
                    "source": c.source_doc,
                    "metadata": c.metadata,
                    "id": ids[idx]
                }
                for idx, c in enumerate(unique_chunks)
            ]
        }

        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_payload, f)

        return {
            "status": "success",
            "total_chunks_created": len(all_chunks),
            "duplicates_skipped": skipped_count,
            "indexed_count": len(unique_chunks),
            "bm25_saved_to": str(self.bm25_path)
        }