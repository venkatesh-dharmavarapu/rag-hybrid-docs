from dataclasses import dataclass
from enum import Enum
from typing import List
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.ingestion.loaders import Document


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class TextChunk:
    content: str
    source_doc: str
    chunk_index: int
    strategy: str
    char_count: int
    metadata: dict


class ChunkingEngine:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        doc: Document,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> List[TextChunk]:
        if strategy == ChunkingStrategy.FIXED:
            raw_chunks = self._fixed_chunk(doc.content)
        elif strategy == ChunkingStrategy.RECURSIVE:
            raw_chunks = self._recursive_chunk(doc.content)
        elif strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self._semantic_chunk(doc.content)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        chunks: List[TextChunk] = []
        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            meta = doc.metadata.copy()
            meta["chunk_index"] = idx
            chunks.append(
                TextChunk(
                    content=text,
                    source_doc=doc.source,
                    chunk_index=idx,
                    strategy=strategy.value,
                    char_count=len(text),
                    metadata=meta,
                )
            )
        return chunks

    def _fixed_chunk(self, text: str) -> List[str]:
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def _recursive_chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def _semantic_chunk(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        result = []
        buffer = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(buffer) + len(p) <= self.chunk_size:
                buffer = f"{buffer}\n\n{p}".strip()
            else:
                if buffer:
                    result.append(buffer)
                buffer = p

        if buffer:
            result.append(buffer)
        return result