from src.ingestion.chunkers import ChunkingEngine, ChunkingStrategy
from src.ingestion.loaders import Document


def test_chunking_engine():
    doc_text = (
        "## Setup Guide\n"
        "First, install the internal CLI tool.\n\n"
        "## Configuration\n"
        "Next, configure your API credentials in the environment file."
    )
    doc = Document(content=doc_text, source="guide.md", file_type="markdown")
    engine = ChunkingEngine(chunk_size=80, chunk_overlap=10)

    chunks = engine.chunk_document(doc, strategy=ChunkingStrategy.RECURSIVE)
    assert len(chunks) >= 2
    assert chunks[0].strategy == "recursive"
    assert chunks[0].source_doc == "guide.md"