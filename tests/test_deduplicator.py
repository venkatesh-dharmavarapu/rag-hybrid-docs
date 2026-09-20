from src.ingestion.chunkers import TextChunk
from src.ingestion.deduplicator import ChunkDeduplicator


def test_deduplicator():
    c1 = TextChunk("First unique chunk", "doc1.md", 0, "fixed", 18, {})
    c2 = TextChunk("Duplicate of first", "doc2.md", 0, "fixed", 18, {})
    c3 = TextChunk("Different content", "doc3.md", 0, "fixed", 17, {})

    # Mock vectors: vec1 and vec2 have ~0.99 similarity, vec3 is perpendicular
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.99, 0.05, 0.0]
    vec3 = [0.0, 1.0, 0.0]

    dedup = ChunkDeduplicator(threshold=0.95)
    unique_chunks, _, skipped = dedup.filter_duplicates([c1, c2, c3], [vec1, vec2, vec3])

    assert len(unique_chunks) == 2
    assert skipped == 1
    assert unique_chunks[0].source_doc == "doc1.md"
    assert unique_chunks[1].source_doc == "doc3.md"