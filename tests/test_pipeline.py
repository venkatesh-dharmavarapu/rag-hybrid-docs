from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline


def test_ingestion_pipeline():
    sample_file = Path("data/raw/sample_api.md")
    assert sample_file.exists(), "Sample file must exist"

    pipeline = IngestionPipeline()
    result = pipeline.run([sample_file])

    assert result["status"] == "success"
    assert result["indexed_count"] > 0
    assert Path(result["bm25_saved_to"]).exists()