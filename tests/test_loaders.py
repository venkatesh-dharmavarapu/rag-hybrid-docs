from pathlib import Path
from src.ingestion.loaders import DocumentLoader


def test_loader_markdown(tmp_path: Path):
    md_file = tmp_path / "sample.md"
    md_file.write_text("# API Reference\n\nInternal auth tokens expire every 15 minutes.")

    loader = DocumentLoader()
    docs = loader.load_file(md_file)

    assert len(docs) == 1
    assert docs[0].source == "sample.md"
    assert "API Reference" in docs[0].content
    assert docs[0].metadata["document_title"] == "API Reference"