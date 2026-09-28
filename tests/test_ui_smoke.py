from pathlib import Path


def test_ui_file_exists():
    ui_path = Path("src/ui/app.py")
    assert ui_path.exists()
    content = ui_path.read_text(encoding="utf-8")
    assert "Streamlit" in content or "streamlit" in content
    assert "Hybrid Search" in content