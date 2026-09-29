from pathlib import Path


def test_demo_script_exists():
    demo_file = Path("demo.py")
    assert demo_file.exists()
    content = demo_file.read_text(encoding="utf-8")
    assert "run_demo" in content
    assert "ERR_RATE_LIMIT" in content