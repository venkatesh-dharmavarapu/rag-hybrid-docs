from pathlib import Path


def test_docker_files_exist():
    dockerfile = Path("Dockerfile")
    compose = Path("docker-compose.yml")

    assert dockerfile.exists(), "Dockerfile must exist"
    assert compose.exists(), "docker-compose.yml must exist"

    df_content = dockerfile.read_text(encoding="utf-8")
    assert "WORKDIR /app" in df_content

    dc_content = compose.read_text(encoding="utf-8")
    assert "rag_api" in dc_content
    assert "rag_dashboard" in dc_content