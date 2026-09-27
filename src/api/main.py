from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn

from src.config import settings
from src.generation.pipeline import RAGPipeline
from src.ingestion.pipeline import IngestionPipeline
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    DocumentListResponse,
)

app = FastAPI(
    title="Enterprise Hybrid RAG API",
    version="1.0.0",
    description="Production-grade RAG engine with hybrid search, reranking, and citation verification.",
)

# Global pipeline instance
rag_service = RAGPipeline()
ingest_service = IngestionPipeline()


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model": settings.LLM_MODEL}


@app.post("/v1/ask", response_model=QueryResponse, tags=["Retrieval & Generation"])
def ask_question(request: QueryRequest):
    try:
        result = rag_service.query(
            question=request.question,
            top_k_retrieval=request.top_k_retrieval,
            top_n_rerank=request.top_n_rerank,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    raw_dir = settings.DATA_RAW_DIR
    if not raw_dir.exists():
        return {"documents": [], "total_count": 0}

    files = [f.name for f in raw_dir.iterdir() if f.is_file()]
    return {"documents": files, "total_count": len(files)}


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    raw_dir = settings.DATA_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / file.filename

    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_service.run([target_path])
        return {
            "status": result.get("status", "success"),
            "filename": file.filename,
            "indexed_count": result.get("indexed_count", 0),
            "message": f"Successfully indexed {file.filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)