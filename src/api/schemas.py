from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the token expiration policy?")
    top_k_retrieval: Optional[int] = Field(15, ge=1, le=50)
    top_n_rerank: Optional[int] = Field(4, ge=1, le=10)


class CitationDetail(BaseModel):
    claim: str
    citation_id: int
    is_supported: bool
    reason: str


class SourceItem(BaseModel):
    citation_id: int
    source: str
    content: str


class ConfidenceDetail(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    composite_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationDetail]
    confidence: ConfidenceDetail
    sources: List[SourceItem]


class IngestResponse(BaseModel):
    status: str
    filename: str
    indexed_count: int
    message: str


class DocumentListResponse(BaseModel):
    documents: List[str]
    total_count: int