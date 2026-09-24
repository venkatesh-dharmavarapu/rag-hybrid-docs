from typing import Any, Dict
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import DocumentReranker
from src.generation.generator import GroundedGenerator
from src.generation.verifier import CitationVerifier
from src.generation.scorer import ConfidenceScorer


class RAGPipeline:
    """Unified RAG pipeline combining hybrid retrieval, reranking, generation, and verification."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever = None,
        reranker: DocumentReranker = None,
        generator: GroundedGenerator = None,
        verifier: CitationVerifier = None,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or DocumentReranker()
        self.generator = generator or GroundedGenerator()
        self.verifier = verifier or CitationVerifier()
        self.scorer = ConfidenceScorer()

    def query(self, question: str, top_k_retrieval: int = 15, top_n_rerank: int = 4) -> Dict[str, Any]:
        # 1. Hybrid Retrieval (Dense + BM25)
        candidates = self.retriever.search(question, top_k=top_k_retrieval)

        if not candidates:
            return {
                "question": question,
                "answer": "No relevant documentation found in internal corpus.",
                "citations": [],
                "scores": {"composite_score": 0.0},
                "sources": []
            }

        # 2. Cross-Encoder Reranking
        top_chunks = self.reranker.rerank(question, candidates, top_n=top_n_rerank)

        # 3. Grounded Generation with Citations
        gen_result = self.generator.generate(question, top_chunks)

        # 4. Verify Citations
        citation_checks = self.verifier.verify_answer(gen_result.answer, top_chunks)

        # 5. Score Confidence
        scores = self.scorer.calculate_score(gen_result.answer, top_chunks, citation_checks)

        return {
            "question": question,
            "answer": gen_result.answer,
            "citations": [
                {
                    "claim": c.claim,
                    "citation_id": c.citation_id,
                    "is_supported": c.is_supported,
                    "reason": c.reason
                }
                for c in citation_checks
            ],
            "confidence": scores,
            "sources": [
                {
                    "citation_id": idx + 1,
                    "source": chunk.get("metadata", {}).get("source", "unknown"),
                    "content": chunk["content"]
                }
                for idx, chunk in enumerate(top_chunks)
            ]
        }