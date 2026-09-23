from dataclasses import dataclass
from typing import Any, Dict, List
import ollama

from src.config import settings


@dataclass
class GeneratedAnswer:
    answer: str
    cited_chunks: List[Dict[str, Any]]
    raw_response: str


class GroundedGenerator:
    """Generates grounded answers from retrieved context chunks using Ollama."""

    def __init__(self, model_name: str = settings.LLM_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def _build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats chunks into numbered context blocks [1], [2], etc."""
        context_lines = []
        for idx, chunk in enumerate(chunks):
            citation_num = idx + 1
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_lines.append(
                f"[{citation_num}] (Source: {source})\n{chunk['content']}\n"
            )
        return "\n".join(context_lines)

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> GeneratedAnswer:
        if not context_chunks:
            return GeneratedAnswer(
                answer="I could not find relevant documentation in the knowledge base to answer this question.",
                cited_chunks=[],
                raw_response=""
            )

        context_text = self._build_context_block(context_chunks)

        system_prompt = (
            "You are an enterprise technical documentation assistant.\n"
            "Strict Guidelines:\n"
            "1. Answer the question using ONLY the facts directly stated in the context blocks below.\n"
            "2. Never assume, extrapolate, or use outside knowledge.\n"
            "3. Every factual sentence or claim MUST end with its bracketed citation number (e.g., [1], [2]).\n"
            "4. If the context does not provide sufficient facts to answer the question completely, "
            "respond exactly with: 'INSUFFICIENT_CONTEXT: The provided documentation does not contain enough information.'"
        )

        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Grounded Answer with inline citations:"
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0}  # Deterministic generation
        )

        answer_text = response["message"]["content"].strip()

        return GeneratedAnswer(
            answer=answer_text,
            cited_chunks=context_chunks,
            raw_response=answer_text
        )