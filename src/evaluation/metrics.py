import re
from typing import Any, Dict, List
import ollama

from src.config import settings


class EvaluationMetrics:
    """Computes RAG evaluation metrics using deterministic checks and LLM-as-judge."""

    def __init__(self, model_name: str = settings.LLM_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    @staticmethod
    def calculate_retrieval_hit(expected_source: str, retrieved_sources: List[Dict[str, Any]]) -> float:
        """Returns 1.0 if expected_source is among retrieved sources, 0.0 otherwise."""
        if expected_source == "none":
            return 1.0
        for s in retrieved_sources:
            if expected_source in s.get("source", ""):
                return 1.0
        return 0.0

    @staticmethod
    def calculate_citation_accuracy(citations: List[Dict[str, Any]]) -> float:
        """Percentage of generated citations verified as supported."""
        if not citations:
            return 1.0
        supported = sum(1 for c in citations if c.get("is_supported", False))
        return round(supported / len(citations), 2)

    def evaluate_correctness(self, question: str, generated_answer: str, ground_truth: str) -> float:
        """LLM-as-judge evaluating whether generated answer aligns with ground truth (0.0 or 1.0)."""
        if "INSUFFICIENT_CONTEXT" in ground_truth:
            if "INSUFFICIENT_CONTEXT" in generated_answer or "not contain" in generated_answer.lower():
                return 1.0
            return 0.0

        prompt = (
            "You are an impartial evaluator.\n"
            f"Question: {question}\n"
            f"Ground Truth Answer: {ground_truth}\n"
            f"Student Answer: {generated_answer}\n\n"
            "Does the student answer convey the essential facts stated in the ground truth?\n"
            "Reply with exactly one word: YES or NO."
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        verdict = response["message"]["content"].strip().upper()
        return 1.0 if bool(re.search(r"\bYES\b", verdict)) else 0.0