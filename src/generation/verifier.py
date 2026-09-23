import re
from dataclasses import dataclass
from typing import Any, Dict, List
import ollama

from src.config import settings


@dataclass
class CitationCheck:
    claim: str
    citation_id: int
    is_supported: bool
    reason: str


class CitationVerifier:
    """Verifies that inline citations actually support their corresponding claims."""

    def __init__(self, model_name: str = settings.LLM_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def extract_claims_with_citations(self, text: str) -> List[Dict[str, Any]]:
        """Parses sentences ending with [n] citations."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pairs = []

        for sentence in sentences:
            citations = re.findall(r"\[(\d+)\]", sentence)
            if citations:
                clean_claim = re.sub(r"\[\d+\]", "", sentence).strip()
                for c_id in citations:
                    pairs.append({
                        "claim": clean_claim,
                        "citation_index": int(c_id)
                    })
        return pairs

    def verify_citation(self, claim: str, source_text: str) -> bool:
        """Prompts LLM-as-judge to verify factual alignment between source and claim."""
        prompt = (
            "You are a factual verification judge. Determine if the claim is substantiated by the source context.\n"
            f"Source: \"{source_text}\"\n"
            f"Claim: \"{claim}\"\n\n"
            "Criteria:\n"
            "- If the key factual information in the claim is supported or directly derived from the source, answer YES.\n"
            "- If the claim introduces fabricated facts, different numbers, or contradicts the source, answer NO.\n\n"
            "Answer with exactly one word: YES or NO."
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )

        reply = response["message"]["content"].strip().upper()
        # Look for YES at the beginning or as a standalone word
        return bool(re.search(r"\bYES\b", reply))

    def verify_answer(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> List[CitationCheck]:
        pairs = self.extract_claims_with_citations(answer)
        checks: List[CitationCheck] = []

        for pair in pairs:
            c_idx = pair["citation_index"]
            claim = pair["claim"]

            if 1 <= c_idx <= len(context_chunks):
                chunk_content = context_chunks[c_idx - 1]["content"]
                is_supported = self.verify_citation(claim, chunk_content)
                checks.append(
                    CitationCheck(
                        claim=claim,
                        citation_id=c_idx,
                        is_supported=is_supported,
                        reason="Verified against source chunk" if is_supported else "Ungrounded claim"
                    )
                )
            else:
                checks.append(
                    CitationCheck(
                        claim=claim,
                        citation_id=c_idx,
                        is_supported=False,
                        reason=f"Citation [{c_idx}] is out of bounds"
                    )
                )

        return checks