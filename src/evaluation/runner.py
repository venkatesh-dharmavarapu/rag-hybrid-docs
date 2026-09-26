import json
from pathlib import Path
from typing import Dict, List
from src.config import settings
from src.ingestion.chunkers import ChunkingStrategy
from src.ingestion.pipeline import IngestionPipeline
from src.generation.pipeline import RAGPipeline
from src.evaluation.metrics import EvaluationMetrics


class BenchmarkRunner:
    """Runs automated evaluations across different chunking configurations."""

    def __init__(self, dataset_path: Path = None):
        self.dataset_path = dataset_path or (settings.BASE_DIR / "src" / "evaluation" / "golden_dataset.json")
        self.metrics = EvaluationMetrics()
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def run_strategy_benchmark(self, strategy: ChunkingStrategy) -> Dict[str, float]:
        print(f"\n==================================================")
        print(f"  Benchmarking Strategy: {strategy.value.upper()}")
        print(f"==================================================")

        # 1. Ingest docs with target strategy
        pipeline = IngestionPipeline()
        # Clean collection for accurate comparison
        try:
            pipeline.chroma_client.delete_collection("internal_docs")
        except Exception:
            pass
        pipeline.collection = pipeline.chroma_client.get_or_create_collection(
            name="internal_docs",
            metadata={"hnsw:space": "cosine"}
        )

        doc_paths = list(settings.DATA_RAW_DIR.glob("*.md"))
        ingest_res = pipeline.run(doc_paths, strategy=strategy)
        print(f"-> Indexed {ingest_res.get('indexed_count', 0)} chunks.")

        # 2. Run Q&A evaluation
        rag = RAGPipeline()
        retrieval_hits: List[float] = []
        correctness_scores: List[float] = []
        citation_scores: List[float] = []

        for item in self.dataset:
            q = item["question"]
            gt = item["ground_truth"]
            exp_src = item["expected_source"]

            res = rag.query(q)

            # Measure metrics
            r_hit = self.metrics.calculate_retrieval_hit(exp_src, res.get("sources", []))
            c_acc = self.metrics.calculate_citation_accuracy(res.get("citations", []))
            corr = self.metrics.evaluate_correctness(q, res.get("answer", ""), gt)

            retrieval_hits.append(r_hit)
            citation_scores.append(c_acc)
            correctness_scores.append(corr)

        avg_hit = sum(retrieval_hits) / len(retrieval_hits)
        avg_corr = sum(correctness_scores) / len(correctness_scores)
        avg_cit = sum(citation_scores) / len(citation_scores)

        return {
            "retrieval_hit_rate": round(avg_hit * 100, 1),
            "correctness": round(avg_corr * 100, 1),
            "citation_accuracy": round(avg_cit * 100, 1),
        }

    def run_all(self) -> Dict[str, Dict[str, float]]:
        strategies = [
            ChunkingStrategy.FIXED,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SEMANTIC,
        ]
        results = {}
        for strat in strategies:
            results[strat.value] = self.run_strategy_benchmark(strat)

        self._print_markdown_table(results)
        return results

    @staticmethod
    def _print_markdown_table(results: Dict[str, Dict[str, float]]):
        print("\n### Chunking Strategy Benchmark Results\n")
        print("| Strategy | Retrieval Hit Rate (%) | Answer Correctness (%) | Citation Accuracy (%) |")
        print("| :--- | :--- | :--- | :--- |")
        for strat, scores in results.items():
            print(f"| **{strat.capitalize()}** | {scores['retrieval_hit_rate']}% | {scores['correctness']}% | {scores['citation_accuracy']}% |")
        print("\n")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()