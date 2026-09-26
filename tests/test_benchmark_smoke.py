from src.evaluation.runner import BenchmarkRunner
from src.ingestion.chunkers import ChunkingStrategy


def test_benchmark_single_strategy_smoke():
    runner = BenchmarkRunner()
    # Test one strategy execution
    scores = runner.run_strategy_benchmark(ChunkingStrategy.RECURSIVE)

    assert "retrieval_hit_rate" in scores
    assert "correctness" in scores
    assert "citation_accuracy" in scores