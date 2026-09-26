from src.evaluation.metrics import EvaluationMetrics


def test_retrieval_hit_metric():
    sources = [{"source": "sample_api.md"}, {"source": "database_ops.md"}]
    hit = EvaluationMetrics.calculate_retrieval_hit("sample_api.md", sources)
    miss = EvaluationMetrics.calculate_retrieval_hit("missing.md", sources)

    assert hit == 1.0
    assert miss == 0.0


def test_citation_accuracy_metric():
    citations = [
        {"claim": "Tokens expire in 15 min", "is_supported": True},
        {"claim": "Gym is free", "is_supported": False},
    ]
    acc = EvaluationMetrics.calculate_citation_accuracy(citations)
    assert acc == 0.5


def test_correctness_llm_judge():
    evaluator = EvaluationMetrics()
    score = evaluator.evaluate_correctness(
        question="How long do tokens last?",
        generated_answer="Tokens expire after 15 minutes of inactivity [1].",
        ground_truth="Tokens expire after 15 minutes of inactivity."
    )
    assert score == 1.0