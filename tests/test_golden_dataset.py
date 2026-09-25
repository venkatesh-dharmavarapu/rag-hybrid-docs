import json
from pathlib import Path


def test_golden_dataset_structure():
    dataset_path = Path("src/evaluation/golden_dataset.json")
    assert dataset_path.exists(), "Dataset file must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 5

    required_keys = {"id", "question", "ground_truth", "expected_source", "category"}
    for item in data:
        assert required_keys.issubset(item.keys())