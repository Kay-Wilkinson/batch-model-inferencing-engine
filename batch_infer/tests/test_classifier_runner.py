from __future__ import annotations

from typing import List

from batch_infer.config import InferenceConfig, RetryConfig
from batch_infer.models.model_runners import ClassifierRunner


def test_classifier_runner_basic_inference_cpu(small_texts: List[str]) -> None:
    cfg = InferenceConfig(
        model_name="stub-classifier",
        task="classification",
        batch_size=2,
        max_new_tokens=0,
        device="cpu",
        input_column="text",
        retry=RetryConfig(),
    )

    runner = ClassifierRunner(cfg)

    results = runner.run_batch(small_texts)
    assert len(results) == len(small_texts)

    for item, orig in zip(results, small_texts, strict=False):
        assert item["input"] == orig
        assert "label" in item
        assert "score" in item
        assert isinstance(item["label"], str)
        assert isinstance(item["score"], float)
        assert item["label"] in {"NEGATIVE", "POSITIVE"}
        assert 0.0 <= item["score"] <= 1.0
