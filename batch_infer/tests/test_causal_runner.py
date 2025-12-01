from __future__ import annotations

from typing import List

from batch_infer.config import InferenceConfig, RetryConfig
from batch_infer.models.model_runners import CausalLMRunner


def test_causal_lm_runner_generates_longer_sequence(small_texts: List[str]) -> None:
    cfg = InferenceConfig(
        model_name="stub-lm",
        task="generation",
        batch_size=2,
        max_new_tokens=5,
        device="cpu",
        input_column="text",
        retry=RetryConfig(),
    )

    runner = CausalLMRunner(cfg)
    results = runner.run_batch(small_texts)

    assert len(results) == len(small_texts)
    for item, orig in zip(results, small_texts, strict=False):
        assert item["input"] == orig
        assert "output" in item
        assert isinstance(item["output"], str)
