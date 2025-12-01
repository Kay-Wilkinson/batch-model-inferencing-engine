from __future__ import annotations

from typing import Any, Dict, List

import pytest

from batch_infer.config import InferenceConfig, RetryConfig
from batch_infer.engine.runners.base_runners import BaseModelRunner
from batch_infer.engine.runners.core_runners import BatchInferenceRunner


class RecordingRunner(BaseModelRunner):

    def __init__(self) -> None:
        self.calls: int = 0
        self.batches: List[List[str]] = []

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        self.calls += 1
        self.batches.append(list(texts))
        return [{"input": t, "output": t.upper()} for t in texts]


def test_batch_inference_batches_and_aggregates_results() -> None:
    cfg = InferenceConfig(
        model_name="dummy",
        task="generation",
        batch_size=4,
        max_new_tokens=5,
        device="cpu",
        input_column="text",
        retry=RetryConfig(
            max_retries=0, initial_backoff=0.0, backoff_multiplier=1.0, max_backoff=0.0
        ),
    )

    runner = RecordingRunner()
    engine = BatchInferenceRunner(cfg, runner)

    texts = [f"sample {i}" for i in range(10)]
    results = engine.run(texts)

    assert len(results) == len(texts)
    assert runner.calls == 3
    batch_lengths = [len(b) for b in runner.batches]
    assert batch_lengths == [4, 4, 2]


class FlakyRunner(BaseModelRunner):
    """Fake runner that fails first time, then succeeds."""

    def __init__(self) -> None:
        self.calls: int = 0

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError("Transient failure")
        self.calls += 1
        return [{"input": t, "output": t.upper()} for t in texts]


def test_batch_inference_retries_on_failure() -> None:
    cfg = InferenceConfig(
        model_name="dummy",
        task="generation",
        batch_size=2,
        max_new_tokens=5,
        device="cpu",
        input_column="text",
        retry=RetryConfig(
            max_retries=1,
            initial_backoff=0.0,  # no real sleeping during tests
            backoff_multiplier=1.0,
            max_backoff=0.0,
        ),
    )

    runner = FlakyRunner()
    engine = BatchInferenceRunner(cfg, runner)

    texts = ["a", "b"]
    results = engine.run(texts)

    assert len(results) == 2
    assert runner.calls == 2


class AlwaysFailRunner(BaseModelRunner):
    def __init__(self) -> None:
        self.calls: int = 0

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        self.calls += 1
        raise RuntimeError("Always failing")


def test_batch_inference_raises_after_max_retries() -> None:
    cfg = InferenceConfig(
        model_name="dummy",
        task="generation",
        batch_size=2,
        max_new_tokens=5,
        device="cpu",
        input_column="text",
        retry=RetryConfig(
            max_retries=1,
            initial_backoff=0.0,
            backoff_multiplier=1.0,
            max_backoff=0.0,
        ),
    )

    runner = AlwaysFailRunner()
    engine = BatchInferenceRunner(cfg, runner)

    with pytest.raises(RuntimeError):
        list(engine.run(["x", "y"]))

    # 1 initial attempt + 1 retry
    assert runner.calls == 2
