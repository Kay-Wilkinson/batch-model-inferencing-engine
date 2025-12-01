from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from batch_infer.config import InferenceConfig, RetryConfig
from batch_infer.engine.runners.base_runners import BaseModelRunner

logger = logging.getLogger(__name__)


class BatchInferenceRunner:
    def __init__(self, config: InferenceConfig, model_runner: BaseModelRunner) -> None:
        self.config = config
        self.model_runner = model_runner

    def run(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        batch: List[str] = []

        total = 0
        for text in tqdm(texts, desc="Running inference"):
            batch.append(text)
            if len(batch) == self.config.batch_size:
                batch_results = self._run_batch(batch)
                results.extend(batch_results)
                total += len(batch)
                batch = []

        if batch:
            batch_results = self._run_batch(batch)
            results.extend(batch_results)
            total += len(batch)

        return results

    def _run_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        retry_cfg: RetryConfig = self.config.retry
        attempt = 0
        backoff = retry_cfg.initial_backoff

        while True:
            try:
                return self.model_runner.run_batch(batch)
            except Exception as exc:
                if attempt >= retry_cfg.max_retries:
                    logger.error(
                        "Batch failed permanently after %d attempts: %s",
                        attempt + 1,
                        exc,
                        exc_info=True,
                    )
                    raise

                logger.warning(
                    "Error during batch inference (attempt %d/%d): %s. Retrying in %.2fs.",
                    attempt + 1,
                    retry_cfg.max_retries + 1,
                    exc,
                    backoff,
                )

                if backoff > 0.0:
                    time.sleep(backoff)

                # exponential increase, capped TODO: Make the cap configurable
                backoff = min(
                    backoff * retry_cfg.backoff_multiplier
                    if backoff > 0.0
                    else retry_cfg.initial_backoff,
                    retry_cfg.max_backoff if retry_cfg.max_backoff > 0.0 else retry_cfg.initial_backoff,
                )
                attempt += 1
