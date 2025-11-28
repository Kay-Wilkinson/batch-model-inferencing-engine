import time
from typing import Iterable, List, Dict, Any

from tqdm import tqdm

from batch_infer.engine.runners.base_runners import BaseModelRunner
from batch_infer.config import InferenceConfig


class BatchInferenceRunner:
    """
    Iterates over input data, build batches, call model_runner.run_batch(batch), accumulate results, track timing and throughput
    This is the core "engine" of the POC
    """
    def __init__(self, cfg: InferenceConfig, model_runner: BaseModelRunner):
        self.cfg = cfg
        self.model_runner = model_runner

    def run(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        batch: List[str] = []

        start_time = time.time()
        total = 0

        for text in tqdm(texts, desc="Running inference"):
            batch.append(text)
            if len(batch) == self.cfg.batch_size:
                batch_results = self._run_batch(batch)
                results.extend(batch_results)
                total += len(batch)
                batch = []

        if batch:
            batch_results = self._run_batch(batch)
            results.extend(batch_results)
            total += len(batch)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Processed {total} samples in {elapsed:.2f}s ({total/elapsed:.2f} samples/s)")

        return results

    def _run_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        return self.model_runner.run_batch(batch)
