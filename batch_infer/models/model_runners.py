from __future__ import annotations

from typing import Any, Dict, List

import torch

from batch_infer.config import InferenceConfig
from batch_infer.engine.runners.base_runners import BaseModelRunner
from batch_infer.transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class CausalLMRunner(BaseModelRunner):
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)
        self.model.eval()

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        encodes = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Support both HF BatchEncoding and stub dict
        if hasattr(encodes, "to"):
            encodes = encodes.to(self.config.device)
        else:
            encodes = {k: v.to(self.config.device) for k, v in encodes.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **encodes,
                max_new_tokens=self.config.max_new_tokens,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return [{"input": inp, "output": out} for inp, out in zip(texts, decoded, strict=False)]


class ClassifierRunner(BaseModelRunner):
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name).to(
            config.device
        )
        self.model.eval()

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run sequence classification on a batch of texts.
        Returns a list of dicts:
            {"input": original_text, "label": <string_label>, "score": <float_prob>}
        """
        encodes = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if hasattr(encodes, "to"):
            encodes = encodes.to(self.config.device)
        else:
            encodes = {k: v.to(self.config.device) for k, v in encodes.items()}

        with torch.no_grad():
            outputs = self.model(**encodes)  # type: ignore[arg-type]

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        scores, label_ids = probs.max(dim=-1)

        id2label = getattr(self.model.config, "id2label", None)

        results: List[Dict[str, Any]] = []
        for text, label_id, score in zip(texts, label_ids, scores, strict=False):
            label_idx = int(label_id.item())
            label_name = (
                id2label[label_idx]
                if id2label is not None and label_idx in id2label
                else str(label_idx)
            )

            results.append(
                {
                    "input": text,
                    "label": label_name,
                    "score": float(score.item()),
                }
            )

        return results
