from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

"""
Minimal mock/stub implementation of HuggingFace-like transformer classes.

This allows:
- Unit tests to run without downloading real models
- Mypy to type-check model interactions
- The batch inference engine to run synthetic / offline tests

These classes mimic the HF interface _just_ enough to support:
- AutoTokenizer
- AutoModelForCausalLM
- AutoModelForSequenceClassification
"""


class AutoTokenizer:
    """
    Maps tokens <-> IDs using tiny generated vocab.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
            for i in range(100):
                vocab[f"tok{i}"] = 3 + i

        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        self.pad_token_id = 0

    @classmethod
    def from_pretrained(cls, model_name: str) -> "AutoTokenizer":
        return cls()

    def __call__(
        self,
        texts: List[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        input_ids = []
        for text in texts:
            toks = text.split()
            ids = [self.vocab.get(tok, 3 + (hash(tok) % 97)) for tok in toks]
            input_ids.append(ids)

        # Pad
        max_len = max(len(x) for x in input_ids)
        padded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in input_ids]

        tensor = torch.tensor(padded, dtype=torch.long)

        return {"input_ids": tensor}

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = True) -> List:
        decoded = []
        for seq in sequences:
            toks = []
            for tok_id in seq.tolist():
                tok = self.inv_vocab.get(tok_id, f"tok{tok_id}")
                if skip_special_tokens and tok in ("<pad>", "<bos>", "<eos>"):
                    continue
                toks.append(tok)
            decoded.append(" ".join(toks))
        return decoded


class AutoModelForCausalLM(nn.Module):
    """Stub causal LM that returns deterministic dummy 'generated' outputs."""

    def __init__(self) -> None:
        super().__init__()
        # Fake embedding + projection, purely for shape
        self.embed = nn.Embedding(200, 16)
        self.lm_head = nn.Linear(16, 200)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "AutoModelForCausalLM":
        return cls()

    def forward(self, input_ids: torch.Tensor) -> nn.utils.rnn.PackedSequence:
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return nn.utils.rnn.PackedSequence(logits)  # unused, but placeholder

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        **_: Any,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        # Create deterministic fake generation: just repeat the last token
        new_tokens = input_ids[:, -1:].repeat(1, max_new_tokens)
        return torch.cat([input_ids, new_tokens], dim=1)


@dataclass
class SequenceClassifierOutput:
    logits: torch.Tensor


class AutoModelForSequenceClassification(nn.Module):
    """
    Produces logits over 2 labels (POSITIVE / NEGATIVE) by default.
    """

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.embed = nn.Embedding(200, 16)
        self.classifier = nn.Linear(16, num_labels)
        self.config = type("Config", (), {})()
        self.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    @classmethod
    def from_pretrained(cls, model_name: str) -> "AutoModelForSequenceClassification":
        return cls()

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SequenceClassifierOutput:
        emb = self.embed(input_ids).mean(dim=1)
        logits = self.classifier(emb)
        return SequenceClassifierOutput(logits=logits)
