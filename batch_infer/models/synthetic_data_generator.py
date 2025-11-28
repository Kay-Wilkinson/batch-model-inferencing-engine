from __future__ import annotations

from dataclasses import dataclass
from random import choice, randint
from typing import Iterable, List, Tuple


@dataclass
class SyntheticSample:
    text: str
    label: str


class SyntheticDataGenerator:
    """Generate small synthetic datasets for classification experiments.

    Two kinds of tasks:
      1. Sentiment-like phrases ("POSITIVE" / "NEGATIVE")
      2. Pattern tasks (e.g. repeated patterns vs random sequences)
    """

    positive_templates = [
        "I really love this {}.",
        "That {} was absolutely lovely",
        "What an amazing {}.",
        "I would definitely recommend this {}.",
    ]

    negative_templates = [
        "I really hate this {}.",
        "This {} is terrible.",
        "What a disappointing {}",
        "This {} made me feel dead inside.",
    ]

    sentiment_objects = ["movie", "product", "service", "experience"]

    @classmethod
    def generate_sentiment(cls, n: int) -> List[SyntheticSample]:
        samples: List[SyntheticSample] = []
        for _ in range(n):
            if randint(0, 1) == 0:
                template = choice(cls.positive_templates)
                obj = choice(cls.sentiment_objects)
                text = template.format(obj)
                label = "POSITIVE"
            else:
                template = choice(cls.negative_templates)
                obj = choice(cls.sentiment_objects)
                text = template.format(obj)
                label = "NEGATIVE"
            samples.append(SyntheticSample(text=text, label=label))
        return samples

    @staticmethod
    def generate_pattern_task(n: int, max_len: int = 6) -> List[SyntheticSample]:
        """Generate simple pattern vs random sequences.

        Example labels:
          - "REPEATED": sequences like "A B A B A B"
          - "RANDOM":   random token sequences
        """
        tokens = ["A", "B", "C", "D"]
        samples: List[SyntheticSample] = []

        for _ in range(n):
            if randint(0, 1) == 0:
                # repeated pattern
                base = choice(tokens)
                seq_len = randint(3, max_len)
                seq = [base for _ in range(seq_len)]
                text = " ".join(seq)
                label = "REPEATED"
            else:
                seq_len = randint(3, max_len)
                seq = [choice(tokens) for _ in range(seq_len)]
                text = " ".join(seq)
                label = "RANDOM"
            samples.append(SyntheticSample(text=text, label=label))

        return samples


def samples_to_csv_rows(samples: Iterable[SyntheticSample]) -> List[dict]:
    return [{"text": s.text, "label": s.label} for s in samples]
