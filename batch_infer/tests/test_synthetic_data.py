from __future__ import annotations

from batch_infer.models.synthetic_data_generator import SyntheticDataGenerator, samples_to_csv_rows


def test_generate_sentiment_samples_length_and_labels() -> None:
    n = 50
    samples = SyntheticDataGenerator.generate_sentiment(n)
    assert len(samples) == n

    labels = {s.label for s in samples}
    assert labels.issubset({"POSITIVE", "NEGATIVE"})


def test_generate_pattern_task_labels() -> None:
    n = 50
    samples = SyntheticDataGenerator.generate_pattern_task(n)
    assert len(samples) == n

    labels = {s.label for s in samples}
    assert labels.issubset({"REPEATED", "RANDOM"})


def test_samples_to_csv_rows_shape() -> None:
    samples = SyntheticDataGenerator.generate_sentiment(5)
    rows = samples_to_csv_rows(samples)

    assert len(rows) == 5
    assert all("text" in r and "label" in r for r in rows)
