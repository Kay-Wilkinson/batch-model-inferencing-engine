from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from batch_infer.io import read_texts_from_csv, write_results_to_csv


def test_write_and_read_csv_round_trip(tmp_path: Path) -> None:
    results: List[Dict[str, Any]] = [
        {"input": "hello", "output": "HELLO"},
        {"input": "world", "output": "WORLD"},
    ]

    path = tmp_path / "test.csv"
    write_results_to_csv(str(path), results)

    assert path.exists()
    texts = list(read_texts_from_csv(str(path), col="input"))
    assert texts == ["hello", "world"]
