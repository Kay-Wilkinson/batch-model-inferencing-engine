import csv
from typing import Any, Dict, Generator, List


def read_texts_from_csv(path: str, col: str = "text") -> Generator[str, None, None]:
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row[col]


def write_results_to_csv(path: str, results: List[Dict[str, Any]]) -> None:
    # assume all results have same keys for now, change this alter as this makes the current implementation really brittle.
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
