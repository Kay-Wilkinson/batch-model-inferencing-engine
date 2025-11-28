from __future__ import annotations

from typing import List

import pytest


@pytest.fixture
def small_texts() -> List[str]:
    return ["hello world", "foo bar", "lorem ipsum", "another sample"]
