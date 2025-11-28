from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseModelRunner:
    @abstractmethod
    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        A BaseModelRunner that takes a list of strings as input and outputs a list of result Dicts
        :param texts: input of original text
        :return: "output" generated text or label/probabilities
        Additional keys are fine e.g. tokens, scores, etc. as the number of integrated models expands
        """
        raise NotImplementedError
