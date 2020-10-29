from abc import abstractmethod
from typing import List

from spacy.lang.en import English


class BaseTokenizer:
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class WhiteSpaceTokenizer(BaseTokenizer):
    def __init__(self, split=" "):
        self.split = split

    def __call__(self, x: str) -> List:
        x = x.lower()
        return x.split(self.split)


class SpacyTokenizer(BaseTokenizer):
    def __init__(self):
        self.nlp = English()

    def __call__(self, x: str) -> List:
        tokens = self.nlp(x)
        return [token.lower_ for token in tokens]
