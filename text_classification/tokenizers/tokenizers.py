import re
from abc import abstractmethod
from typing import List

from spacy.lang.en import English


class BaseTokenizer:
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class WhiteSpaceTokenizer(BaseTokenizer):
    def __init__(self, split=" ") -> None:
        self.split = split

    def __call__(self, x: str) -> List[str]:
        x = x.lower()
        return x.split(self.split)


class SpacyTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.nlp = English()

    def __call__(self, x: str) -> List[str]:
        tokens = self.nlp(x)
        return [token.lower_ for token in tokens]


class TokenizerSST(BaseTokenizer):
    def __init__(self, split=" ") -> None:
        self.split = split

    def __call__(self, x: str) -> List[str]:
        x = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", x)
        x = re.sub(r"\s{2,}", " ", x)
        return x.strip().lower().split(self.split)
