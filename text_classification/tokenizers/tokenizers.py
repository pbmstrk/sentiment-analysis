from abc import abstractmethod

from spacy.lang.en import English


class BaseTokenizer:
    @abstractmethod
    def __call__(self, x):
        return

    @abstractmethod
    def __str__(self):
        pass


class SimpleTokenizer(BaseTokenizer):
    def __init__(self, split=" "):
        self.split = split

    def __call__(self, x: str):
        x = x.lower()
        return x.split(self.split)

    def __str__(self):
        return f"SimpleTokenizer(split={self.split})"


class SpacyTokenizer(BaseTokenizer):
    def __init__(self):
        self.nlp = English()

    def __call__(self, x: str):
        tokens = self.nlp(x)
        return [token.lower_ for token in tokens if not token.is_punct]

    def __str__(self):
        return "SpacyTokenizer()"
