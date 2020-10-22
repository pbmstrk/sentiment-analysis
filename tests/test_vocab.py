from text_classification.datasets.base import Example
from text_classification.datasets import TextDataset
from text_classification.vocab import Vocab

import pytest


class TestVocab:

    def test_from_dataset(self):

        exs = [
            Example(text=["Test", "the", "vocab", "class"], label=None),
            Example(text=["another", "example"], label=None)
        ]

        dataset = TextDataset(exs)

        vocab = Vocab(dataset)

        assert len(vocab) == 6 + vocab.num_all_special_tokens

    def test_from_list(self):

        lst = ["Test", "the", "vocab", "class"] + ["another", "example"]

        vocab = Vocab(lst)

        assert len(vocab) == 6 + vocab.num_all_special_tokens
        
    def test_error(self):

        exs = [
            Example(text="Test the vocab class", label=None)
        ]

        dataset = TextDataset(exs)

        with pytest.raises(ValueError):
            Vocab(dataset)

