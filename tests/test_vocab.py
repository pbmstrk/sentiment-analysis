from text_classification.datasets import TextDataset
from text_classification.datasets.base import Example
from text_classification.vocab import Vocab


class TestVocab:
    def test_from_dataset(self):

        dataset = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        vocab = Vocab(dataset)

        assert len(vocab) == 6 + vocab.num_all_special_tokens

    def test_from_list(self):

        lst = ["Test", "the", "vocab", "class"] + ["another", "example"]

        vocab = Vocab(lst)

        assert len(vocab) == 6 + vocab.num_all_special_tokens

    def test_from_multiple_datasets(self):

        dataset1 = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        dataset2 = TextDataset([Example(text=["Second", "dataset"], label=None)])

        vocab = Vocab([dataset1, dataset2])

        assert len(vocab) == 8 + vocab.num_all_special_tokens

    def test_special_tokens(self):

        dataset = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        special_tokens = {"sos_token": "<sos>", "eos_token": "<eos>"}

        vocab = Vocab(dataset, special_tokens=special_tokens)

        assert hasattr(vocab, "sos_token") and hasattr(vocab, "eos_token")
