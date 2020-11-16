from text_classification.datasets import TextDataset
from text_classification.datasets.base import Example
from text_classification.encoders.base import VocabMixin


class TestVocabMixin:

    def test_from_dataset(self):

        dataset = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        vocab = VocabMixin().add_vocab(dataset)

        assert vocab.num_tokens == 6 + 2

    def test_from_list(self):

        lst = ["Test", "the", "vocab", "class"] + ["another", "example"]

        vocab = VocabMixin().add_vocab(lst, pad_token=None, unk_token=None)

        assert vocab.num_tokens == 6

    def test_from_multiple_datasets(self):

        dataset1 = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        dataset2 = TextDataset([Example(text=["Second", "dataset"], label=None)])

        vocab = VocabMixin().add_vocab([dataset1, dataset2])

        assert vocab.num_tokens == 8 + 2 # number of special tokens

    def test_special_tokens(self):

        dataset = TextDataset(
            [
                Example(text=["Test", "the", "vocab", "class"], label=None),
                Example(text=["another", "example"], label=None),
            ]
        )

        special_tokens = {"sos_token": "<sos>", "eos_token": "<eos>"}

        vocab = VocabMixin().add_vocab(dataset, special_tokens=special_tokens)

        assert hasattr(vocab, "sos_token_index") and hasattr(vocab, "eos_token_index")
