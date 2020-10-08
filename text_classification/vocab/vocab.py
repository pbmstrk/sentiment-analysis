from collections import Counter
from typing import List, Optional, Union

from ..datasets.base import TextDataset


class Vocab:

    r"""
    Vocabulary class to build vocab from Dataset instance.

    Args:
        data: Dataset instance from which to construct vocab, or desired vocab_list.
            If list is passed, the arguments min_freq and max_size are ignored.
        min_freq: Minimum frequency of a token to be included in vocabularly.
        max_size: Maximum size of vocabularly. Tokens are added in order of frequency.
        special_tokens: List of special tokens, added to vocabularly. Useful if creating
            encodings later.

    Example::

        # using MRDataset() as an example
        >>> RNFModel = RNF(data=MRDataset(), min_freq=3)
    """


    def __init__(
        self,
        data: Union[List, TextDataset],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        special_tokens: List[str] = ["<PAD>", "<UNK>"],
    ):

        self.vocab_count = self.process_dataset(data)
        self.min_freq = min_freq
        self.max_size = max_size
        self.special_tokens = special_tokens

        self.wordlist = []
        self.wordlist.extend(self.special_tokens)

        self.encoding = {}

        # actually build encoding here
        self.build_vocab(
            self.vocab_count,
            min_freq = self.min_freq if isinstance(data, TextDataset) else 1, 
            max_size = self.max_size if isinstance(data, TextDataset) else None
        )

    @property
    def attributes(self):
        return {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "special_tokens": self.special_tokens,
            "size": len(self.encoding),
        }

    def __len__(self):
        return len(self.encoding)

    def __getitem__(self, word):
        self.encoding.get(word, self.special_tokens.index("<UNK>"))

    @staticmethod
    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    @classmethod
    def process_dataset(cls, data):
        list_of_vocab = cls.flatten([example[0] for example in data])
        vocab_count = Counter(list_of_vocab)
        return vocab_count

    def __iter__(self):
        return iter(self.encoding)

    def build_vocab(self, vocab_count, min_freq, max_size):
        # sort vocab s.t. words that occur most frequently added first
        sorted_vocab_count = {
            k: v
            for k, v in reversed(sorted(vocab_count.items(), key=lambda item: item[1]))
        }

        for word in sorted_vocab_count:
            if sorted_vocab_count[word] >= min_freq:
                self.wordlist.append(word)
            if max_size and len(self.wordlist) == max_size:
                break

        self.encoding.update({tok: i for i, tok in enumerate(self.wordlist)})
