from typing import Union, List, Dict, Optional
from abc import abstractmethod

from text_classification.datasets import TextDataset

from collections import Counter, OrderedDict

def flatten(lst) -> List:
    return [item for sublist in lst for item in sublist]

class BaseEncoder:
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

class VocabMixin:

    def add_vocab(
        self,
        data: Union[List[str], TextDataset, List[TextDataset]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<unk>",
        special_tokens: Optional[Dict[str, str]] = None,
    ):

        vocab_count = self._process_dataset(data)
        wordlist = []

        if pad_token:
            wordlist.append(pad_token)
            self.pad_token_index = wordlist.index(pad_token)
        if unk_token:
            wordlist.append(unk_token)
            self.unk_token_index = wordlist.index(unk_token)

        if special_tokens:
            for key, value in special_tokens.items():
                wordlist.append(value)
                setattr(self, f"{key}_index", wordlist.index(value))

        self.encoding = OrderedDict()

        self._build_vocab(vocab_count, wordlist, min_freq, max_size)

        return self

    @property
    def num_tokens(self):
        return len(self.encoding)
        
    def convert_token(self, token):
        assert hasattr(self, "encoding")

        if hasattr(self, "unk_token_index"):
            return self.encoding.get(token, self.unk_token_index)
        try:
            return self.encoding[token]
        except KeyError:
            print("No UNK token, cannot process unknown tokens")
            raise

    def convert_token_to_ids(self, tokens):

        return list(map(self.convert_token, tokens))

    @staticmethod
    def _process_dataset(data: Union[TextDataset, List[str], List[TextDataset]]
    ) -> Counter:
        if isinstance(data, TextDataset):
            list_of_vocab = flatten([example[0] for example in data])
        elif all(isinstance(element, TextDataset) for element in data):
            list_of_vocab = []
            for dataset in data:
                list_of_vocab.extend(flatten([example[0] for example in dataset]))
        elif all(isinstance(element, str) for element in data):
            list_of_vocab = data
        vocab_count = Counter(list_of_vocab)
        return vocab_count

    def _build_vocab(self, vocab_count, wordlist, min_freq, max_size):

        sorted_vocab_count = {
            k: v
            for k, v in reversed(sorted(vocab_count.items(), key=lambda item: item[1]))
        }

        for word in sorted_vocab_count:
            if sorted_vocab_count[word] >= min_freq:
                wordlist.append(word)
            if max_size and len(wordlist) == max_size:
                break

        self.encoding.update({tok: i for i, tok in enumerate(wordlist)})

class TargetEncodingMixin:

    def add_target_encoding(self, target_encoding):
        self.target_encoding = target_encoding

    def convert_targets(self, targets):
        if not hasattr(self, "target_encoding"):
            return targets
        return list(map(self.target_encoding.get, targets))
