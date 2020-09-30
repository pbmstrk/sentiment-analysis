import torchtext
import torch

import numpy as np
import pytorch_lightning as pl

from collections import namedtuple, Counter
from torch.utils.data import Dataset


# use a named-tuple to store data examples
Example = namedtuple("Example", ["text", "label"])


class SSTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        element = self.dataset[idx]
        X = element.text
        Y = element.label
        return X, Y


class SSTDataModuleBase(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3

        self.targetEncoding = {"negative": 0, "positive": 1}

        # other attributes are defined are calling setup() function

    def _format_data(self, dataset):

        tokenized_dataset = []
        for element in dataset:
            encoding = self._tokenize(element)
            tokenized_dataset.append(Example(text=encoding[0], label=encoding[1]))

        return tokenized_dataset

    def embedding_matrix(self):

        glove = torchtext.vocab.GloVe(name="6B", dim=300, unk_init=torch.Tensor.normal_)
        matrix_len = len(self._wordlist)
        weights_matrix = np.zeros((matrix_len, 300))

        for i, word in enumerate(self._wordlist):
            try:
                weights_matrix[i] = glove.vectors[glove.stoi[word]]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.5, size=(300,))

        return weights_matrix

    @staticmethod
    def _flatten(lst):
        return [item for sublist in lst for item in sublist]

    def _build_vocab(self, data):
        vocab_counter = Counter(self._flatten([example.text for example in data]))
        return vocab_counter

    def _build_encoding(self, vocab_count, min_freq=3):

        self._wordlist = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.encoding = {}

        svocabCount = {
            k: v
            for k, v in reversed(sorted(vocab_count.items(), key=lambda item: item[1]))
        }

        for word in svocabCount:
            if svocabCount[word] >= min_freq:
                self._wordlist.append(word)
        self.encoding.update({tok: i for i, tok in enumerate(self._wordlist)})

    def _tokenize(self, element):

        text = torch.tensor(
            [self.SOS_token]
            + [self.encoding.get(word, self.UNK_token) for word in element.text]
            + [self.EOS_token]
        )
        label = torch.tensor(self.targetEncoding[element.label])

        return text, label

    def setup(self, stage=None, min_freq=3):

        TEXT = torchtext.data.Field(tokenize="spacy", lower=True)
        LABEL = torchtext.data.Field(sequential=False)

        train_data, val_data, test_data = torchtext.datasets.SST.splits(
            TEXT,
            LABEL,
            filter_pred=lambda ex: ex.label != "neutral",
            train_subtrees=True,
        )

        vocab_counter = self._build_vocab(train_data)
        self._build_encoding(vocab_counter, min_freq)

        if stage == "fit" or stage is None:
            self.sst_train = SSTDataset(self._format_data(train_data))
            self.sst_val = SSTDataset(self._format_data(val_data))

        if stage == "test" or stage is None:
            self.sst_test = SSTDataset(self._format_data(test_data))

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
