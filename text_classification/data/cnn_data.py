import torch
import numpy as np

from text_classification.data.base import SSTDataModuleBase
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SSTDataModuleCNN(SSTDataModuleBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _collate_fn(batch):
        # get inputs and targets
        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # pad the sequences
        x = pad_sequence(data, batch_first=True)

        return x, torch.Tensor(targets).float()

    def train_dataloader(self):
        return DataLoader(
            self.sst_train, batch_size=64, collate_fn=self._collate_fn, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.sst_val, batch_size=64, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.sst_test, batch_size=64, collate_fn=self._collate_fn)
