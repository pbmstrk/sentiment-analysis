import torch
import numpy as np

from sentiment_analysis.data.base import SSTDataModuleBase
from torch.utils.data import DataLoader


class SSTDataModuleMLP(SSTDataModuleBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _collate_fn(batch):
        # get data and targets from batch
        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        lengths = [len(el) for el in data]
        offsets = np.cumsum(lengths)
        offsets = np.concatenate([[0], offsets[:-1]])

        return (
            torch.LongTensor(torch.cat(data).long()),
            torch.Tensor(targets).float(),
            torch.LongTensor(offsets),
        )

    def train_dataloader(self):
        return DataLoader(
            self.sst_train, batch_size=64, collate_fn=self._collate_fn, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.sst_val, batch_size=64, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.sst_test, batch_size=64, collate_fn=self._collate_fn)
