import pytest
import torch
from torch.utils.data import TensorDataset

from text_classification.datamodule import DataModule


class TestDataModule:
    def test_datamodule(self):

        inputs = torch.randn((32, 10))
        targets = torch.randn((32,))

        data = TensorDataset(inputs, targets)

        # access train loader
        dm = DataModule(data)
        dm.train_dataloader()

        with pytest.raises(ValueError):
            dm.val_dataloader()
        with pytest.raises(ValueError):
            dm.test_dataloader()

        # access all
        dm = DataModule(train=data, val=data, test=data)

        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
