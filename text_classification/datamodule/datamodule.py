import pytorch_lightning as pl
from typing import Optional, Callable
from ..datasets import TextDataset
from ..encoders.encoders import BaseEncoder

from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: TextDataset,
        encoder: Callable,
        val: Optional[TextDataset] = None,
        test: Optional[TextDataset] = None,
        batch_size: Optional[int] = 16,
    ):

        self.train = train
        self.val = val
        self.test = test
        self.encoder = encoder
        self.batch_size = batch_size

        self.attributes = {}
        self.attributes["datamodule"] = {"batch_size": self.batch_size}
        self.attributes["dataset"] = train.attributes
        if isinstance(self.encoder, BaseEncoder):
            if isinstance(getattr(type(self.encoder), "attributes", None), property):
                self.attributes["encoder"] = encoder.attributes

    def setup(self):

        pass

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.encoder,
        )

    def val_dataloader(self) -> DataLoader:

        if not self.val:
            raise ValueError

        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=self.encoder
        )

    def test_dataloader(self) -> DataLoader:

        if not self.test:
            raise ValueError

        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=self.encoder
        )
