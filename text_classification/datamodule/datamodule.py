from typing import Callable, Optional

from torch.utils.data import DataLoader

from ..datasets import TextDataset


class DataModule:
    def __init__(
        self,
        train: TextDataset,
        collate_fn: Optional[Callable] = None,
        val: Optional[TextDataset] = None,
        test: Optional[TextDataset] = None,
        batch_size: int = 16,
    ):

        self.train = train
        self.val = val
        self.test = test
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:

        if not self.val:
            raise ValueError

        return DataLoader(
            self.val, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:

        if not self.test:
            raise ValueError

        return DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
