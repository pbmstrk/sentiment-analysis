from inspect import Parameter, signature
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str = "Adam",
        optimizer_args: Dict = {"lr": 0.001},
        scheduler_name: Optional[str] = None,
        scheduler_args: Optional[Dict] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.check_optimizer_args()

        if scheduler_name:
            self.scheduler_name = scheduler_name
            self.scheduler_args = scheduler_args
            self.check_scheduler_args()

    @staticmethod
    def try_get_func(module, name):
        try:
            func = getattr(module, name)
        except AttributeError:
            print(f"{name} is not a valid name in the {module} module.")
            raise
        return func

    def check_optimizer_args(self):
        self.try_get_func(torch.optim, self.optimizer_name)

    def check_scheduler_args(self):
        func = self.try_get_func(torch.optim.lr_scheduler, self.scheduler_name)

        for name, p in signature(func).parameters.items():
            if name != "optimizer" and p.default == Parameter.empty:
                assert (
                    name in self.scheduler_args.keys()
                ), f"{self.scheduler_name} expects a value for {name}"

    def forward(self, batch):
        return self.model(batch)

    def step(self, batch, batch_idx):

        # get predictions
        y = batch[1]
        y_hat = self(batch)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute acc
        _, pred = torch.max(y_hat.data, 1)
        correct = (pred == y).sum()
        acc = correct.float() / len(y)

        return {"loss": loss, "acc": acc, "batch_size": len(y)}

    def epoch_end(self, outputs, prefix="train"):

        total = sum([x["batch_size"] for x in outputs])
        loss = sum([x["loss"] * x["batch_size"] for x in outputs]) / total
        acc = sum([x["acc"] * x["batch_size"] for x in outputs]) / total

        self.log(prefix + "_epoch_loss", loss)
        self.log(prefix + "_epoch_acc", acc)

    def training_step(self, batch, batch_idx):

        return self.step(batch, batch_idx)

    def training_epoch_end(self, outputs):

        self.epoch_end(outputs, prefix="train")

    def validation_step(self, batch, batch_idx):

        return self.step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        self.epoch_end(outputs, prefix="val")

    def test_step(self, batch, batch_idx):

        return self.step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        self.epoch_end(outputs, prefix="test")

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), **self.optimizer_args
        )

        if hasattr(self, "scheduler_name"):
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(
                optimizer, **self.scheduler_args
            )
            return [optimizer], [scheduler]

        return optimizer
