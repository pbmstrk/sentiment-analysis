import torch

import pytorch_lightning as pl
import torch.nn.functional as F


class BaseClassifier(pl.LightningModule):
    def __init__(self):

        super().__init__()

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):

        y = batch[1]
        y_hat = self(batch)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute acc
        _, pred = torch.max(y_hat.data, 1)
        correct = (pred == y).sum()
        acc = correct.float() / len(y)

        return {"loss": loss, "acc": acc, "batch_size": len(y)}

    def training_epoch_end(self, outputs):

        total = sum([x["batch_size"] for x in outputs])
        avg_loss = sum([x["loss"] * x["batch_size"] for x in outputs]) / total
        avg_acc = sum([x["acc"] * x["batch_size"] for x in outputs]) / total

        return {"epoch_train_loss": avg_loss, "epoch_train_acc": avg_acc}

    def validation_step(self, batch, batch_idx):

        y = batch[1]
        y_hat = self(batch)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute acc
        _, pred = torch.max(y_hat.data, 1)
        correct = (pred == y).sum()
        acc = correct.float() / len(y)

        return {"val_loss": loss, "val_acc": acc, "batch_size": len(y)}

    def validation_epoch_end(self, outputs):

        total = sum([x["batch_size"] for x in outputs])
        avg_loss = sum([x["val_loss"] * x["batch_size"] for x in outputs]) / total
        avg_acc = sum([x["val_acc"] * x["batch_size"] for x in outputs]) / total

        return {"epoch_val_loss": avg_loss, "epoch_val_acc": avg_acc}

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        outputs = self.validation_epoch_end(outputs)
        return {
            "test_loss": outputs["epoch_val_loss"],
            "test_acc": outputs["epoch_val_acc"],
        }
