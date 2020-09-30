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

        x, y, offsets = batch
        y_hat = self(batch)

        # compute loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # compute acc
        preds = torch.round(torch.sigmoid(y_hat))
        correct = (preds == y).float().sum()
        acc = correct / len(y)

        return {"loss": loss, "acc": acc, "batch_size": len(y)}

    def training_epoch_end(self, outputs):

        total = sum([x["batch_size"] for x in outputs])
        avg_loss = sum([x["loss"] * x["batch_size"] for x in outputs]) / total
        avg_acc = sum([x["acc"] * x["batch_size"] for x in outputs]) / total

        return {"epoch_train_loss": avg_loss, "epoch_train_acc": avg_acc}

    def validation_step(self, batch, batch_idx):

        x, y, offsets = batch
        y_hat = self(batch)

        # compute loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # compute acc
        preds = torch.round(torch.sigmoid(y_hat))
        correct = (preds == y).float().sum()
        acc = correct / len(y)

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
