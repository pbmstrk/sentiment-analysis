import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class BaseClassifier(pl.LightningModule):
    def __init__(self):

        super().__init__()

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

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
    