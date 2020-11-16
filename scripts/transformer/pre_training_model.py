import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from text_classification.models.transformer import Encoder

WARMUP_STEPS = 1000
MAX_STEPS = 120000
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# from huggingface
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TransformerWithMLMHead(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hid_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        pf_dim: int = 1024,
        dropout: float = 0.1,
        mlp_dim: int = 256,
        max_length: int = 284,
        padding_idx: int = 0
    ):

        self.encoder = Encoder(
            input_size,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            padding_idx,
            max_length,
        )

        self.lm_head = nn.Linear(hid_dim, input_size)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.encoder.tok_embedding.weight

    def forward(self, x):
        hidden_states = self.encoder(x)
        logits = self.lm_head(hidden_states)

        return logits

    def step(self, batch, batch_idx):

        # get predictions
        x, y = batch
        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.size(-1)), y.view(-1), ignore_index=-1
        )

        return {"loss": loss, "batch_size": x.shape[0]}

    def epoch_end(self, outputs, prefix="train"):

        total = sum([x["batch_size"] for x in outputs])
        loss = sum([x["loss"] * x["batch_size"] for x in outputs]) / total

        self.log(prefix + "_epoch_loss", loss)

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

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

        lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
        )
        return [optimizer], [lr_scheduler]
