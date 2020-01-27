import torch
import torch.nn as nn

from text_classification.models.base import BaseClassifier


class FastText(BaseClassifier):
    def __init__(self, input_size, embed_mat=None):

        super().__init__()

        self.embedding = nn.EmbeddingBag(input_size, 300, mode="mean")
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )
        self.fc = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-05)
        return optimizer

    def forward(self, batch):

        inputs, _, offsets = batch
        # inputs: [SUM(SEQ_LENGTHS)]

        x = self.embedding(inputs, offsets)
        # x: [BATCH_SIZE, EMBED_DIM]

        x = self.fc(x).squeeze()
        # x: [BATCH_SIZE]

        return x
