import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification.models.base import BaseClassifier


class TextCNN(BaseClassifier):
    def __init__(self, input_size, embed_mat=None):

        super().__init__()
        self.embedding = nn.Embedding(input_size, 300, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=300, out_channels=100, kernel_size=fs)
                for fs in [3, 4, 5]
            ]
        )

        self.fc = nn.Linear(3 * 100, 1)

        self.dropout = nn.Dropout()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-05)
        return optimizer

    def forward(self, batch):

        inputs, _ = batch
        # inputs:  [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embedding(inputs).permute(0, 2, 1)
        # embeds = [BATCH_SIZE, EMBED_DIM, LONGEST_SEQ]

        convs = [F.relu(conv(embeds)) for conv in self.convs]
        # convs: List[[BATCH_SIZE, CONV_OUT_DIM, LONGEST_SEQ - KERNEL_SIZE + 1]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convs]
        # pooled: List[[BATCH_SIZE, CONV_OUT_DIM]]

        conv_out = self.dropout(torch.cat(pooled, dim=1))
        # conv_out: [BATCH_SIZE, N_FILTERS * CONV_OUT_DIM]

        return self.fc(conv_out).squeeze()
