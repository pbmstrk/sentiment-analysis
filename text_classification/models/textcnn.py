from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_class: int = 2,
        embed_dim: int = 300,
        kernel_sizes: List[int] = [3, 4, 5],
        out_channels: int = 100,
        dropout: float = 0.5,
        embed_mat: Optional[np.ndarray] = None,
        freeze_embed: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_class = num_class
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.dropout = dropout
        self.embed_mat = embed_mat
        self.freeze_embed = freeze_embed

        self.embedding = nn.Embedding(self.input_size, self.embed_dim, padding_idx=0)
        if self.embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(self.embed_mat).float()
            )
        if self.freeze_embed:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.out_channels,
                    kernel_size=fs,
                )
                for fs in self.kernel_sizes
            ]
        )

        self.fc = nn.Linear(len(self.kernel_sizes) * self.out_channels, self.num_class)

        self.drop = nn.Dropout()

    def forward(self, batch):

        inputs, _ = batch
        # inputs:  [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embedding(inputs).permute(0, 2, 1)
        # embeds = [BATCH_SIZE, EMBED_DIM, LONGEST_SEQ]

        convs = [F.relu(conv(embeds)) for conv in self.convs]
        # convs: List[[BATCH_SIZE, CONV_OUT_DIM, LONGEST_SEQ - KERNEL_SIZE + 1]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convs]
        # pooled: List[[BATCH_SIZE, CONV_OUT_DIM]]

        conv_out = self.drop(torch.cat(pooled, dim=1))
        # conv_out: [BATCH_SIZE, N_FILTERS * CONV_OUT_DIM]

        return self.fc(conv_out)
