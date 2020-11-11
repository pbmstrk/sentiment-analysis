from typing import Optional, Tuple

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
        kernel_sizes: Tuple[int] = (3, 4, 5),
        out_channels: int = 100,
        dropout: float = 0.5,
        embed_dropout: float = 0.4,
        embed_mat: Optional[np.ndarray] = None,
        freeze_embed: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_size, embed_dim, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )
        if freeze_embed:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=out_channels,
                    kernel_size=fs,
                )
                for fs in kernel_sizes
            ]
        )

        self.fc = nn.Linear(len(kernel_sizes) * out_channels, num_class)

        self.drop = nn.Dropout(dropout)
        self.embed_drop = nn.Dropout(embed_dropout)

    def forward(self, batch):

        inputs, _ = batch
        # inputs:  [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embed_drop(self.embedding(inputs).permute(0, 2, 1))
        # embeds = [BATCH_SIZE, EMBED_DIM, LONGEST_SEQ]

        convs = [F.relu(conv(embeds)) for conv in self.convs]
        # convs: List[[BATCH_SIZE, CONV_OUT_DIM, LONGEST_SEQ - KERNEL_SIZE + 1]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convs]
        # pooled: List[[BATCH_SIZE, CONV_OUT_DIM]]

        conv_out = self.drop(torch.cat(pooled, dim=1))
        # conv_out: [BATCH_SIZE, N_FILTERS * CONV_OUT_DIM]

        return self.fc(conv_out)
