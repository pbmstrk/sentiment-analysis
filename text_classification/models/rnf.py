from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedLSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, time_axis: int, dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_axis = time_axis
        self.dropout = dropout
        self.lstm = nn.LSTM(self.input_dim, self.output_dim, batch_first=True)

        self.lstm_dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):

        batch_size = x.shape[0]
        time_steps = x.shape[self.time_axis]
        embed_dim = x.shape[-1]
        outputs = torch.zeros(batch_size, time_steps, embed_dim, device=x.device)

        for i in range(time_steps):
            x_input = torch.index_select(
                x,
                dim=self.time_axis,
                index=torch.tensor([i], device=x.device).long(),
            ).squeeze(1)

            _, (hidden_t, _) = self.lstm(x_input)

            outputs[:, i, :] = self.lstm_dropout(hidden_t)

        return outputs


def format_conv_input(x, filter_width, sent_len):

    chunks = []
    if sent_len - filter_width + 1 <= 1:
        return x.unsqueeze(1)
    for i in range(sent_len - filter_width + 1):
        chunk = x[:, i : i + filter_width, :]
        chunk = chunk.unsqueeze(1)
        chunks.append(chunk)
    return torch.cat(chunks, 1)


class RNF(nn.Module):

    r"""
    Convolutional Neural Networks with Recurrent Neural Filters

    Implementation of model which uses reccurent networks as convolution
    filters.

    Reference: `Yi Yang (2018). Convolutional neural networks with recurrent neural filters. <https://www.aclweb.org/anthology/D18-1109/>`_

    Args:
        input_size: Input size, for most cases size of vocabularly.
        num_class: Number of classes.
        filter_width: Width of the filter for the recurrent model.
        embed_dim: Size of the pre-trained word embeddings.
        hidden_dim: Size of the output layer of the LSTM.
        embed_dropout: Dropout applied to the word embeddings
        dropout: Dropout applied to the output of the LSTM.
        embed_mat: Pre-trained word-embedddings. Size should match (input_size, embed_dim)
        freeze_embed: Freeze embedding weights during training. For example, to keep pre-trained
            vectors (e.g. GloVe) fixed during training
        optimizer_name: Optimzer to use during training, pass name of optimizer found in
            torch.optim module
        optimizer_args: Arguments to pass to optimizer.

    Example::

        # for binary classification
        >>> RNFModel = RNF(input_size=100, num_class=2)
    """

    def __init__(
        self,
        input_size: int,
        num_class: int = 2,
        filter_width: int = 6,
        embed_dim: int = 300,
        hidden_dim: int = 300,
        embed_dropout: float = 0.4,
        dropout: float = 0.4,
        embed_mat: Optional[np.ndarray] = None,
        freeze_embed: bool = True,
    ):
        super().__init__()

        self.filter_width = filter_width

        self.embedding = nn.Embedding(input_size, embed_dim, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )
        if freeze_embed:
            self.embedding.weight.requires_grad = False

        self.time_lstm = TimeDistributedLSTM(
            embed_dim, hidden_dim, time_axis=1, dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, num_class)

        self.embed_drop = nn.Dropout(p=embed_dropout)

    def forward(self, batch):

        inputs, _ = batch
        # inputs: [BATCH_SIZE, LONGEST_SEQ]

        embedded = self.embed_drop(self.embedding(inputs))
        # embedded: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        lstm_inputs = format_conv_input(
            embedded, filter_width=self.filter_width, sent_len=embedded.shape[1]
        )
        # lstm_inputs: [BATCH SIZE, LONGEST SEQ - FILTER_WIDTH + 1,
        # FILTER_WIDTH, EMBED_DIM]

        lstm_outputs = self.time_lstm(lstm_inputs)
        # lstm_outputs: [BATCH SIZE, LONGEST SEQ - FILTER_WIDTH + 1, LSTM_OUT]

        lstm_outputs = F.max_pool1d(
            lstm_outputs.permute(0, 2, 1), kernel_size=lstm_outputs.shape[1]
        ).squeeze(-1)
        # lstm_outputs: [BATCH SIZE, LSTM_OUT]

        outputs = self.fc(lstm_outputs)
        return outputs
