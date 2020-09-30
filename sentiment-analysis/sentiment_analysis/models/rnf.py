import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sentiment_analysis.models.base import BaseClassifier


class TimeDistributedLSTM(pl.LightningModule):
    def __init__(self, time_axis):
        super().__init__()

        self.time_axis = time_axis
        self.lstm = nn.LSTM(300, 300, batch_first=True)

    def forward(self, x):

        batch_size = x.shape[0]
        time_steps = x.shape[self.time_axis]
        embed_dim = x.shape[-1]
        outputs = torch.zeros(batch_size, time_steps, embed_dim, device=self.device)

        for i in range(time_steps):
            x_input = torch.index_select(
                x,
                dim=self.time_axis,
                index=torch.tensor([i], device=self.device).long(),
            ).squeeze()

            _, (hidden_t, _) = self.lstm(x_input)

            outputs[:, i, :] = hidden_t

        return outputs


def format_conv_input(x, filter_width, sent_len):

    chunks = []
    for i in range(sent_len - filter_width + 1):
        chunk = x[:, i : i + filter_width, :]
        chunk = chunk.unsqueeze(1)
        chunks.append(chunk)
    return torch.cat(chunks, 1)


class RNF(BaseClassifier):
    def __init__(self, input_size, embed_mat=None):
        super().__init__()

        self.embedding = nn.Embedding(input_size, 300, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )

        self.filter_width = 5
        self.time_lstm = TimeDistributedLSTM(time_axis=1)

        self.fc = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1),
        )

    def forward(self, batch):

        inputs, _ = batch
        # inputs: [BATCH_SIZE, LONGEST_SEQ]

        embedded = self.embedding(inputs)
        # embedded: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        lstm_inputs = format_conv_input(
            embedded, filter_width=self.filter_width, sent_len=embedded.shape[1]
        )
        # lstm_inputs: [BATCH SIZE, LONGEST SEQ - FILTER_WIDTH + 1,
        # FILTER_WIDTH, EMBED_DIM]

        lstm_outputs = self.time_lstm(lstm_inputs)
        # lstm_outputs: [BATCH SIZE, LONGEST SEQ - FILTER_WIDTH + 1,
        # FILTER_WIDTH, LSTM_OUT]

        lstm_outputs = F.max_pool1d(
            lstm_outputs.permute(0, 2, 1), kernel_size=lstm_outputs.shape[1]
        ).squeeze()
        # lstm_outputs: [BATCH SIZE, LSTM_OUT]

        outputs = self.fc(lstm_outputs)
        return outputs.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-05)
        return optimizer
