import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from text_classification.models.base import BaseClassifier


class SentimentLSTM(BaseClassifier):
    def __init__(self, input_size, embed_mat=None):

        super().__init__()

        self.embedding = nn.Embedding(input_size, 300, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )
        self.lstm = nn.LSTM(300, 256, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-05)
        return optimizer

    def forward(self, batch):

        inputs, _, seqlengths = batch
        # inputs: [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embedding(inputs.long())
        # embeds: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        embeds = self.dropout(embeds)

        inputs = pack_padded_sequence(
            embeds, seqlengths, enforce_sorted=False, batch_first=True
        )
        # inputs: [SUM(SEQ_LENGTHS), EMBED_DIM)

        _, (hidden, _) = self.lstm(inputs)
        # packed_outputs: [SUM(SEQ_LENGTHS), LSTM_OUT]
        # hidden: [1, BATCH_SIZE, LSTM_OUT]

        last_state = hidden[-1]
        # lastState: [BATCH_SIZE, LSTM_OUT]

        output = self.dropout(F.relu(self.linear1(last_state)))
        output = self.linear2(output).squeeze()

        return output
