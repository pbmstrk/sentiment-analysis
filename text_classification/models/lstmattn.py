import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from text_classification.models.base import BaseClassifier


class Attention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super().__init__()
        self.w = nn.Linear(2 * hidden_size, att_dim)
        self.u_w = nn.Linear(att_dim, 1)

    def forward(self, outputs, mask):

        outputs = outputs.permute(0, 2, 1)
        # outputs: [BATCH_SIZE, LSTM_OUT, LONGEST_SEQ]

        # compute u_{it} representations of each of the hidden
        u_it = torch.einsum("ki,bij->bkj", self.w.weight, outputs)
        # u_it: [BATCH_SIZE, ATTN_OUT, LONGEST_SEQ]

        # compute alpha
        alpha = torch.einsum("ij, bjk->bk", self.u_w.weight, u_it)
        # alpha: [BATCH_SIZE, LONGEST_SEQ]

        # use mask
        alpha = alpha.masked_fill(mask == 0, -1e-10)

        return F.softmax(alpha, dim=1)


class AttentionLSTM(BaseClassifier):
    def __init__(self, input_size, embed_mat=None):

        super().__init__()

        self.embedding = nn.Embedding(input_size, 300, padding_idx=0)
        if embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(embed_mat).float()
            )
        self.attention = Attention(256, 200)
        self.lstm = nn.LSTM(300, 256, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.embed_dropout = nn.Dropout(0.1)

    def _create_mask(self, inputs):
        mask = inputs != 0
        return mask

    def forward(self, batch):

        inputs, _, seqlengths = batch
        # inputs: [BATCH_SIZE, LONGEST_SEQ]

        mask = self._create_mask(inputs)
        # mask: [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embedding(inputs.long())
        # embeds: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        embeds = self.embed_dropout(embeds)

        inputs = pack_padded_sequence(
            embeds, seqlengths, enforce_sorted=False, batch_first=True
        )
        # inputs: [SUM(SEQ_LENGTHS), EMBED_DIM]

        packed_outputs, _ = self.lstm(inputs)
        # packed_outputs: [SUM(SEQ_LENGTHS), N_DIR * LSTM_OUT]
        # hidden: [N_DIR * N_LAYERS, BATCH_SIZE, LSTM_OUT]

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [BATCH_SIZE, LONGEST_SEQ, N_DIR * LSTM_OUT]

        a = self.attention(outputs, mask)
        # a: [BATCH_SIZE, LONGEST_SEQ]

        context = torch.einsum("bj, bjk -> bk", a, outputs)
        # context: [BATCH_SIZE, N_DIR * LSTM_OUT]

        # linear layer
        output = self.dropout(F.relu(self.fc1(context)))
        output = self.fc2(output)

        return output.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-05)
        return optimizer
