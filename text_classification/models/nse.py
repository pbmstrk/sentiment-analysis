from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSE(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_class: int = 2,
        n_units: int = 300,
        mlp_dim: int = 1024,
        embed_mat: Optional[np.ndarray] = None,
        dropout: float = 0.4,
        freeze_embed: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_class = num_class
        self.n_units = n_units
        self.mlp_dim = mlp_dim
        self.embed_mat = embed_mat
        self.dropout = dropout
        self.freeze_embed = freeze_embed

        self.embedding = nn.Embedding(self.input_size, self.n_units, padding_idx=0)
        if self.embed_mat is not None:
            self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(self.embed_mat).float()
            )
        if self.freeze_embed:
            self.embedding.weight.requires_grad = False

        self.read_lstm = nn.LSTM(self.n_units, self.n_units, batch_first=True)
        self.write_lstm = nn.LSTM(2 * self.n_units, self.n_units, batch_first=True)
        self.compose_layer = nn.Linear(2 * self.n_units, 2 * self.n_units)

        self.drop = nn.Dropout(self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.n_units, self.mlp_dim),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.mlp_dim, self.num_class),
        )

    def _init_hidden(self, batch_size, hidden_dim, device):
        return (
            torch.zeros(1, batch_size, hidden_dim, device=device).requires_grad_(),
            torch.zeros(1, batch_size, hidden_dim, device=device).requires_grad_(),
        )

    def _compose(self, o_t, m_t):

        c_t = self.compose_layer(torch.cat([o_t, m_t], dim=1))  # equation (4)
        # c_t: [BATCH_SIZE, 2*N_UNITS]

        return c_t

    def _read(self, M_t, x_t, hidden):

        o_t, hidden = self.read_lstm(
            F.dropout(x_t, 0.3, training=self.training), hidden
        )  # equation (1)
        o_t = o_t.squeeze(1)
        # o_t: [BATCH_SIZE, N_UNITS]

        z_t = F.softmax(torch.einsum("bo,bot->bt", o_t, M_t), dim=1)  # equation (2)
        # z_t: [BATCH_SIZE, SEQ_LEN]

        m_rt = torch.einsum("bt,bot->bo", z_t, M_t)  # equation (3)
        # m_rt: [BATCH_SIZE, N_UNITS]

        return o_t, m_rt, z_t, hidden

    def _write(self, M_t, c_t, z_t, hidden):

        # get device tensors are currently on
        device = M_t.device

        batch_size, n_units, seq_len = M_t.shape

        h_t, hidden = self.write_lstm(
            F.dropout(c_t.unsqueeze(1), 0.3, training=self.training), hidden
        )  # equation (5)
        # h_t: [BATCH_SIZE, 1, N_UNITS]

        z_t_e_kT = torch.einsum(
            "ki,kj->kji", [z_t, torch.ones(batch_size, n_units, device=device)]
        )
        # z_t_e_kT: [BATCH_SIZE, N_UNITS, SEQ_LEN]

        h_t_e_l = torch.einsum(
            "ki,kj->kij",
            [h_t.squeeze(1), torch.ones(batch_size, seq_len, device=device)],
        )
        # h_t_e_l: [BATCH_SIZE, N_UNITS, SEQ_LEN]

        M_t = (1 - z_t_e_kT) * M_t + h_t_e_l * z_t_e_kT  # equation (6)
        # M_t: [BATCH_SIZE, N_UNITS, SEQ_LEN]

        return M_t, h_t, hidden

    def forward(self, batch):

        inputs, _, seqlengths = batch
        # inputs: [BATCH_SIZE, LONGEST_SEQ]

        embeds = self.embedding(inputs)
        # embeds: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        M_t = embeds.permute(0, 2, 1)
        # M_t: [BATCH_SIZE, LONGEST_SEQ, EMBED_DIM]

        all_outputs = torch.zeros(
            inputs.shape[0], inputs.shape[1], self.n_units, device=inputs.device
        )
        idx = torch.tensor(seqlengths)
        idx = idx - 1

        read_hidden = self._init_hidden(
            inputs.shape[0], self.n_units, device=inputs.device
        )
        write_hidden = self._init_hidden(
            inputs.shape[0], self.n_units, device=inputs.device
        )

        for i in range(inputs.shape[1]):

            x_t = torch.index_select(
                embeds, 1, torch.tensor([i], device=embeds.device).long()
            )
            # x_t: [BATCH_SIZE, 1, DIM]

            o_t, m_rt, z_t, read_hidden = self._read(M_t, x_t, read_hidden)

            c_t = self._compose(o_t, m_rt)

            M_t, h_t, write_hidden = self._write(M_t, c_t, z_t, write_hidden)

            all_outputs[:, i, :] = h_t.squeeze(1)

        output = all_outputs[torch.arange(all_outputs.size(0)), idx]
        output = self.fc(output)

        return output
