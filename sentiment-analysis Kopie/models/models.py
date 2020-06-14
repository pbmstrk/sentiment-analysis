import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class fasttext(BaseModel):
    
    def __init__(self, vocab_size, embed_mat):
        
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, 300, 
            mode="mean").from_pretrained(torch.from_numpy(embed_mat).float())
        self.fc = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1),
        )

    def forward(self, batch):

        inputs, _, offsets = batch

        x = self.embedding(inputs, offsets)
        x = self.fc(x)
        return x.squeeze()

class lstm(nn.Module):

    def __init__(self, vocab_size, embed_mat):
    
        super().__init__()    

        self.embedding = nn.Embedding(vocab_size, 300, 
            padding_idx=0).from_pretrained(torch.from_numpy(embed_mat).float())
        self.lstm = nn.LSTM(300, 256, num_layers = 1, batch_first = True)
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout()

    def forward(self, batch):

        inputs, _, seqlengths = batch

        # input size: (batchsize, longest_seq)
    
        # get embeddings for each sequence
        embeds = self.embedding(inputs.long())
        # size after embedding: (batchsize, longest_seq, embed_dim)

        embeds = self.dropout(embeds)

        # as sequences are padded, pack them
        inputs = pack_padded_sequence(embeds, seqlengths, 
              enforce_sorted=False, batch_first=True)
        # size after packing: (sum(seqlengths), embed_dim)

        # as we leave the second argument empty, the hidden states are
        # initialized to zero - and also reset after every batch
        packed_output, (hidden, cell) = self.lstm(inputs)
        # packed output size: (sum(seqlengths), hidden_dim)
        # hidden size: (n_layers, batchsize, hidden_dim)

        # need hidden state from last layer
        lastState = hidden[-1]

        # linear layer
        output = self.dropout(F.relu(self.linear1(lastState)))
        output = self.linear2(output)

        return output.squeeze()