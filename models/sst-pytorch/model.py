import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

  def __init__(self, embeddingDIM, hiddenDIM1, hiddenDIM2, outputDIM, 
                 n_layers, vocab_size):
    
    super(LSTM, self).__init__()    

    self.embedding = nn.Embedding(vocab_size, embeddingDIM, padding_idx=0)
    self.lstm = nn.LSTM(embeddingDIM, hiddenDIM1, num_layers = n_layers, batch_first = True)
    self.linear1 = nn.Linear(hiddenDIM1, hiddenDIM2)
    self.linear2 = nn.Linear(hiddenDIM2, outputDIM)
    self.dropout = nn.Dropout()
    

  def forward(self, inputs, seqlengths):

    # input size: (batchsize, longest_seq)
    
    # get embeddings for each sequence
    embeds = self.embedding(inputs)
    # size after embedding: (batchsize, longest_seq, embed_dim)

    
    # as sequences are padded, pack them
    inputs = pack_padded_sequence(embeds, seqlengths, 
              enforce_sorted=False, batch_first=True)
    # size after packing: (sum(seqlengths), embed_dim)

    # as we leave the second argument empty, the hidden states are
    # initialized to zero - and also reset after every batch
    _, (hidden, _) = self.lstm(inputs)
    # packed output size: (sum(seqlengths), hidden_dim)
    # hidden size: (n_layers, batchsize, hidden_dim)

    # need hidden state from last layer
    lastState = hidden[-1]

    lastState = self.dropout(lastState)

    # linear layer
    output = self.dropout(torch.tanh(self.linear1(lastState)))
    output = self.linear2(output).squeeze()

    return output