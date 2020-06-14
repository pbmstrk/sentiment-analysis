import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.attention import Attention

class fasttext(BaseModel):
    
    def __init__(self, vocab_size, embed_mat=None):
        
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

class lstm(BaseModel):

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


class lstmAttn(BaseModel):

    def __init__(self, vocab_size, embed_mat):
    
        super().__init__()    

        self.embedding = nn.Embedding(vocab_size, 300, 
            padding_idx=0).from_pretrained(torch.from_numpy(embed_mat).float())
        self.attention = Attention()
        self.lstm = nn.LSTM(300, 256, batch_first = True)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def create_mask(self, inputs):
        mask = (inputs != 0)
        return mask.unsqueeze(1)

    def forward(self, batch):

        inputs, _, seqlengths = batch
        # input size: (batchsize, longest_seq)

        # create mask for attention
        mask = self.create_mask(inputs)
    
        # get embeddings for each sequence
        embeds = self.embedding(inputs.long()).to(device)
        # size after embedding: (batchsize, longest_seq, embed_dim)

        # as sequences are padded, pack them
        inputs = pack_padded_sequence(embeds, seqlengths, 
              enforce_sorted=False, batch_first=True)
        # size after packing: (sum(seqlengths), embed_dim)

        # as we leave the second argument empty, the hidden states are
        # initialized to zero - and also reset after every batch
        packed_outputs, (hidden, cell) = self.lstm(inputs)
        # hidden size: (n_layers, batchsize, hidden_dim)

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # [batchsize, longest seq, hidden size]

        a = self.attention(hidden.squeeze(0), outputs, mask)

        weighted = torch.bmm(a, outputs).squeeze(1)

        # linear layer
        output = self.dropout(F.relu(self.fc1(weighted)))
        output = self.fc2(output)

        return output.squeeze()

class cnn(BaseModel):
    def __init__(self, vocab_size, embed_mat):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx = 0).from_pretrained(torch.from_numpy(embed_mat).float())
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = 300, 
                                              out_channels = 100, 
                                              kernel_size = fs)
                                    for fs in [3,4,5]
                                    ])
        
        self.fc = nn.Linear(3 * 100, 1)
        
        self.dropout = nn.Dropout()
        
    def forward(self, batch):

        inputs, _ = batch
        
        #text = [batch size, sent len]
        embedded = self.embedding(inputs)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat).squeeze()