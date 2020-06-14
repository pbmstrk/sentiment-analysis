import torch 
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

  def __init__(self):
    super().__init__()
    

  def forward(self, hidden, outputs, mask):

    dim = hidden.shape[1]
    scale = 1/torch.sqrt(torch.tensor(dim).float())

    # hidden size: [batchsize, hid_dim]
    # outputs: [batchsize, seq length, hid_dim] 
    outputs = outputs.permute(0,2,1)
    hidden = hidden.unsqueeze(1)

    attention = torch.bmm(hidden, outputs)
    attention = attention.masked_fill(mask == 0, -1e-10)
    attention.mul_(scale)

    return F.softmax(attention, dim=2)