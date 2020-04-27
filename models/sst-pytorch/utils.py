import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

import spacy
spacy_en = spacy.load('en')

def tokenizer(text):
    text = text.lower()
    tokens = spacy_en.tokenizer(text)
    tokens = [token.text for token in tokens]
    return tokens

def add_special_tokens(lst, bos_token, eos_token):
  res = [bos_token] + lst + [eos_token]
  return res


class Vocabulary:
      
    def __init__(self, vocabCount, min_freq):
        # initialize PAD and UNK tokens
        self.PAD_token = 0   
        self.UNK_token = 1
        self.BOS_token = 2
        self.EOS_token = 3
        self.vocabCount = vocabCount
        self.min_freq = min_freq
        # initialize list of words and vocab dictionary
        self.wordlist = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.word2index = {}

        self.build_vocab(self.vocabCount)

    def __len__(self):
      return len(self.word2index)

    def __getitem__(self, word):
      return self.word2index.get(word, 1)

    def __iter__(self):
      return iter(self.word2index)

    def build_vocab(self, vocabCount):
      for word in self.vocabCount:
        if vocabCount[word] >= self.min_freq:
          self.wordlist.append(word)
      self.word2index.update({tok: i for i, tok in enumerate(self.wordlist)})


class WordDataset(data.Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):  
        X = self.X[idx]
        Y = self.y[idx]
        return X, Y

def generate_batch(batch):
    
    # get inputs and targets
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # to be able to pack sequences later on, need
    # the original sequence lengths
    seqlengths = [len(el) for el in data]
    
    # pad the sequences
    x = pad_sequence(data, batch_first=True)

    return x, torch.Tensor(targets), seqlengths
