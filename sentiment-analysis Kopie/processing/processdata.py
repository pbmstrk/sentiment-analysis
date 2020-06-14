import logging
import torchtext
from collections import Counter
from torch.utils import data
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def formatdata(dataset):

    targetEncoding = {'negative': 0, 'positive': 1}

    labels = []
    phrases = []
    for element in dataset:
        labels.append(element.label)
        phrases.append(element.text)

    labels = [targetEncoding[label] for label in labels]

    return phrases, labels

def loaddata():
    logging.info("Retrieving dataset")

    TEXT = torchtext.data.Field(tokenize='spacy')
    LABEL = torchtext.data.Field(sequential=False)

    traindata, devdata, testdata = torchtext.datasets.SST.splits(TEXT, LABEL, 
                                filter_pred=lambda ex: ex.label != 'neutral')

    X_train, y_train = formatdata(traindata)
    X_dev, y_dev = formatdata(devdata)
    X_test, y_test = formatdata(testdata)

    examples = {"train": X_train, "dev": X_dev, "test": X_test}
    labels = {"train": y_train, "dev": y_dev, "test": y_test}

    return examples, labels

class Vocabulary:
      
    def __init__(self, vocabCount, min_freq):
        
        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3
        self.vocabCount = vocabCount
        self.min_freq = min_freq
        # initialize list of words and vocab dictionary
        self.wordlist = ["<unk>", "<sos>", "<eos>"]
        self.word2index = {}
        # build vocab
        self.build_vocab(vocabCount)

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, word):
        return self.word2index.get(word, 1)

    def __iter__(self):
        return iter(self.word2index)

    def build_vocab(self, vocabCount):
        # sort vocab s.t. words that occur most frequently added first
        svocabCount = {k: v for k, v in reversed(sorted(vocabCount.items(), 
                                                      key=lambda item: item[1]))}
        
        for word in svocabCount:
            if svocabCount[word] >= self.min_freq:
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

def toids(dataset, vocab):

    return [torch.tensor([vocab.SOS_token] + [vocab[word] for word in phrase] + [vocab.EOS_token]) for phrase in dataset]

def build_vocab(examples):

    logging.info("Building vocab")  
    vocabCount = Counter([item for sublist in examples["train"] for item in sublist])
    vocab = Vocabulary(vocabCount, 3)

    return vocab

def process_data_sst(config):

    batchfn = {"fasttext": generate_batch_ft, "lstm": generate_batch_lstm}

    phrases, labels = loaddata()
    vocab = build_vocab(phrases)
    embed_mat = get_embed_matrix(vocab)

    X_trainNum = toids(phrases["train"], vocab)
    X_devNum = toids(phrases["dev"], vocab)
    X_testNum = toids(phrases["test"], vocab)

    trainingset = WordDataset(X_trainNum, labels["train"])
    devset = WordDataset(X_devNum, labels["dev"])
    testset = WordDataset(X_testNum, labels["test"])  

    generate_batch = batchfn[config.model["type"]]

    logging.info("Constructing dataloaders")
    train_generator = data.DataLoader(trainingset, batch_size=config.batch_size, 
        collate_fn=generate_batch, shuffle=True)
    dev_generator = data.DataLoader(devset, batch_size=len(labels["dev"]), 
        collate_fn=generate_batch)
    test_generator = data.DataLoader(testset, batch_size=len(labels["test"]), 
        collate_fn=generate_batch) 

    dataloaders = {"train": train_generator, "dev": dev_generator, "test": test_generator}
    return dataloaders, embed_mat

def download_embeddings():
    logging.info("Downloading embeddings")
    glove = torchtext.vocab.GloVe(name='42B', dim=300, unk_init = torch.Tensor.normal_)
    return glove

def generate_batch_ft(batch):
    
    # get data and targets from batch
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [len(el) for el in data]
    offsets = np.cumsum(lengths)
    offsets = np.concatenate([[0], offsets[:-1]])

    return torch.LongTensor(torch.cat(data).long()), torch.Tensor(targets).float(), torch.LongTensor(offsets)

def generate_batch_lstm(batch):

    # get inputs and targets
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # to be able to pack sequences later on, need
    # the original sequence lengths
    seqlengths = [len(el) for el in data]
    
    # pad the sequences
    x = pad_sequence(data, batch_first=True)

    return x, torch.Tensor(targets).float(), seqlengths

def get_embed_matrix(vocab):

    glove = download_embeddings()
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
  
    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = glove.vectors[glove.stoi[word]]
        
        except KeyError:

            weights_matrix[i] = np.random.normal(scale=0.5, size=(300, ))

    return weights_matrix   
