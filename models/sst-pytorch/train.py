import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter

from utils import *
from model import LSTM 

import copy

def train(m, lossFun, optim, scheduler, epochs, train_loader, val_loader, train_eval,
          saveModel = False, verbose=False):

    print("Summary of model\n")
    print(m)
    print("\n")

    # initialize lists to store loss and accuracy
    trainLossVec , valLossVec, trainAccuracyVec, valAccuracyVec = [], [], [], []
    bestLoss, bestEpoch = 100, 0

    for epoch in range(epochs):

        # Set model to training mode
        m.train()
    
        # Loop over each batch from the training set
        for batch_idx, (inputs, targets, seqlengths) in enumerate(train_loader):
            
            # Zero gradient buffers
            optim.zero_grad()
            # Foward pass and compute loss on batch
            outputs = m(inputs.long().to(device), seqlengths)
            batchloss = lossFun(outputs, targets.to(device))
            # Backpropagate and update weights
            batchloss.backward()
            # gradient clipping
            #torch.nn.utils.clip_grad_norm_(m.parameters(), 1., norm_type=2)
            # optimizer step
            optim.step()
    
        # set model to evaluation mode
        m.eval()
        with torch.no_grad():
            
            # evaluate model on training and validation data
            train_metrics = evaluate(m, train_eval, lossFun)
            val_metrics = evaluate(m, val_loader, lossFun)
            # update accuracy
            trainAccuracyVec.append(train_metrics["acc"])
            valAccuracyVec.append(val_metrics["acc"])
            # update loss
            trainLossVec.append(train_metrics["loss"])
            valLossVec.append(val_metrics["loss"])
            
            if scheduler != None:
              scheduler.step()
    
            # check if new best for validation accuracy
            if valLossVec[-1] < bestLoss:
                bestLoss = valLossVec[-1]
                bestEpoch = epoch
                if saveModel == True:
                    torch.save(m.state_dict(), "bestModel.pt")
                    print("New best value for validation loss: Saved model to bestModel.pt")
            
            # print information about training progress
            if verbose == True:
                print(("Epoch: {} \t Loss (train): {:.3f} (val): {:.3f} \t" +
              "Acc (train) {:.3f} (val): {:.3f}").format(epoch + 1,
                            trainLossVec[-1], valLossVec[-1], trainAccuracyVec[-1], valAccuracyVec[-1]))
            # clean up
            del inputs, targets, outputs

    return trainLossVec, valLossVec, trainAccuracyVec, valAccuracyVec, bestEpoch

def eval_batch(model, batch, lossFun):

    inputs, targets, seqlengths = batch
    model.eval()
    with torch.no_grad():
        batchsize = len(targets)
        logits = model(inputs.long().to(device), seqlengths)
        # for loss
        lossVal = lossFun(logits, targets.to(device)).item() * batchsize
        # for accuracy
        pred = logits >= 0
        correct = (pred == targets.to(device)).sum().item()

    return correct, lossVal


def evaluate(model, dataLoader, lossFun):

    correct = 0
    loss = 0
    total = 0
    for batch in dataLoader:
        c, l = eval_batch(model, batch, lossFun)
        correct += c
        loss += l
        total += batch[1].shape[0]

    acc = correct/total
    loss = loss/total
    return {"acc": acc, "loss": loss}

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load data
    traindf = pd.read_csv('./data/train.csv')
    devdf = pd.read_csv('./data/dev.csv')
    testdf = pd.read_csv('./data/test.csv')
    
    # store data and labels in arrays
    X_train, y_train = np.array(traindf.iloc[:, 0]), np.array(traindf.iloc[:, 1])
    X_dev, y_dev = np.array(devdf.iloc[:, 0]), np.array(devdf.iloc[:, 1])
    X_test, y_test = np.array(testdf.iloc[:, 0]), np.array(testdf.iloc[:, 1])

    # tokenize sentences
    X_train = [tokenizer(phrase) for phrase in X_train]
    X_dev = [tokenizer(phrase) for phrase in X_dev]
    X_test = [tokenizer(phrase) for phrase in X_test]

    # create vocabularly
    vocabCount = Counter([item for sublist in X_train for item in sublist])
    vocab = Vocabulary(vocabCount, 5)

    # create numeric training data using mapping
    X_trainNum = [torch.tensor(add_special_tokens([vocab[word] for word in phrase], vocab.BOS_token, vocab.EOS_token)) for phrase in X_train]
    X_devNum = [torch.tensor(add_special_tokens([vocab[word] for word in phrase], vocab.BOS_token, vocab.EOS_token)) for phrase in X_dev]
    X_testNum = [torch.tensor(add_special_tokens([vocab[word] for word in phrase], vocab.BOS_token, vocab.EOS_token)) for phrase in X_test]  


    # create datasets and dataloaders
    trainingset = WordDataset(X_trainNum, y_train)
    valset = WordDataset(X_devNum, y_dev)
    testset = WordDataset(X_testNum, y_test)

    training_generator = data.DataLoader(trainingset, batch_size=64, collate_fn=generate_batch, shuffle = True)
    val_generator = data.DataLoader(valset, batch_size=len(y_dev), collate_fn=generate_batch)
    test_generator = data.DataLoader(testset, batch_size=len(y_test), collate_fn=generate_batch)

    training_eval = data.DataLoader(trainingset, batch_size=512, collate_fn = generate_batch)

    torch.manual_seed(42)
    model = LSTM(256, 256, 256, 1, 1, len(vocab)).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay = 1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6) 
    loss = nn.BCEWithLogitsLoss()

    trainLossVec, valLossVec, trainAccuracyVec, valAccuracyVec, bestEpoch = train(model, loss, optimizer, 
                  scheduler, 25, training_generator, val_generator, training_eval, verbose = True, saveModel=True)