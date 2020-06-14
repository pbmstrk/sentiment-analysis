import torch
from utils.util import togpu

def eval_batch(model, batch, lossFun, device):

  batch = togpu(batch, device)
  _, targets, *args = batch
  model.eval()
  with torch.no_grad():
    batchsize = len(targets)
    logits = model(batch)
    # for loss
    lossVal = lossFun(logits, targets).item() * batchsize
    # for accuracy
    preds = torch.round(torch.sigmoid(logits))
    correct = (preds == targets).float().sum().item()

  return correct, lossVal

def eval(model, dataLoader, lossFun, device):

  correct = 0
  loss = 0
  total = 0
  for batch in dataLoader:
    c, l = eval_batch(model, batch, lossFun, device)
    correct += c
    loss += l
    total += batch[1].shape[0]

  acc = correct/total
  loss = loss/total
  return {"acc": acc, "loss": loss}