import torch
from utils.util import togpu
import argparse
from logger.logger import set_logger
from utils.util import togpu, setdevice, Config
from processing.processdata import process_data_sst, get_embed_matrix
import models.models as models
import os
import logging
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--setup-file', default='setup.json',
                    help="path to setup.json file")
parser.add_argument('--checkpoint-file', default='bestModel.pt',
                    help="path to .pt file")


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

if __name__ == '__main__':

    set_logger('eval.log')

    device = setdevice()

    args = parser.parse_args()

    json_path = args.setup_file
    assert os.path.isfile(
        json_path), "No json setup file found at {}".format(json_path)
    config = Config(json_path)

    dataloaders, embed_mat = process_data_sst(config)

    model = config.init_obj('model', models, embed_mat = embed_mat).to(device)
    model.load('bestModel.pt')
    loss = nn.BCEWithLogitsLoss()

    logging.info(eval(model, dataloaders["test"], loss, device))

