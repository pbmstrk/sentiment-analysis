from utils.util import togpu, setdevice, Config
from evaluate import eval_batch, eval
import logging
from logger.logger import set_logger
import torch
import torchtext
from processing.processdata import process_data_sst, get_embed_matrix
import models.models as models
import torch.nn as nn
import torch.optim as optim
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='data/small',
                    help="path to config.json file")



def train(m, lossFun, optim, scheduler, epochs, train_loader, val_loader, device,
          saveModel = False, verbose=False):

    print("Summary of model\n")
    print(m)
    print("\n")

    bestLoss = 100

    for epoch in range(epochs):

        # Set model to training mode
        m.train()
    
        # Loop over each batch from the training set
        for batch in train_loader:

            # send batch to gpu
            batch = togpu(batch, device)
            # divide up
            _, targets, *args = batch
            # Zero gradient buffers
            optim.zero_grad()
            # Foward pass and compute loss on batch
            outputs = m(batch)
            batchloss = lossFun(outputs, targets)
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
            train_metrics = eval(m, train_loader, lossFun, device)
            val_metrics = eval(m, val_loader, lossFun, device)
    
            # opportunity to not include a scheduler
            if scheduler != None:
                scheduler.step()
    
            # check if new best for validation accuracy
            if val_metrics["loss"] < bestLoss:
                bestLoss = val_metrics["loss"]
                if saveModel == True:
                    model.save("bestModel.pt")
                    print("New best value for validation loss: Saved model to bestModel.pt")
            
            # print information about training progress
            if verbose == True:
                print(("Epoch: {} \t Loss (train): {:.3f} (val): {:.3f} \t" +
              "Acc (train) {:.3f} (val): {:.3f}").format(epoch + 1,
                            train_metrics["loss"], val_metrics["loss"], 
                            train_metrics["acc"], val_metrics["acc"]))
            # clean up
            del batch, targets, outputs

    return 


if __name__ == '__main__':

    set_logger("train.log")

    args = parser.parse_args()
    json_path = args.config_file
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    config = Config(json_path)

    device = setdevice()

    #models = {"fasttext": fasttext}



    dataloaders, embed_mat = process_data_sst(config)

    logging.info("Initialising model")
    torch.manual_seed(42)
    #model = models["fasttext"](embed_mat.shape[0], embed_mat)
    model = config.init_obj('model', models, embed_mat.shape[0], embed_mat)
    loss = nn.BCEWithLogitsLoss()
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    #optimizer = optim.Adam([
    #            {'params': model.fc.parameters(), 'lr': 3e-4},
    #            {'params': model.embedding.weight, 'lr': 1e-5}
    #        ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9) 

    train(model, loss, optimizer, scheduler, 30, dataloaders["train"], dataloaders["dev"], device,
      verbose=True, saveModel=True)
    

