
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



from tqdm import tqdm
from torch.utils.data import DataLoader
from meta_learn.utils import get_siamese_omniglot
from meta_learn.siamese.model import SiameseCNN
from meta_learn.dataset import *

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    avg_loss = list()
    for data in tqdm(dataloader):
        x1, x2, y = data
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction error
        pred = model(x1, x2)
        loss = loss_fn(pred, y)
        avg_loss.append(loss.detach().item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(avg_loss)

def test(testloader, model):   
    with torch.no_grad():
        correct, total = 0, 0
        for data in testloader:
            x1, x2, y = data
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            output = model(x1, x2)

            predicted = torch.round(output.data)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(correct, total)
        return correct*1.0/total
    
def main(config):
    #Set up config params
    epochs = config.epochs
    batch_size = config.batch_size
    way = config.num_ways
    path = config.path
    logdir = "{}/{}/_{}_{}".format(path, "model", config.num_ways, 1)
    writer = SummaryWriter(logdir)

    # Download Omniglot Dataset
    parent_dir = os.path.abspath(os.path.join(path, os.pardir))
    data_path = f"{parent_dir}/data/"
    train_set, test_set = get_siamese_omniglot(data_path)

    #Set up training dataset and neural network 
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    model = SiameseCNN()

    if torch.cuda.is_available(): 
        model.cuda()
        
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    #Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, loss_fn, optimizer)

        if t%config.log_every == 0:
            #Test neural network
            test_accuracy = test(testloader, model)
            print(f"Train Loss: {train_loss}    Test Acc: {test_accuracy}")
            writer.add_scalar('Train Loss', train_loss, t)
            writer.add_scalar('Meta-Test Accuracy', test_accuracy, t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--path', type=str, default=os.getcwd())
    main(parser.parse_args())