import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import errno
import argparse
import pandas as pd
import numpy as np
import pathlib

from torch.utils.data import DataLoader
# from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Resize

from dataset import *
from model import *

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
path = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    avg_loss = list()
    for i, data in enumerate(dataloader):
        x1, x2, y = data
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction error
        pred = model(x1, x2)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(avg_loss)

def test(testloader, net):   
    with torch.no_grad():
        right, error = 0, 0
        for _, (test1, test2) in enumerate(testloader, 1):
            # print("\tIteration: {}/{}".format(_, len(testloader)))
            test1, test2 = test1.to(device), test2.to(device)
            output = net.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            
            if pred == 0:
                right += 1
            else: 
                error += 1

        return right*1.0/(right+error)
    
def main(config):
    epochs = config.epochs
    batch_size = config.batch_size
    way = config.num_classes
    transforms = Compose([ToTensor()])
    train_path = "{}/data/images_background".format(path)
    test_path = "{}/data/images_evaluation".format(path)
    num_examples = 30000

    #Log training progress
    logdir = "{}/{}_{}_{}".format(path, config.logdir, config.num_classes, 1)
    writer = SummaryWriter(logdir)

    #Set up training dataset and neural network 
    trainSet = OmniglotTrain(train_path, transform=transforms, num_examples=num_examples)
    trainloader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=4)
    net = SiameseCNN()
    if torch.cuda.is_available(): 
        net.cuda()
        
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=6e-5, weight_decay=1e-3)

    #Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, net, loss_fn, optimizer)

        if t%config.log_every == 0:
            #Test neural network
            testSet = OmniglotTest(test_path, transform=transforms, times=400, way=way)
            testloader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=4)
            test_accuracy = test(testloader, net)

            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), t)
            writer.add_scalar('Meta-Test Accuracy', test_accuracy, t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='run/siamese')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=1)
    main(parser.parse_args())