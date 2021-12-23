import torch
import argparse
import pathlib
import numpy as np
import torch.optim as optim

from model import *
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
path = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, optimizer, num_batches=100):
    avg_loss = list()
    for i, batch in enumerate(dataloader):
        train_inputs, train_targets = batch["train"] # (batch_size, way, shot, 28, 28)
        test_inputs, test_targets = batch["test"] # (batch_size, ??, shot, 28, 28)
        train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

        # Compute prediction error
        logits = model(train_inputs, train_targets, test_inputs)
        loss = model.loss_function(logits, test_targets)
        avg_loss.append(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i > num_batches:
            break
    
    return np.mean(avg_loss)

def test(dataloader, model, num_batches=100):   
    # Evaluate model 
    with torch.no_grad():
        accuracy_list = list()
        avg_loss = list()
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print("\tBatch {}/{}".format(i, num_batches))
            train_inputs, train_targets = batch["train"] # (batch_size, way, shot, 28, 28)
            test_inputs, test_targets = batch["test"] # (batch_size, ??, shot, 28, 28)
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

            # calculate the accuracy
            logits = model(train_inputs, train_targets, test_inputs)
            loss = model.loss_function(logits, test_targets)
            avg_loss.append(loss)

            test_predictions = torch.argmax(logits, dim=1)
            accuracy = torch.mean((test_predictions == test_targets).float())
            accuracy_list.append(accuracy.item())

            if i > num_batches:
                break

        return np.mean(accuracy_list), np.mean(avg_loss)

def main(config):
    batch_size = config.batch_size
    way = config.num_classes
    shot = config.num_samples
    epochs = config.epochs

    data_path = "{}/data".format(path)
    dataset = omniglot(data_path, ways=way, shots=shot, test_shots=shot, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, num_workers=4)
    logdir = "{}/{}_{}_{}".format(path, config.logdir, config.num_classes, 1)
    writer = SummaryWriter(logdir)

    net = ProtoNetwork(1, 64)
    if torch.cuda.is_available(): 
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=6e-5, weight_decay=1e-3)

    #Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(dataloader, net, optimizer, num_batches=200)

        #Test neural network
        if t%config.log_every == 0:
            test_accuracy, test_loss = test(dataloader, net)
            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), t)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), t)
            writer.add_scalar('Meta-Test Accuracy', test_accuracy, t)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--logdir', type=str, default='run/proto')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=1)
    main(parser.parse_args())
