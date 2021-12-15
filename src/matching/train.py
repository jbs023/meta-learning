import torch
import torch.nn as nn
import torch.optim as optim

import os
import errno
import argparse
import pandas as pd
import numpy as np

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from model import *

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

#Set up necessary folders for saving results
PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
if not os.path.exists(PATH+"/models/matching"):
    os.makedirs(PATH+"/models/matching")
if not os.path.exists(PATH+"/data"):
    os.makedirs(PATH+"/data/")
if not os.path.exists(PATH+"/output"):
    os.makedirs(PATH+"/output/")

def train(dataloader, model, loss_fn, optimizer, suffix, num_batches=100):
    global PATH
    avg_loss = list()
    avg_accuracy = list()
    for i, batch in enumerate(dataloader):
        train_inputs, train_targets = batch["train"] # (batch_size, way, shot, 28, 28)
        test_inputs, test_targets = batch["test"] # (batch_size, ??, shot, 28, 28)
        train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

        # Compute prediction error
        logits = model(train_inputs, train_targets, test_inputs)
        print(logits.shape, test_targets.shape)
        loss = loss_fn(logits, test_targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("Iteration Duration: {}".format(time.time()-start_time))
        if i % 10 == 0:
            test_predictions = torch.argmax(logits, dim=1)
            accuracy = torch.mean((test_predictions == test_targets).float())
            print('Loss: {0:>7f}  Accuracy: {1:.4f} [{2}/{3}]'.format(loss.item(), accuracy.item(), i, num_batches))
            avg_loss.append(loss.item())
            avg_accuracy.append(accuracy.item())

            # Save every N iterations
            torch.save(net.state_dict(), PATH+"/models/matching/matching-train-{}.pth".format(suffix))

        if i > num_batches:
            break
    
    torch.save(net.state_dict(), PATH+"/models/matching/matching-train-{}.pth".format(suffix))
    return np.mean(avg_loss), np.mean(avg_accuracy)

def test(dataloader, model, num_batches=100):   
    # Evaluate model 
    with torch.no_grad():
        accuracy_list = list()
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print("\tBatch {}/{}".format(i, num_batches))
            train_inputs, train_targets = batch["train"] # (batch_size, way, shot, 28, 28)
            test_inputs, test_targets = batch["test"] # (batch_size, ??, shot, 28, 28)
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            # calculate the accuracy
            logits = model(train_inputs, train_targets, test_inputs)
            test_predictions = torch.argmax(logits, dim=1)

            accuracy = torch.mean((test_predictions == test_targets).float())
            accuracy_list.append(accuracy.item())

            if i > num_batches:
                break

        print('*'*70)
        print('\t\tTest set \tprecision:\t%f'%(np.mean(accuracy_list)))
        print('*'*70)

        return np.mean(accuracy_list)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments for the swarm consensus simulation.')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Parameter to determine test vs train')
    parser.add_argument('-s', '--file_suffix', default="net", required=False, help='suffix for the output files')

    args = parser.parse_args()
    suffix = args.file_suffix
    batch_size = 32
    way = 5
    shot = 1
    epochs = 100

    dataset = omniglot(PATH+"/data", ways=way, shots=shot, test_shots=shot, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, num_workers=4)

    if args.train:
        net = MatchingNetwork(1, 64)
        if torch.cuda.is_available(): 
            net.cuda()

        loss_fn = nn.NLLLoss()
        optimizer = optim.Adam(net.parameters(), lr=6e-5, weight_decay=1e-3)
        track_loss = list()

        #Train network
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss, accuracy = train(dataloader, net, loss_fn, optimizer, suffix, num_batches=200)

            #Save loss to a file
            track_loss.append((loss, accuracy))
            df = pd.DataFrame(track_loss, columns=['Loss', 'Accuracy'])
            df.to_csv(PATH+"/output/matching_final_loss_{}.csv".format(suffix))

        print("Done!")
        print('Finished Training')

        #Save loss over time and network weights
        filePath = PATH+"/models/matching/matching-final-{}.pth".format(suffix)
        torch.save(net.state_dict(), filePath)
        print("Saved PyTorch Model State to {}".format(filePath))
    else:
        #Test neural network
        if os.path.exists(PATH+"/models/matching/matching-final-{}.pth".format(suffix)):
            net = MatchingNetwork(1, 64)
            net.load_state_dict(torch.load(PATH+"/models/matching/matching-final-{}.pth".format(suffix)))
            if torch.cuda.is_available(): 
                net.cuda()

            results = list()
            for i in range(0, 10):
                print("Trial {}".format(i))
                result = test(dataloader, net)
                results.append(result)

            mean = np.mean(results)
            std = np.std(results)
            min = np.min(results)
            max = np.max(results)
            print("Final Results: {} +- {} ({} - {})".format(mean, std, min, max))

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PATH)
