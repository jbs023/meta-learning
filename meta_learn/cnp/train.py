import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_learn.cnp.model import CNP
from meta_learn.datasets import (
    BatchMetaDataLoader,
    ClassSplitter,
    get_omniglot,
    get_sinusoid,
    Sinusoid,
)
from torch.distributions import MultivariateNormal
from meta_learn.proto.model import ProtoNetwork

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, optimizer, num_batches):
    avg_loss = list()

    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            #Move data onto GPU 
            train_inputs, train_targets, train_amp, train_phase = batch["train"]
            test_inputs, test_targets, test_amp, test_phase = batch["test"]

            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
            test_inputs = test_inputs.to(device)
            test_targets = test_targets.to(device)

            #Forward pass
            mu, sigma = model(train_inputs, train_targets, test_inputs)

            #Explicitly save values for distribution
            #NOTE: Doing my best to avoid hidden side effects
            model.mu = mu
            model.sigma = sigma

            #Calculate loss
            target_mask = torch.ones(test_targets.shape).to(device)
            loss = model.loss_fn(test_targets, target_mask)

            # Compute prediction error
            pbar.set_postfix(loss="{0:.4f}".format(loss.detach().item()))
            avg_loss.append(loss.detach().item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch_idx > num_batches:
                break

    return np.mean(avg_loss)


def test(dataloader, model, num_batches, logdir):
    avg_loss = list()
    count = 0

    with torch.no_grad():
        with tqdm(dataloader, total=num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                model.zero_grad()
                model.zero_grad()

                train_inputs, train_targets, train_amp, train_phase = batch["train"]
                test_inputs, test_targets, test_amp, test_phase = batch["test"]

                train_inputs = train_inputs.to(device)
                train_targets = train_targets.to(device)
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)

                #Forward pass
                mu, sigma = model(train_inputs, train_targets, test_inputs)

                #Explicitly save values for distribution
                #NOTE: Doing my best to avoid hidden side effects
                model.mu = mu
                model.sigma = sigma

                #Setup target_mask (i.e., all ones)
                target_mask = torch.ones(test_targets.shape).to(device)

                #Calculate loss
                loss = model.loss_fn(test_targets, target_mask)

                #Plot one function
                if count == 0:
                    model.plot_functions(train_inputs.cpu(), train_targets.cpu(), test_inputs.cpu(), test_targets.cpu(), mu.cpu(), sigma.cpu(), logdir)
                    count += 1

                # Compute prediction error
                pbar.set_postfix(loss="{0:.4f}".format(loss.detach().item()))
                avg_loss.append(loss.detach().item())

                if batch_idx > num_batches:
                    break

    return np.mean(avg_loss)

def main(config):
    bs = config.batch_size
    way = config.num_ways
    shot = config.num_shots
    epochs = config.epochs
    path = config.path
    num_batches = config.num_batches
    logdir = "{}/{}/{}_way_{}_shot".format(path, "model", config.num_ways, config.num_shots)
    writer = SummaryWriter(logdir)

    train_ds, test_ds = get_sinusoid(shot)
    trainloader = BatchMetaDataLoader(train_ds, batch_size=bs, shuffle=True)
    testloader = BatchMetaDataLoader(test_ds, batch_size=bs, shuffle=False)

    model = CNP(2, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device=device)

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, optimizer, num_batches)

        # Test neural network
        if t % config.log_every == 0:
            test_loss = test(testloader, model, num_batches, logdir)
            print( f"Train Loss: {train_loss}    Test Loss: {test_loss}")
            writer.add_scalar("Train Loss", train_loss, t)
            writer.add_scalar("Test Loss", test_loss, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ways", type=int, default=5)
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--num_batches", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--path", type=str, default=os.getcwd())
    main(parser.parse_args())
