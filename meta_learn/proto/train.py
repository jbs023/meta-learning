import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from meta_learn.proto.model import ProtoNetwork
from meta_learn.utils import get_meta_omniglot

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    avg_loss = list()
    for batch in tqdm(dataloader):
        support_x, support_y = batch["support"]  # (batch_size, way, shot, 28, 28)
        query_x, query_y = batch["query"]  # (batch_size, 1, shot, 28, 28)

        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        # Compute prediction error
        logits = model(support_x, support_y, query_x)
        loss = loss_fn(logits, query_y)
        avg_loss.append(loss.detach().item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(avg_loss)


def test(dataloader, model, loss_fn):
    # Evaluate model
    with torch.no_grad():
        accuracy_list = list()
        avg_loss = list()
        for i, batch in enumerate(dataloader):
            support_x, support_y = batch["support"]  # (batch_size, way, shot, 28, 28)
            query_x, query_y = batch["query"]  # (batch_size, 1, shot, 28, 28)

            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # calculate the accuracy
            logits = model(support_x, support_y, query_x)
            loss = loss_fn(logits, query_y)
            avg_loss.append(loss.detach().item())

            test_predictions = torch.argmax(logits, dim=1)
            accuracy = torch.mean((test_predictions == query_y).float())
            accuracy_list.append(accuracy.item())

        return np.mean(accuracy_list), np.mean(avg_loss)


def main(config):
    bs = config.batch_size
    way = config.num_classes
    shot = config.num_samples
    epochs = config.epochs
    path = config.path
    logdir = "{}/{}/_{}_{}".format(path, "model", config.num_classes, 1)
    writer = SummaryWriter(logdir)

    parent_dir = os.path.abspath(os.path.join(path, os.pardir))
    data_path = f"{parent_dir}/data/"
    train_set, test_set = get_meta_omniglot(data_path, way, shot)
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)
    testloader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)

    model = ProtoNetwork(1, 32)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, loss_fn, optimizer)

        # Test neural network
        if t % config.log_every == 0:
            test_accuracy, test_loss = test(testloader, model, loss_fn)
            print(f"Train Loss: {train_loss}    Test Loss: {test_loss}      Test Acc: {test_accuracy}")
            writer.add_scalar("Train Loss", train_loss, t)
            writer.add_scalar("Test Loss", test_loss, t)
            writer.add_scalar("Meta-Test Accuracy", test_accuracy, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--path", type=str, default=os.getcwd())
    main(parser.parse_args())
