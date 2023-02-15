<<<<<<< HEAD
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_learn.datasets import BatchMetaDataLoader, get_omniglot
from meta_learn.proto.model import ProtoNetwork

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
=======
import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from meta_learn.proto.model import ProtoNetwork
from meta_learn.datasets import BatchMetaDataLoader, get_omniglot


from torch.utils.tensorboard import SummaryWriter
>>>>>>> main

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

<<<<<<< HEAD

=======
>>>>>>> main
def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)
<<<<<<< HEAD

=======
    
>>>>>>> main
    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
<<<<<<< HEAD
    squared_distances = torch.sum(
        (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
    )
    return F.cross_entropy(-squared_distances, targets, **kwargs)


def get_accuracy(prototypes, embeddings, targets):
    sq_distances = torch.sum(
        (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
    )
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())


=======
    squared_distances = torch.sum((prototypes.unsqueeze(2)
        - embeddings.unsqueeze(1)) ** 2, dim=-1)
    return F.cross_entropy(-squared_distances, targets, **kwargs)

def get_accuracy(prototypes, embeddings, targets):
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())

>>>>>>> main
def train(dataloader, model, optimizer, num_batches):
    avg_loss = list()

    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

<<<<<<< HEAD
            train_inputs, train_targets = batch["train"]
=======
            train_inputs, train_targets = batch['train']
>>>>>>> main
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

<<<<<<< HEAD
            test_inputs, test_targets = batch["test"]
=======
            test_inputs, test_targets = batch['test']
>>>>>>> main
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets, 5)
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx > num_batches:
                break

    return np.mean(avg_loss)


def test(dataloader, model, num_batches):
    # Evaluate model
    accuracy_list = list()
    avg_loss = list()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            model.zero_grad()

<<<<<<< HEAD
            train_inputs, train_targets = batch["train"]
=======
            train_inputs, train_targets = batch['train']
>>>>>>> main
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

<<<<<<< HEAD
            test_inputs, test_targets = batch["test"]
=======
            test_inputs, test_targets = batch['test']
>>>>>>> main
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets, 5)
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)
            avg_loss.append(loss.detach().item())

            accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
            accuracy_list.append(accuracy.item())

            if batch_idx > num_batches:
                break

    return np.mean(accuracy_list), np.mean(avg_loss)


def main(config):
    bs = config.batch_size
    way = config.num_ways
    shot = config.num_shots
    epochs = config.epochs
    path = config.path
    num_batches = config.num_batches
    logdir = "{}/{}/{}_way_{}_shot".format(path, "model", config.num_ways, 1)
    writer = SummaryWriter(logdir)

    parent_dir = os.path.abspath(os.path.join(path, os.pardir))
    data_path = f"{parent_dir}/data/"
    download = True
    if os.path.exists(f"{data_path}/omniglot"):
        download = False

<<<<<<< HEAD
    train_dataset, test_dataset = get_omniglot(data_path, way, download)
    trainloader = BatchMetaDataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=4
    )
    testloader = BatchMetaDataLoader(
        test_dataset, batch_size=bs, shuffle=True, num_workers=4
    )

    model = ProtoNetwork(1, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
=======

    train_dataset, test_dataset = get_omniglot(data_path, way, download)
    trainloader = BatchMetaDataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    testloader = BatchMetaDataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=4)

    model = ProtoNetwork(1, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
>>>>>>> main
    model.to(device=device)

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, optimizer, num_batches)

        # Test neural network
        if t % config.log_every == 0:
            test_accuracy, test_loss = test(testloader, model, num_batches)
<<<<<<< HEAD
            print(
                f"Train Loss: {train_loss}    Test Loss: {test_loss}      Test Acc: {test_accuracy}"
            )
=======
            print(f"Train Loss: {train_loss}    Test Loss: {test_loss}      Test Acc: {test_accuracy}")
>>>>>>> main
            writer.add_scalar("Train Loss", train_loss, t)
            writer.add_scalar("Test Loss", test_loss, t)
            writer.add_scalar("Meta-Test Accuracy", test_accuracy, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ways", type=int, default=5)
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--path", type=str, default=os.getcwd())
    main(parser.parse_args())
