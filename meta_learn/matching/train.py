import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from meta_learn.datasets import BatchMetaDataLoader, omniglot
from meta_learn.matching.model import MatchingNetwork

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def pairwise_cosine_similarity(embeddings1, embeddings2, eps=1e-8):
    sq_norm1 = torch.sum(embeddings1 ** 2, dim=2, keepdim=True)
    sq_norm2 = torch.sum(embeddings2 ** 2, dim=2).unsqueeze(1)
    dot_product = torch.bmm(embeddings1, embeddings2.transpose(1, 2))
    inverse_norm = torch.rsqrt(torch.clamp(sq_norm1 * sq_norm2, min=eps ** 2))
    return dot_product * inverse_norm

def matching_log_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    batch_size, num_samples, _ = test_embeddings.shape
    similarities = pairwise_cosine_similarity(embeddings, test_embeddings, eps=eps)
    logsumexp = torch.logsumexp(similarities, dim=1, keepdim=True)

    max_similarities, _ = torch.max(similarities, dim=1, keepdim=True)
    exp_similarities = torch.exp(similarities - max_similarities)

    sum_exp = exp_similarities.new_zeros((batch_size, num_classes, num_samples))
    indices = targets.unsqueeze(-1).expand_as(exp_similarities)
    sum_exp.scatter_add_(1, indices, exp_similarities)

    return torch.log(sum_exp) + max_similarities - logsumexp

def train(dataloader, model, optimizer, num_batches):
    avg_loss = list()

    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            logits = matching_log_probas(train_embeddings,
                                        train_targets,
                                        test_embeddings,
                                        5)
            loss = F.nll_loss(logits, test_targets)

            # Compute prediction error
            avg_loss.append(loss.detach().item())

            # Backpropagation
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
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            logits = matching_log_probas(train_embeddings,
                                        train_targets,
                                        test_embeddings,
                                        5)
            loss = F.nll_loss(logits, test_targets)

            avg_loss.append(loss.detach().item())

            test_predictions = torch.argmax(logits, dim=1)
            accuracy = torch.mean((test_predictions == test_targets).float())
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
    logdir = "{}/{}/_{}_{}".format(path, "model", config.num_ways, 1)
    writer = SummaryWriter(logdir)

    parent_dir = os.path.abspath(os.path.join(path, os.pardir))
    data_path = f"{parent_dir}/data/"
    download = True
    if os.path.exists(f"{data_path}/omniglot"):
        download = False

    train_dataset = omniglot(data_path,
                       shots=shot,
                       ways=way,
                       shuffle=True,
                       meta_train=True,
                       download=download)
    test_dataset = omniglot(data_path,
                       shots=shot,
                       ways=way,
                       shuffle=True,
                       meta_val=True,
                       download=download)
    trainloader = BatchMetaDataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    testloader = BatchMetaDataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=4)

    model = MatchingNetwork(1, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device=device)

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, optimizer, num_batches)

        # Test neural network
        if t % config.log_every == 0:
            test_accuracy, test_loss = test(testloader, model, num_batches)
            print(f"Train Loss: {train_loss}    Test Loss: {test_loss}      Test Acc: {test_accuracy}")
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
