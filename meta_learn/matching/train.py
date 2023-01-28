import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from meta_learn.matching.model import MatchingNetwork
from meta_learn.utils import get_meta_omniglot

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

def train(dataloader, model, way, optimizer):
    avg_loss = list()
    # for batch in tqdm(dataloader):
    #     support_x, support_y = batch["support"]  # (batch_size, way, shot, 28, 28)
    #     query_x, query_y = batch["query"]  # (batch_size, 1, shot, 28, 28)

    #     support_x, support_y = support_x.to(device), support_y.to(device)
    #     query_x, query_y = query_x.to(device), query_y.to(device)

    #     # Compute prediction error
    #     support_embedded = model(support_x)
    #     query_embedded = model(query_x)
    #     logits = matching_log_probas(support_embedded,
    #                                  support_y,
    #                                  query_embedded,
    #                                  way)
    #     loss = F.nll_loss(logits, query_y)

    #     # Backpropagation
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     # Log performance
    #     avg_loss.append(loss.detach().item())
    with tqdm(dataloader, total=500) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['support']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['query']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            logits = matching_log_probas(train_embeddings,
                                        train_targets,
                                        test_embeddings,
                                        way)
            loss = F.nll_loss(logits, test_targets)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # calculate the accuracy
                log_probas = matching_log_probas(train_embeddings,
                                                 train_targets,
                                                 test_embeddings,
                                                 way)
                test_predictions = torch.argmax(log_probas, dim=1)
                accuracy = torch.mean((test_predictions == test_targets).float())
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx > 500:
                break

    return np.mean(avg_loss)


def test(dataloader, model, way):
    # Evaluate model
    with torch.no_grad():
        accuracy_list = list()
        avg_loss = list()
        for i, batch in enumerate(dataloader):
            support_x, support_y = batch["support"]  # (batch_size, way, shot, 28, 28)
            query_x, query_y = batch["query"]  # (batch_size, 1, shot, 28, 28)

            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # Compute prediction error
            support_embedded = model(support_x)
            query_embedded = model(query_x)
            logits = matching_log_probas(support_embedded,
                                        support_y,
                                        query_embedded,
                                        way)
            loss = F.nll_loss(logits, query_y)
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

    model = MatchingNetwork(1, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if torch.cuda.is_available():
        model.cuda()

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(trainloader, model, way, optimizer)

        # Test neural network
        if t % config.log_every == 0:
            test_accuracy, test_loss = test(testloader, model, way)
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
