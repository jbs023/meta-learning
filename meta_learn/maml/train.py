from collections import OrderedDict
import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from meta_learn.maml.model import ConvMaml
from meta_learn.datasets import BatchMetaDataLoader, omniglot


from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params


def train(dataloader, model, optimizer, num_batches, batch_size, step_size):
    avg_loss = list()
    avg_acc = list()

    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)

            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=step_size,
                                                    first_order=True)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(batch_size)
            accuracy.div_(batch_size)

            outer_loss.backward()
            optimizer.step()

            avg_loss.append(outer_loss.detach().item())
            avg_acc.append(accuracy.item())
            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx > num_batches:
                break

    return np.mean(avg_acc), np.mean(avg_loss)


def test(dataloader, model, num_batches, batch_size, step_size):
    # Evaluate model
    accuracy_list = list()
    avg_loss = list()
    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)

            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=step_size,
                                                    first_order=True)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

                accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(batch_size)
            accuracy.div_(batch_size)
            avg_loss.append(outer_loss.detach().item())
            accuracy_list.append(accuracy.item())

            if batch_idx > num_batches:
                break

    return np.mean(accuracy_list), np.mean(avg_loss)


def main(config):
    bs = config.batch_size
    ss = config.step_size
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

    model = ConvMaml(1, way, 64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device=device)

    # Train network
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train(trainloader, model, optimizer, num_batches, bs, ss)

        # Test neural network
        if t % config.log_every == 0:
            test_accuracy, test_loss = test(testloader, model, num_batches, bs, ss)
            print(f"Train Loss: {train_loss}    Train Acc: {train_acc}      Test Loss: {test_loss}      Test Acc: {test_accuracy}")
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
    parser.add_argument('--step-size', type=float, default=0.4)
    main(parser.parse_args())
