import argparse
import os
import torch
import pathlib

from model import MANN
from load_data import DataGenerator
from google_drive_downloader import GoogleDriveDownloader as gdd

from torch.utils.tensorboard import SummaryWriter

# Get cpu or gpu device for training.
path = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    logdir = "{}/{}_{}_{}".format(path, config.logdir, config.num_classes, config.num_samples)
    writer = SummaryWriter(logdir)

    # Download Omniglot Dataset
    omniglot_path = "{}/data/omniglot_resized".format(path)
    if not os.path.isdir(omniglot_path):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path="{}/data/omniglot_resized.zip".format(path),
                                            unzip=True)
    assert os.path.isdir(omniglot_path)

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes, 
                                   config.num_samples, 
                                   data_folder=omniglot_path,
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples, 
                 model_size=config.model_size)
    model = model.double()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1, 
                                        config.num_samples + 1, 
                                        config.num_classes, 
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)
            
            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy', 
                              pred.eq(labels).double().mean().item(),
                              step)
            print("Epoch: {}".format(step))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='run/simple_mann')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())