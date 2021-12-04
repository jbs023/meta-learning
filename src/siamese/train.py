import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import errno
import argparse
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
# from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Resize

from dataset import *
from model import *

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = '../../models/siamese'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def train(dataloader, model, loss_fn, optimizer, suffix):
    global PATH
    size = len(dataloader.dataset)

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

        # print("Iteration Duration: {}".format(time.time()-start_time))
        if i % 10 == 0:
            t_loss, current = loss.item(), i * len(x1)
            avg_loss.append(t_loss)
            print(f"loss: {t_loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Save every N iterations
            torch.save(net.state_dict(), PATH+"/siamese-train-{}.pth".format(suffix))
    
    torch.save(net.state_dict(), PATH+"/siamese-train-{}.pth".format(suffix))
    return np.mean(avg_loss)

def test(testloader, net):   
    # Evaluate model 
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

        print('*'*70)
        print('\t\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(right, error, right*1.0/(right+error)))
        print('*'*70)

        return right*1.0/(right+error)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments for the swarm consensus simulation.')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Parameter to determine test vs train')    
    parser.add_argument('-d', '--distortion', action='store_true',
                        default=False, required=False, help='Apply the 8 affine distortions to each image')
    parser.add_argument('-c', '--convolutional', action='store_true', default=False, required=False, 
                        help='flag to choose the Siamese CNN instead of the default feed foward network')
    parser.add_argument('-n', '--num_examples', default=None, required=False,
                        help='Number of pairs of images. If none, it runs one example pair for every image (19280)')
    parser.add_argument('-s', '--file_suffix', default="net", required=False, help='suffix for the output files')

    args = parser.parse_args()
    num_examples = args.num_examples
    if not (num_examples is None): 
        num_examples = int(args.num_examples)

    distortions = args.distortion
    suffix = args.file_suffix
    train_path = "../../data/images_background"
    test_path = "../../data/images_evaluation"
    batch_size = 128
    way = 5
    epochs = 300 if not args.convolutional else 200

    transforms = Compose([Resize(28), ToTensor()])

    if args.train:
        #Set up training dataset and neural network 
        trainSet = OmniglotTrain(train_path, transform=transforms, distortions=distortions, 
                                num_examples=num_examples)
        trainloader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=4)
        net = SiameseNetwork() if not args.convolutional else SiameseCNN()
        if torch.cuda.is_available(): 
            net.cuda()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=6e-5, weight_decay=1e-3)
        track_loss = list()

        #Train network
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(trainloader, net, loss_fn, optimizer, suffix)

            #Save loss to a file
            track_loss.append(loss)
            df = pd.DataFrame(track_loss, columns=['Loss'])
            df.to_csv("siamese_final_loss_{}.csv".format(suffix))

        print("Done!")
        print('Finished Training')

        #Save loss over time and network weights
        filePath = PATH+"/siamese-final-{}.pth".format(suffix)
        torch.save(net.state_dict(), filePath)
        print("Saved PyTorch Model State to {}".format(filePath))
    else:
        #Test neural network
        testSet = OmniglotTest(test_path, transform=transforms, times=400, way=way)
        testloader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=4)

        if os.path.exists(PATH+"/siamese-train-{}.pth".format(suffix)):
            net = SiameseNetwork() if not args.convolutional else SiameseCNN()
            if torch.cuda.is_available(): net.cuda()
            net.load_state_dict(torch.load(PATH+"/siamese-train-{}.pth".format(suffix)))

            results = list()
            for i in range(0, 10):
                print("Trial {}".format(i))
                result = test(testloader, net)
                results.append(result)

            mean = np.mean(results)
            std = np.std(results)
            min = np.min(results)
            max = np.max(results)
            print("Final Results: {} +- {} ({} - {})".format(mean, std, min, max))

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PATH)
