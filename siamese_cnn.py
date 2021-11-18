import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

import os
import errno
import argparse

from torch.utils.data import DataLoader
from dataset import *

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = './cnn_models/'

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x1, x2):
        #Forward pass for net1
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = F.relu(self.conv4(x1))
        x1 = torch.flatten(x1, 1)
        x1 = torch.sigmoid(self.fc1(x1))

        #Forward pass for net2
        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = F.relu(self.conv4(x2))
        x2 = torch.flatten(x2, 1)
        x2 = torch.sigmoid(self.fc1(x2))

        #Distance computation
        x = torch.abs(x1-x2)
        x = torch.sigmoid(self.fc2(x))

        return x.reshape((x.shape[0],1))

def train(dataloader, model, loss_fn, optimizer):
    global PATH
    size = len(dataloader.dataset)

    train_loss = list()
    avg_loss = list()
    for i, data in enumerate(dataloader):
        # print("Iteration: {}".format(i))
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

            #Save every 1000 iterations
            torch.save(net.state_dict(), PATH+"/siamese-train.pth")

        #Check for convergence (i.e loss over the last 20 iterations is the same)
        train_loss.append(loss.item())
        if len(train_loss) >= 20:
            train_loss = train_loss[1:]
        
        diff = list([abs(train_loss[i] - train_loss[i-1]) for i in range(1, len(train_loss))])
        avg_diff = np.average(diff)
        if i >= 20 and avg_diff <= 0.0001:
            #Save model and break
            torch.save(net.state_dict(), PATH+"/siamese-train.pth")

            loss, current = loss.item(), i * len(x1)
            print(f"Converged with a loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return True, np.mean(avg_loss)
    return False, np.mean(avg_loss)

def test(testloader, net):   
    # Evaluate model 
    with torch.no_grad():
        right, error = 0, 0
        for _, (test1, test2) in enumerate(testloader, 1):
            print("Iteration: {}/{}".format(_, len(testloader)))
            test1, test2 = test1.to(device), test2.to(device)
            output = net.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            
            if pred == 0:
                right += 1
            else: 
                error += 1

        print('*'*70)
        print('\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(right, error, right*1.0/(right+error)))
        print('*'*70)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs experiments for the swarm consensus simulation.')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Parameter to determine test vs train')    
    parser.add_argument('-d', '--distorition_flag', action='store_true', default=False,
                        help='Apply the 8 affine distortions to each image')
    parser.add_argument('-n', '--num_examples', default=None,
                        help='Number of pairs of images. If none, it runs one example pair for every image (19280)')

    args = parser.parse_args()
    distortions = args.distorition_flag
    num_examples = int(args.num_examples)
    train_path = "./images_background"
    test_path = "./images_evaluation"
    batch_size = 128
    way = 20
    epochs = 210

    if args.train:
        #Set up training dataset and neural network 
        trainSet = OmniglotTrain(train_path, transform=transforms.ToTensor(), 
                                distortions=distortions, num_examples=num_examples)
        trainloader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=4)

        net = SiameseCNN()
        if torch.cuda.is_available(): net.cuda()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=5e-5, weight_decay=1e-5)
        track_loss = list()

        #Train network
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            converged, loss = train(trainloader, net, loss_fn, optimizer)
            track_loss.append(loss)
            df = pd.DataFrame(track_loss, columns=['Loss'])
            df.to_csv("siamese_final_loss.csv")

            if converged:
                break

        print("Done!")
        print('Finished Training')

        #Save loss over time and network weights
        torch.save(net.state_dict(), PATH+"/siamese-final.pth")
        print("Saved PyTorch Model State to {}".format(PATH+"/siamese-final.pth"))
    else:
        #Test neural network
        testSet = OmniglotTest(test_path, transform=transforms.ToTensor(), times=400, way=way)
        testloader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=4)

        if os.path.exists(PATH+"/siamese-cnn-no-transform.pth"):
            net = SiameseCNN()
            if torch.cuda.is_available(): net.cuda()
            net.load_state_dict(torch.load(PATH+"/siamese-cnn-no-transform.pth"))
            test(testloader, net)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PATH)
