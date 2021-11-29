import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11025, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        
        self.fc4 = nn.Linear(2048,1) # check numbers

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, 1)# flatten x1 
        x2 = torch.flatten(x2, 1)# flatten x2

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))
        x2 = F.relu(self.fc3(x2))

        distance = torch.abs(x1 - x2)
        x = torch.sigmoid(self.fc4(distance))
        return x.reshape((x.shape[0],1))

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