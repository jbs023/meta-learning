import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y, num_classes=5):
    # x: Batch_size x Way x Dim
    # y: Batch_size x (Way*Shot) x Dim

    #Get X and Y to be the same shape
    x = x.unsqueeze(2).repeat(1,1,y.size(1),1)
    y = y.unsqueeze(1).repeat(1,num_classes,1,1)
    
    dist = torch.pow(x - y, 2).sum(3)
    return dist

def get_prototypes(embeddings, num_classes=5):
    batch_size, embeddings_size = embeddings.size(0), embeddings.size(-1)

    #Move batches from row into a matrix
    num_samples = int(embeddings.size(1)/num_classes)
    new_embeddings = embeddings.view((batch_size, num_classes, num_samples, embeddings_size))
    prototypes = torch.mean(new_embeddings, dim=1)

    return prototypes

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ) 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ProtoNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(ProtoNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels),
            Flatten()
        )

    def forward(self, train_inputs, train_targets, test_inputs):
        train_embedding = self.encoder(train_inputs.view(-1, *train_inputs.shape[2:]))
        train_embedding = train_embedding.view(*train_inputs.shape[:2], -1)

        test_embedding = self.encoder(test_inputs.view(-1, *test_inputs.shape[2:]))
        test_embedding = test_embedding.view(*test_inputs.shape[:2], -1)

        prototypes = get_prototypes(train_embedding, 5)
        dist = euclidean_dist(prototypes, test_embedding)
        return F.log_softmax(-dist, dim=1)