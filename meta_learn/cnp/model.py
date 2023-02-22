#This code is a modified version of: https://github.com/deepmind/neural-processes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class CNP_encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(CNP_encoder, self).__init__()
        
        #TODO: Fix input size to account for concated data
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, context_x, context_y):
        #Concat important information together
        #NOTE: The code only support batche sizes of one, which need to be expanded
        #to work with pytorch
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        x = torch.cat([context_x, context_y], dim=-1)

        #NOTE: This is for conceptual clarity not computational efficiency.
        #This loop can definitely be vectorized.

        #Encode each shot (i.e., element of the support set)
        data = list()
        for idx in range(x.shape[1]):
            value = x[:, idx]
            value = F.relu(self.fc1(value))
            value = F.relu(self.fc2(value))
            value = F.relu(self.fc3(value))
            value = self.fc4(value)
            data.append(value.unsqueeze(1))

        #Collect each encoded value into a single tensor
        x = torch.cat(data, dim=1)

        # Aggregator: take the mean over all shots
        x = torch.mean(x, dim=1)
        return x
    
class CNP_decoder(nn.Module):
    def __init__(self, hidden_size=128):
        super(CNP_decoder, self).__init__()
        
        #The extra one comes from when you concat target values
        self.fc1 = nn.Linear(hidden_size+1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        
    def forward(self, representation, target_x, shots):
        #Expand the representation, so it can be used for each shot
        representation = torch.tile(representation, (1, shots, 1))

        #Concat representation and target values
        target_x = target_x.unsqueeze(0)
        x = torch.concat([representation, target_x], dim=-1)

        #Forward pass through the decoder
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Calculate mu and sigma of decoded values,
        # across appropriate dimension
        mu = x[:,:,0]
        log_sigma = x[:,:,1]

        #Bound the variance
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)
        return mu, sigma

    
class CNP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNP, self).__init__()
        self.encoder = CNP_encoder(input_size, hidden_size)
        self.decoder = CNP_decoder(hidden_size)

        #These values get updated during training
        self.mu = 0
        self.sigma = 1
    
    def forward(self, context_x, context_y, target_x):
        #Encode
        encoded_values = self.encoder(context_x, context_y)
        
        #Decode
        mu, sigma = self.decoder(encoded_values, target_x, context_x.shape[0])        
        return mu, sigma
        
    def loss_fn(self, target_y, target_mask):
        #Calculate loss
        target_y = target_y.squeeze(-1)
        target_mask = target_mask.squeeze(-1)
        mvn = Normal(self.mu, self.sigma)
        loss = -torch.sum(mvn.log_prob(target_y) * target_mask.float())
        return loss