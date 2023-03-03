#This code is a modified version of: https://github.com/deepmind/neural-processes

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

class DeterministicEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(DeterministicEncoder, self).__init__()
        
        #TODO: Fix input size to account for concated data
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        self.context_projection = nn.Linear(1, hidden_size)
        self.target_projection = nn.Linear(1, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 8)
        
    def forward(self, context_x, context_y, target_x):
        #Concat important information together
        #NOTE: The code only support batche sizes of one, which need to be expanded
        #to work with pytorch
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0)
        target_x = target_x.unsqueeze(0)
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

        #Aggregation is replace with mutlihead attention
        context_x = self.context_projection(context_x)
        target_x = self.target_projection(target_x)
        attn_output, attn_weights = self.attention(context_x, target_x, x)

        return attn_output

class LatentEncoder(nn.Module):
    def __init__(self, hidden_size=128, num_latents=2):
        super(LatentEncoder, self).__init__()
        
        #TODO: Fix input size to account for concated data
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        #Additional MLP to learn Gaussian params
        small_hidden = int((hidden_size+num_latents)/2)
        self.fc5 = nn.Linear(hidden_size, small_hidden)
        self.mu_layer = nn.Linear(small_hidden, num_latents)
        self.sigma_layer = nn.Linear(small_hidden, num_latents)
        
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

        #Layer that reduces to parameters of Gaussian
        x = F.relu(self.fc5(x))
        mu = self.mu_layer(x)
        log_sigma = self.sigma_layer(x)
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)

        return Normal(mu, sigma)

class Decoder(nn.Module):
    def __init__(self, hidden_size=128):
        super(Decoder, self).__init__()
        
        #The extra one comes from when you concat target values
        self.fc1 = nn.Linear(hidden_size+1+10, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, representation, target_x, shots):
        #Concat representation and target values
        target_x = target_x.unsqueeze(0)
        x = torch.concat([representation, target_x], dim=-1)

        #Forward pass through the decoder
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Calculate mu and sigma of decoded values,
        # across appropriate dimension
        # mu = x[:,:,0]
        # log_sigma = x[:,:,1]

        #Bound the variance
        # sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)
        return x

    
class ANP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANP, self).__init__()
        self.det_encoder = DeterministicEncoder(input_size, hidden_size)
        self.lat_encoder = LatentEncoder(hidden_size, num_latents=10)
        self.decoder = Decoder(hidden_size)
        self.mse_loss = nn.MSELoss()

        total_params = 0
        for model in [self.det_encoder, self.lat_encoder, self.decoder]:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            total_params += sum([np.prod(p.size()) for p in model_parameters])
        print(f"Num params: {total_params}")

        #These values get updated during training
        self.mu = 0
        self.sigma = 1
    
    def forward(self, context_x, context_y, target_x, target_y=None):
        #Encode
        prior = self.lat_encoder(context_x, context_y)
        latent_rep = None
        if target_y is None:
            latent_rep = prior.sample((1, context_x.shape[0]))
        else:
            posterior = self.lat_encoder(target_x, target_y)
            latent_rep = posterior.sample((1, context_x.shape[0]))

        latent_rep = latent_rep.squeeze(2)
        deter_rep = self.det_encoder(context_x, context_y, target_x)
        representation = torch.concat([deter_rep, latent_rep], dim=-1)

        #Decode
        y_pred = self.decoder(representation, target_x, context_x.shape[0])   

        # get log probability
        target_y = target_y.unsqueeze(0)
        mse = self.mse_loss(y_pred, target_y)
        
        # get KL divergence between prior and posterior
        kl = self.kl_div(prior.mean, prior.stddev, posterior.mean, posterior.stddev)
        
        # maximize prob and minimize KL divergence
        loss = mse + kl
        return y_pred, kl, loss
        
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div