#Parts of the code borrowed from here: http://cs330.stanford.edu/  
#Specifically: http://cs330.stanford.edu/material/hw1.zip
import copy
import torch
from torch import nn

class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()
        
        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
    
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = torch.nn.LSTM(num_classes + input_size, 
                                    model_size, 
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images
            
            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:
            
            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #Set query label to 0
        local_labels = copy.deepcopy(input_labels)
        local_labels[:,-1] = 0

        #Reshape and concat data
        batch_size = input_images.size(0)
        input_images = input_images.view(batch_size, self.num_classes*(self.samples_per_class+1), self.input_size)
        local_labels = local_labels.view(batch_size, self.num_classes*(self.samples_per_class+1), self.num_classes)
        x = torch.cat([input_images, local_labels], dim=2)

        #Forward Propagate data through the network
        x , _ = self.layer1(x)
        x , _ = self.layer2(x)

        x = x.view(batch_size, self.samples_per_class+1, self.num_classes, self.num_classes)
        return x


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs
            
            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels
                
        Returns:
            scalar loss
        """
        loss_func = nn.CrossEntropyLoss()   
        return loss_func(preds[:,-1, :, :], labels[:, -1, :, :])