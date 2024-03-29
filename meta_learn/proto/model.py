import torch.nn as nn

<<<<<<< HEAD

=======
>>>>>>> main
def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
<<<<<<< HEAD
        nn.MaxPool2d(2),
    )


=======
        nn.MaxPool2d(2)
    )

>>>>>>> main
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
<<<<<<< HEAD
            conv3x3(hidden_size, out_channels),
=======
            conv3x3(hidden_size, out_channels)
>>>>>>> main
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)
