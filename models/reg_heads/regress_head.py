import torch
from torch import nn

class MLP_head(nn.Module):

    def __init__(self, h_dim):
        super().__init__()
        self.activation = nn.ReLU()

        self.layer1 = nn.Linear(h_dim, h_dim//2)
        self.layer2 = nn.Linear(h_dim//2, h_dim//4)
        self.layer3 = nn.Linear(h_dim//4, 1)

    def forward(self, x):

        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        # output = self.softmax(self.layer3(x))
        output = self.layer3(x)

        return output