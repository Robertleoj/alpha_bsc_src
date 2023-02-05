from .nn_base import NNBase
from torch import nn
import torch


ROWS = 7
COLS = 7

class Connect4NN(NNBase):
    def __init__(self):
        super().__init__()
        
        # input is 3 x 7 x 7
        self.inp_nn = nn.Conv2d(
            in_channels = 3,
            out_channels = 3,
            kernel_size = 2,
            padding=(1, 1)
        )
        # output is 3 x 8 x 8
        
        self.out = nn.Linear(3 * 8 * 8, COLS)

    def forward(self, x):
        # Input has shape 3 x ROWS x COLS
        x = self.inp_nn(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

    
