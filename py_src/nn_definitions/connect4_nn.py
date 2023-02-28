from .nn_base import Middle
from torch import nn
import torch


ROWS = 6
COLS = 7

class Connect4ValueHead(nn.Module):
    def __init__(self, inp_channels) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(inp_channels * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        # output is 1
    def forward(self, x):
        return self.out(x)

class Connect4PolicyHead(nn.Module):
    def __init__(self, inp_channels) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(inp_channels * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, COLS)
        )
       
    def forward(self, x):
        return self.out(x)


class Connect4NN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input is 2 x 6 x 7
        pad = nn.ZeroPad2d((0, 0, 1, 0))

        inp_out_channels = 8

        inp_nn = nn.Conv2d(
            in_channels = 2,
            out_channels = inp_out_channels,
            kernel_size = (3, 3),
            padding=(1, 1)
        )

        self.inp = nn.Sequential(
            pad, inp_nn
        )

        middle_out_channels = 16

        self.middle = Middle([
            inp_out_channels, 
            64, 64, 64, 64, 64, 64,
            middle_out_channels
        ])
        # self.middle = Middle([6, 24, 12])

        self.policy_head = Connect4PolicyHead(middle_out_channels)
        self.value_head = Connect4ValueHead(middle_out_channels)

    def forward(self, x):
        # Input has shape 2 x ROWS x COLS
        x = self.inp(x)
        x = self.middle(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return (policy, value.squeeze(-1))

