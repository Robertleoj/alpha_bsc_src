from .nn_base import Middle
from torch import nn
import torch


ROWS = 6
COLS = 7

class Connect4ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(12 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # output is 1
    def forward(self, x):
        return self.out(x)

class Connect4PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(12 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, COLS)
        )
       
    def forward(self, x):
        return self.out(x)


class Connect4NN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input is 3 x 6 x 7
        pad = nn.ZeroPad2d((0, 0, 1, 0))

        inp_nn = nn.Conv2d(
            in_channels = 3,
            out_channels = 6,
            kernel_size = (3, 3),
            padding=(1, 1)
        )

        self.inp = nn.Sequential(
            pad, inp_nn
        )


        #self.middle = Middle([6, 24, 24, 24, 24, 24, 24, 12])
        self.middle = Middle([6, 24, 12])

        self.policy_head = Connect4PolicyHead()
        self.value_head = Connect4ValueHead()

    def forward(self, x):
        # Input has shape 3 x ROWS x COLS
        x = self.inp(x)
        x = self.middle(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return (policy, value.squeeze(-1))

