from .nn_base import Middle, ResNetBlock, ChannelChange
from torch import nn



"""
board is an 2x8x8, where the first channel is the player's pieces, and the second channel is the opponent's pieces.
"""


BSIZE = 8

class BreakthroughValueHead(nn.Module):
    def __init__(self, inp_channels) -> None:
        super().__init__()
        self.out = nn.Sequential(
            ChannelChange(inp_channels, 16),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        # output is 1
    def forward(self, x):
        return self.out(x)

class BreakthroughPolicyHead(nn.Module):
    def __init__(self, inp_channels, bsize) -> None:
        super().__init__()
        self.bsize = bsize
        self.out = nn.Sequential(
            ChannelChange(inp_channels, bsize ** 2) if inp_channels != 64 else nn.Identity(),
            ResNetBlock(64),
            ResNetBlock(64),
            ResNetBlock(64),
        )
        
    def forward(self, x):
        x = self.out(x)
        return x.reshape(-1, int(self.bsize ** 2), int(self.bsize ** 2))


class BreakthroughNN(nn.Module):
    def __init__(self, bsize=BSIZE):
        super().__init__()

        
        middle_out_channels = 64

        self.middle = Middle([
            2, 16, 32, 64, 128, 128, 128, 64, 
            middle_out_channels
        ])

        self.policy_head = BreakthroughPolicyHead(middle_out_channels, bsize)
        self.value_head = BreakthroughValueHead(middle_out_channels)

    def forward(self, x):
        # Input has shape 2 x ROWS x COLS
        x = self.middle(x)
        pol = self.policy_head(x)
        val = self.value_head(x)
        return (pol, val.squeeze(-1))