from .nn_base import Middle, ResNetBlock, ChannelChange
from torch import nn



"""
board is an 2x8x8, where the first channel is the player's pieces, and the second channel is the opponent's pieces.
"""


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
    def __init__(self, inp_channels) -> None:
        super().__init__()

        ch = 128

        self.out = nn.Sequential(
            ChannelChange(inp_channels, ch) if inp_channels != ch else nn.Identity(),
            ResNetBlock(ch),
            ResNetBlock(ch),
            ResNetBlock(ch),
            ChannelChange(ch, 3)
        )
        
    def forward(self, x):
        x = self.out(x)
        return x


class BreakthroughNN(nn.Module):
    def __init__(self):
        super().__init__()

        middle_out_channels = 64

        self.middle = Middle([
            2, 128, 128, 128, 128, 128, 128, 128, 
            middle_out_channels
        ])

        self.policy_head = BreakthroughPolicyHead(middle_out_channels)
        self.value_head = BreakthroughValueHead(middle_out_channels)

    def forward(self, x):
        # Input has shape 2 x ROWS x COLS
        x = self.middle(x)
        pol = self.policy_head(x)
        val = self.value_head(x)
        return (pol, val.squeeze(-1))