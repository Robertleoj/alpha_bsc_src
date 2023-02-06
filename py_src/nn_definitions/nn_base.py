import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x) + x



class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = lambda:  nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3,
            padding=1
        )
        bn = lambda: nn.BatchNorm2d(channels)
        
        self.out = nn.Sequential(
            bn(), nn.ReLU(), conv(),
            bn(), nn.ReLU(), conv()
        )

    def forward(self, x):
        return self.out(x)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.out = Residual(ResBlock(channels))

    def forward(self, x):
        return self.out(x)

class ChannelChange(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return self.out(x)

class Middle(nn.Module):

    def __init__(self, channel_list):
        super().__init__()
        resblocks = [
            ResNetBlock(c) for c in channel_list
        ]

        chann_changes = [
            ChannelChange(cin, cout) 
            for (cin, cout) in zip(channel_list, channel_list[1:])
        ]

        out = []
        for(res, cchange) in zip(resblocks, chann_changes):
            out.append(res)
            out.append(cchange)

        out.append(resblocks[-1])

        self.out = nn.Sequential(*out)

    def forward(self, x):
        return self.out(x)
    
        
