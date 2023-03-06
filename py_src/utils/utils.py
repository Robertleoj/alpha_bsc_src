import os
from config import dynamic_window, buffer_generations, endgame_training
import torch
from dataclasses import dataclass

@dataclass
class Data:
    states: torch.Tensor
    policies: torch.Tensor
    outcomes: torch.Tensor
    moves_left: torch.Tensor


def make_folder(path):
    """Create a folder if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def training_gens(generation: int):
    if dynamic_window.on:
        return dynamic_window_gen(generation)
    else:
        return buffer_generations


def dynamic_window_gen(generation: int):
    """Returns the number of training generations based on the current generation"""
    if dynamic_window.on:
        return min(
                dynamic_window.max,
                dynamic_window.min + generation // dynamic_window.increase_every,
            )
    else:
        raise RuntimeError("Dynamic window is off!")
        