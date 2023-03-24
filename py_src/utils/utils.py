import os
from config import config
import torch
from dataclasses import dataclass
import sys
import os

@dataclass
class Data:
    states: torch.Tensor
    policies: torch.Tensor
    outcomes: torch.Tensor
    moves_left: torch.Tensor
    weights: torch.Tensor


def make_folder(path):
    """Create a folder if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def training_gens(generation: int):
    dw_config = config['dynamic_window']
    if dw_config['on']:
        return dynamic_window_gen(generation)
    else:
        return config['buffer_generations']


def dynamic_window_gen(generation: int):
    """Returns the number of training generations based on the current generation"""

    dw_config = config['dynamic_window']
    
    if dw_config['on']:
        return min(
                dw_config['max'],
                dw_config['min'] + generation // dw_config['increase_every'],
            )
    else:
        raise RuntimeError("Dynamic window is off!")
        
def set_run(run_name, game):

    if not os.path.exists(f"../vault/{game}/{run_name}"):
        print("Run does not exist! Exiting")
        exit(1)

    os.chdir(f"../vault/{game}/{run_name}")
    config.initialize("py_config.json")

