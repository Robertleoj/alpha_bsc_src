import os
from config import config
import torch
from dataclasses import dataclass
import sys
import os
import shutil
from pathlib import Path

@dataclass
class Data:
    state: torch.Tensor
    policy: torch.Tensor
    outcome: torch.Tensor
    moves_left: torch.Tensor
    weight: torch.Tensor

@dataclass
class CompetitionResultPlayer:
    white: list[float]
    black: list[float]

@dataclass 
class CompetitionResult:
    p1: CompetitionResultPlayer
    p2: CompetitionResultPlayer

def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def sevenzip_cmd():
    possibilities = ['7z', '7zz']
    for p in possibilities:
        if cmd_exists(p):
            return p

    raise RuntimeError("No compression program")


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

    run_path = Path(f"../vault/{game}/{run_name}")
    print(f"Setting run to {run_name} for game {game}, path {run_path}")

    if not run_path.exists():
        print("Run does not exist! Exiting")
        exit(1)


    os.chdir(run_path)
    config.initialize("py_config.json")

