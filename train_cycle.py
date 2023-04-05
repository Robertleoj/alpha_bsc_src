#!/bin/python3
from sys import argv
import os
from pathlib import Path


if len(argv) <= 1:
    print("Usage: python3 train_cycle.py <run_name> [<num_cycles> <game_name>]")
    exit(1)

run_name = Path(argv[1])
game_name = 'connect4'


num_cycles = 10
if len(argv) >= 3:
    num_cycles = int(argv[2])

if len(argv) >= 4:
    game_name = argv[3]

run_dir = Path(f'./vault/{game_name}/{run_name}')

if not run_dir.exists():
    print("Run directory does not exist")
    exit(1)

use_eval = game_name == 'connect4'

def self_play():
    os.chdir('./cpp_src')
    cmd = f'./self_play {run_name} {game_name}'
    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    if(exit_code != 0):
        exit(0)

    os.chdir('..')

def train():
    os.chdir('./py_src')
    cmd = f'python3 train.py {run_name} {game_name}'
    print(f"Running: {cmd}")
    exit_code = os.system(cmd)
    if(exit_code != 0):
        exit(0)
    os.chdir('..')

def evaluate():
    if not use_eval:
        return
    os.chdir('./cpp_src')
    exit_code = os.system(f'./eval_agent {run_name}')
    if(exit_code != 0):
        exit(0)
    os.chdir('..')


for i in range(num_cycles):
    print(f"Cycle {i+1}/{num_cycles}")
    evaluate()
    self_play()
    train()

evaluate()
    
