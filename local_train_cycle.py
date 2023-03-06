#!/bin/python3
from sys import argv
import os

num_cycles = 10

if len(argv) >= 2:
    num_cycles = int(argv[1])


def self_play():
    os.chdir('./cpp_src')
    exit_code = os.system('./self_play')
    if(exit_code != 0):
        exit(0)

    os.chdir('..')

def train():
    os.chdir('./py_src')
    exit_code = os.system('python3 train.py')
    if(exit_code != 0):
        exit(0)
    os.chdir('..')

def evaluate():
    os.chdir('./cpp_src')
    exit_code = os.system('./eval_agent')
    if(exit_code != 0):
        exit(0)
    os.chdir('..')


for i in range(num_cycles):
    print(f"Cycle {i+1}/{num_cycles}")
    evaluate()
    self_play()
    train()
evaluate()
    
