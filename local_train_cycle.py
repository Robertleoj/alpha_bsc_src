from sys import argv
import os

num_cycles = 10

if len(argv) >= 2:
    num_cycles = int(argv[1])


def self_play():
    os.chdir('./cpp_src')
    os.system('./self_play')
    os.chdir('..')

def train():
    os.chdir('./py_src')
    os.system('python3 train.py')
    os.chdir('..')

for i in range(num_cycles):
    print(f"Cycle {i+1}/{num_cycles}")
    self_play()
    train()
    
