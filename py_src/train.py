from training import Model
from utils import set_run
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python3 train.py <run_name> [<game>]")

run_name = sys.argv[1]

game = 'connect4'

if(len(sys.argv) == 3):
    game = sys.argv[2]



set_run(run_name, game)

model = Model('connect4')
model.train()
model.save_and_next_gen()
