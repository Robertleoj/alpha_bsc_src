from nn_definitions import Connect4NN, BreakthroughNN
import torch
from pathlib import Path
from sys import argv


if len(argv) < 2:
    print("Usage: python3 init_conn4_net.py <path_to_save_model>")
    exit(1)

path = Path(argv[1])

if not path.exists():
    print("Path does not exist")
    exit(1)

game = 'connect4'
if len(argv) > 2:
    game = argv[2]


if game == 'connect4':
    mdl = Connect4NN()
elif game == 'breakthrough':
    mdl = BreakthroughNN()
else:
    print("Game not supported")
    exit(1)

print("Initialized model")

serialized = torch.jit.script(mdl)
serialized.save(path / "0.pt")
