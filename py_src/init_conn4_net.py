from nn_definitions import Connect4NN
import torch
from pathlib import Path
from sys import argv


if len(argv) != 2:
    print("Usage: python3 init_conn4_net.py <path_to_save_model>")
    exit(1)

path = Path(argv[1])

if not path.exists():
    print("Path does not exist")
    exit(1)

mdl = Connect4NN()
print("Initialized model")

serialized = torch.jit.script(mdl)
serialized.save(path / "0.pt")
