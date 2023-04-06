from nn_definitions import Connect4NN, BreakthroughNN
import torch
from pathlib import Path
from sys import argv
from torchsummary import summary


if len(argv) < 2:
    print("Usage: python3 init_conn4_net.py <path_to_save_model> [<game_name>]")
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

    summary(mdl, (2, 6, 7))
    
elif game == 'breakthrough':
    mdl = BreakthroughNN()
    
    summary(mdl, (2, 8, 8))

else:
    print("Game not supported")
    exit(1)

# mdl.eval()

print("Initialized model")

serialized = torch.jit.script(mdl)
serialized.save(path / "0.pt")
