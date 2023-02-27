from nn_definitions import Connect4NN
import torch
from pathlib import Path

mdl = Connect4NN()

# traced = torch.jit.trace(mdl, torch.randn(5, 3, 6, 7))

serialized = torch.jit.script(mdl)
path = Path("../models/connect4")
path.mkdir(parents=True, exist_ok=True)
serialized.save("../models/connect4/0.pt")
