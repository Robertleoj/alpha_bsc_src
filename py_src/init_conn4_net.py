from nn_definitions import Connect4NN
import torch

mdl = Connect4NN()

traced = torch.jit.trace(mdl, torch.randn(5, 3, 6, 7))

#serialized = torch.jit.script(mdl)
traced.save("../models/connect4/0.pt")
