import torch
from training import get_dataloader
from DB import DB

model = torch.jit.load("../models/connect4/0.pt")

db = DB()

dl = get_dataloader(db, 'connect4', 0)

for states, policies, outcomes in dl:
    # states = states.to('cuda')
    # policies = policies.to('cuda')
    # outcomes = outcomes.to('cuda')
    print(f"{states.shape}")
    print(f"{policies.shape}")
    print(f"{outcomes.shape}")
    break



# inp = torch.randn((1, 3, 6, 7))kk

# out = model(inp)

# p, v = out

# print(p.shape)
# print(p)
# print(v.shape)
# print(v)


# model = Connect4NN()