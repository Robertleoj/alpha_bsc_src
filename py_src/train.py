import torch
from training import get_dataloader, loss_fn
from DB import DB
import config
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda'


model = torch.jit.load("../models/connect4/0.pt").to(device)
db = DB()
dl = get_dataloader(db, 'connect4', 0)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

iteration = 0

losses = []

for states, policies, outcomes in dl:
    states = states.to(device)
    policies = policies.to(device)
    outcomes = outcomes.to(device)
    
    optimizer.zero_grad()
    
    nn_pol, nn_val = model(states)
    torch.log_softmax(nn_pol, 1)

    # Print shapes
    # print("states")
    # print(states.shape)

    # print("values")
    # print(nn_val.shape)
    # print(outcomes.shape)

    # print("policies")
    # print(nn_pol.shape)
    # print(policies.shape)
    


    loss = loss_fn(nn_val, nn_pol, outcomes, policies)    
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


    if iteration % 100 == 0:
        print(f"[ {iteration + 1} / {config.num_iterations} ]loss={loss.item()}")
    
    iteration += 1

    if iteration > config.num_iterations:
        break
    

    
plt.plot(losses)
plt.savefig("./figures/losses.png")
# plt.show()

# torch.jit.save(model, "../models/connect4/1.pt")