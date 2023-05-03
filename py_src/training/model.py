from config import config
import torch
import pathlib
from DB import DB
from training import loss_fn, get_dataloader
from tqdm import tqdm
from torch import nn

class Model:

    def __init__(self, game: str, device='cuda', generation:int=None, prefetch=True):
        self.device = device
        self.loss_fn = loss_fn

        self.game = game
        self.model_path = pathlib.Path('./models')

        self.db = DB()

        if generation is None:
            self._set_newest_generation()
        else:
            self._set_generation(generation)

    def _set_newest_generation(self) -> None:
        generation = self.db.newest_generation()
        self._set_generation(generation)

    def _set_generation(self, generation:int) -> None:
        self.generation = generation
        self.dl = self.get_dataloader(self.generation)
        self.nn = self.get_checkpoint(self.generation)

    def get_dataloader(self, generation:int) -> torch.utils.data.DataLoader:
        return get_dataloader(generation)

    def get_checkpoint(self, generation:int) -> nn.Module:
        
        path = self.model_path / f"{generation}.pt"

        if not path.exists(): 
            raise FileNotFoundError(f"Model {generation} not found")

        return torch.jit.load(path).to(self.device)
        
        
    def get_optimizer(self) -> torch.optim.Optimizer:
        lr = config['learning_rate']
        if isinstance(lr, dict):
            # get the highest key
            key = str(max(
                [int(g) for g in lr.keys() if int(g) <= self.generation], 
                default=0
            ))
            lr = lr[key]
            # set learning rate

        return torch.optim.Adam(
            self.nn.parameters(), 
            lr=lr,
            weight_decay=config['weight_decay']
        )

    def save_nn(self):
        new_gen_path_train = self.model_path / f"{self.generation}.pt"
        # new_gen_path_infr = self.model_path / f"{self.generation}.pi"
        model = self.nn.eval()
        serialized = torch.jit.script(model)
        serialized.save(new_gen_path_train)
        # optimized = torch.jit.optimize_for_inference(serialized)
        # optimized.save(new_gen_path_infr)

    def add_db_gen(self):
        self.db.add_generation(self.generation)

    def save_and_next_gen(self):
        self.generation += 1
        self.save_nn()
        self.add_db_gen()

        # using class is not allowed after calling this
        del self.__dict__

    def log_loss(self, iteration:int, loss: float):
        self.db.add_loss(self.generation, iteration, loss)
    
    def train(self, num_iterations: int=None):
        if num_iterations is None:
            num_iterations = config['num_iterations']

        iteration = 0

        losses = []
        
        tqdm_dl = tqdm(self.dl, total=num_iterations)

        optimizer = self.get_optimizer()

        for states, policies, outcomes, weights in tqdm_dl:
            states = states.to(self.device)
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)
            weights = weights.to(self.device)
            
            optimizer.zero_grad()
            
            nn_pol, nn_val = self.nn(states)
            nn_pol = nn_pol.reshape(nn_pol.shape[0], -1)
            nn_pol = torch.log_softmax(nn_pol, 1)

            policies = policies.reshape(policies.shape[0], -1)

            loss = loss_fn(nn_val, nn_pol, outcomes, policies, weights)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if iteration % 100 == 0:
                tqdm_dl.set_description(f"loss={loss.item():.4f}")
                tqdm_dl.refresh()
            
            iteration += 1

            if (iteration % config['log_loss_interval']) == 0:
                self.log_loss(
                    iteration=iteration, 
                    loss=sum(losses)/len(losses)
                )

                losses = []

            if iteration > num_iterations:
                break
        
        tqdm_dl.close()


