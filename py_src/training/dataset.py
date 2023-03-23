from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch

from DB.db import DB
import random
from config import config
import utils

from prefetch import load_generations
from utils import Data


class EndgameSampling:
    def __init__(self, data:Data, generation) -> None:
        # hyperparams
        print("Initializing endgame sampling")
        self.p_max = config['endgame_training']['p_max']
        self.p_min = config['endgame_training']['p_min']
        self.shift = config['endgame_training']['shift']
        self.gen_uniform = config['endgame_training']['generation_uniform']

        self.num_samples = len(data.states)
        self.data = data
        self.weights = None
        self.population = list(range(self.num_samples))
        self.generation = generation
        self.generate_weights()
        self.idx_q = []

        print("Done initializing endgame sampling")
        
    def sample(self):
        if len(self.idx_q) == 0:
            self.refresh_indices()

        idx = self.idx_q.pop()

        return self.data.states[idx], self.data.policies[idx], self.data.outcomes[idx]

    def refresh_indices(self):
        mult_sample = torch.multinomial(self.weights, config['endgame_sampling_q_size'], replacement=True)
        self.idx_q = mult_sample.tolist()
    
    def generate_weights(self):
        self.weights = self.sjonn_dist(self.data.moves_left)

    def sjonn_dist(self, moves_left: torch.Tensor):
        """Returns a distribution that is more likely to sample games that are closer to the end"""
        p = self.p_max - (self.p_max - self.p_min) * self.generation / self.gen_uniform
        p = max(p, self.p_min)
        return 1/(moves_left *p + self.shift)


class UniformSampler:
    def __init__(self, data:Data):
        self.data = data
        self.num_samples = data.states.shape[0]

    def sample(self):
        idx = random.randint(0, self.num_samples - 1)
        return self.data.states[idx], self.data.policies[idx], self.data.outcomes[idx]


def get_sampler(data:Data, generation: int):
    if config['sample_method'] == 'uniform':
        return UniformSampler(data)
    elif config['sample_method'] == 'endgame':
        return EndgameSampling(data, generation)


class DS(IterableDataset):
    def __init__(self, generation:int) -> None:
        
        num_generations = utils.training_gens(generation)

        min_gen = max(generation - num_generations + 1, 0)
        training_generations = list(range(min_gen, generation + 1))

        print(f"Training generations: {','.join(map(str, training_generations))}")

        data: Data = load_generations(training_generations)

        self.sampler = get_sampler(data, generation)

    def __iter__(self):
        while True:
            yield self.sampler.sample()


def get_dataloader( 
    generation: int
):

    dataset = DS(generation)

    return DataLoader(
        dataset=dataset, 
        batch_size=config['batch_size'], 
        num_workers=config['dl_num_workers'], 
        pin_memory=True,
        prefetch_factor=config['dl_prefetch_factor']
    )
