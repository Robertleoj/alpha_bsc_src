import bz2
import io
import lzma
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch
from DB.db import DB
import random
from config import config
import utils
from utils import Data
from pathlib import Path

CHUNK_SAMPLES = 20

class UniformSampler:
    def __init__(self, generations:int):
        self.chunk_paths = []

        for generation in generations:
            for file in (Path('./cached_data')/f"{generation}/").glob('*.pt'):
                self.chunk_paths.append(file)

        self.num_chunks = len(self.chunk_paths)
        self.buffer:list[Data] = []

    def load_chunks(self):

        for _ in range(CHUNK_SAMPLES):
            idx = random.randint(0, self.num_chunks - 1)

            with bz2.open(self.chunk_paths[idx], 'rb') as f:
                chunk = torch.load(f)

            self.buffer.extend(chunk)

        random.shuffle(self.buffer)

    def sample(self):

        if len(self.buffer) == 0:
            self.load_chunks()

        data = self.buffer.pop()

        return data.state, data.policy, data.outcome, data.weight



def get_sampler(generations):
    if config['sample_method'] == 'uniform':
        return UniformSampler(generations)


class DS(IterableDataset):
    def __init__(self, generation:int) -> None:
        
        num_generations = utils.training_gens(generation)

        min_gen = max(generation - num_generations + 1, 0)
        training_generations = list(range(min_gen, generation + 1))

        print(f"Training generations: {','.join(map(str, training_generations))}")

        db = DB()
        for gen in training_generations:
            db.prefetch_generation(gen)

        self.sampler = get_sampler(training_generations)

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
