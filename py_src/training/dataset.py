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

class UniformSampler:
    def __init__(self, generations:int):
        self.chunk_paths:list[Path] = []

        self.use_randomized_cap = (
            config.cpp_conf_has_key('use_randomized_cap') 
            and config.cpp_conf('use_randomized_cap')
        )

        for generation in generations:
            for file in (Path('./cached_data')/f"{generation}/").glob('*.pt'):
                self.chunk_paths.append(file)


        self.num_chunks = len(self.chunk_paths)
        self.buffer:list[Data] = []

    def load_chunks(self):

        for _ in range(config['cache_sample_chunks']):
            idx = random.randint(0, self.num_chunks - 1)
            chunk_path = self.chunk_paths[idx]

            if self.use_randomized_cap:
                uid_name = chunk_path.stem
                gen = chunk_path.parent.stem


                c2_path = Path(f'./rand_cap_cache/{gen}/{uid_name}.pt')

                if not c2_path.exists():
                    c2_path.parent.mkdir(parents=True, exist_ok=True)
                    with bz2.open(chunk_path, 'rb') as f:
                        unfiltered_chunk = torch.load(f)

                    chunk = [c for c in unfiltered_chunk if c.weight > 0]

                    buffer = io.BytesIO()
                    torch.save(chunk, buffer)
                    with bz2.open(c2_path, "wb") as f:
                        f.write(buffer.getvalue())

                    # with bz2.open(c2_path, 'wb') as f:
                        # torch.save(chunk, f)
                else:
                    try:
                        with bz2.open(c2_path, 'rb') as f:
                            chunk = torch.load(f)
                    except Exception as e:
                        print(e.args)
                        print(c2_path)
                        print("Ignoring error.")
                        continue

            else:
                with bz2.open(chunk_path, 'rb') as f:
                    chunk = torch.load(f)

            self.buffer.extend(chunk)

        random.shuffle(self.buffer)

    def sample(self):

        while True:
            if len(self.buffer) == 0:
                self.load_chunks()

            data = self.buffer.pop()
            if data.weight < 0:
                continue

            if data.weight < 1 and self.use_randomized_cap:
                data.weight = 1.0

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
