from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

from DB.db import DB
import random
import config
from prefetch import load_generations

class DS(IterableDataset):
    def __init__(self, generation:int, game: str) -> None:
        db = DB()
        
        min_gen = max(generation - config.buffer_generations, 0)

        training_generations = list(range(min_gen, generation + 1))

        self.data = load_generations(game, training_generations)

        self.size = self.data[0].shape[0]

    def __iter__(self):

        while True:
            idx = random.randint(0, self.size - 1)
            yield self.data[0][idx], self.data[1][idx], self.data[2][idx]



def get_dataloader(
    game: str, 
    generation: int
):

    dataset = DS(generation, game)

    return DataLoader(
        dataset=dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dl_num_workers, 
        pin_memory=True,
        prefetch_factor=config.dl_prefetch_factor
    )