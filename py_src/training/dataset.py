from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

from DB.db import DB
from random import choice
import config

class DS(IterableDataset):
    def __init__(self, generation:int, game: str, db: DB) -> None:
        self.db = db
        self.db_ids = db.get_ids(generation, game)

    def __iter__(self):
        # while True:
        #     rand_idx = choice(self.db_ids)

        #     yield self.db.get_training_sample(rand_idx)

        while True:
            yield choice(self.db_ids)


class Collate:
    def __init__(self, db: DB):
        self.db = db
        
    def __call__(self, lis):
        return self.db.get_training_samples(lis)  


def get_dataloader(
    db: DB,
    game: str, 
    generation: int
):
    return DataLoader(
        collate_fn=Collate(db), 
        dataset=DS(generation, game, db), 
        batch_size=config.batch_size, 
        num_workers=config.dl_num_workers, 
        pin_memory=True,
        prefetch_factor=config.dl_prefetch_factor
    )