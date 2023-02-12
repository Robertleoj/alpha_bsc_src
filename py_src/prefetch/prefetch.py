from DB import DB
from utils import make_folder
import torch
import os
from colorama import Fore, init
init(autoreset=True)


def prefetch_generation(game:str, generation:int):
    db = DB()
    states, policies, outcomes = db.prefetch_generation(game, generation)

    print(f"Prefeched {states.shape[0]} samples for {Fore.GREEN}{game}{Fore.RESET} generation {Fore.GREEN}{generation}{Fore.RESET}.")

    # Ensure that folders exist
    make_folder(f"training_data/{game}")
    torch.save((states, policies, outcomes), f"training_data/{game}/{generation}.pt")


def generation_exists(game:str, generation:int):
    return os.path.exists(f"training_data/{game}/{generation}.pt")


def load_generation(game:str, generation:int):
    
    if not generation_exists(game, generation):
        prefetch_generation(game, generation)

    return torch.load(f"training_data/{game}/{generation}.pt")


def load_generations(game:str, generations:list):

    generation_data = []

    for generation in generations:
        generation_data.append(load_generation(game, generation))

    states = torch.concat([x[0] for x in generation_data], 0)
    policies = torch.concat([x[1] for x in generation_data], 0)
    outcomes = torch.concat([x[2] for x in generation_data], 0)

    return states, policies, outcomes
    
