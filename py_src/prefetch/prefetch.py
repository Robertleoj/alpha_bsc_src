from DB import DB
from utils import make_folder, Data
import torch
import os
from colorama import Fore, init
init(autoreset=True)


def prefetch_generation(game:str, generation:int):
    db = DB()
    states, policies, outcomes, moves_left = db.prefetch_generation(game, generation)

    print(f"Prefeched {states.shape[0]} samples for {Fore.GREEN}{game}{Fore.RESET} generation {Fore.GREEN}{generation}{Fore.RESET}.")

    # Ensure that folders exist
    make_folder(f"training_data/{game}")
    torch.save(
        Data(
            states=states, 
            policies=policies, 
            outcomes=outcomes,
            moves_left=moves_left
        ), 
        f"training_data/{game}/{generation}.pt"
    )


def generation_exists(game:str, generation:int):
    return os.path.exists(f"training_data/{game}/{generation}.pt")


def load_generation(game:str, generation:int):
    
    if not generation_exists(game, generation):
        prefetch_generation(game, generation)

    return torch.load(f"training_data/{game}/{generation}.pt")


def load_generations(game:str, generations:list):
    """Loads multiple generations of data

    Args:
        game (str): _description_
        generations (list): _description_

    Returns:
        tuple(Tensor*): Return states, policies, outcomes, games_left
    """

    generation_data = []

    for generation in generations:
        generation_data.append(load_generation(game, generation))

    states = torch.concat([x.states for x in generation_data], 0)
    policies = torch.concat([x.policies for x in generation_data], 0)
    outcomes = torch.concat([x.outcomes for x in generation_data], 0)
    moves_left = torch.concat([x.moves_left for x in generation_data], 0)

    return Data(states, policies, outcomes, moves_left)
    
