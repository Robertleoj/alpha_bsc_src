from DB import DB
from utils import make_folder, Data
import torch
import os
from colorama import Fore, init
from pathlib import Path

init(autoreset=True)

CACHED_DATA_PATH = Path("./cached_data")
CACHED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def prefetch_generation(generation:int):

    db = DB()
    states, policies, outcomes, moves_left, weights = db.prefetch_generation(generation)

    print(f"Prefeched {states.shape[0]} samples for generation {Fore.GREEN}{generation}{Fore.RESET}.")

    # Ensure that folders exist

    torch.save(
        Data(
            states=states, 
            policies=policies, 
            outcomes=outcomes,
            moves_left=moves_left,
            weights=weights
        ), 
        CACHED_DATA_PATH/f"{generation}.pt"
    )


def generation_exists(generation:int):
    return (CACHED_DATA_PATH/f"{generation}.pt").exists()


def load_generation(generation:int):
    
    if not generation_exists(generation):
        prefetch_generation(generation)


    data = torch.load(CACHED_DATA_PATH/f"{generation}.pt")
    ok = False
    try:
        data.weights
        ok = True
    except:
        pass

    if ok:
        return data
    else:
        new_data = Data(data.states, data.policies, data.outcomes, data.moves_left, torch.ones(data.states.shape[0]))
        torch.save(new_data, CACHED_DATA_PATH/f"{generation}.pt")
        return new_data


def load_generations(generations:list):
    """Loads multiple generations of data

    Args:
        generations (list): _description_

    Returns:
        tuple(Tensor*): Return states, policies, outcomes, games_left
    """

    generation_data = []

    for generation in generations:
        generation_data.append(load_generation(generation))

    states = torch.concat([x.states for x in generation_data], 0)
    policies = torch.concat([x.policies for x in generation_data], 0)
    outcomes = torch.concat([x.outcomes for x in generation_data], 0)
    moves_left = torch.concat([x.moves_left for x in generation_data], 0)
    weights = torch.concat([x.weights for x in generation_data], 0)

    return Data(states, policies, outcomes, moves_left, weights)
    
