from pathlib import Path
from sys import argv
from utils import set_run
from DB import DB
from multiprocessing import Pool, cpu_count
import os
import json
import bz2
import torch

CACHE_PATH = Path('cached_data/')
CPP_CONF = None


def get_curr_playouts():
    pth = Path('num_playouts.json')
    if pth.exists():
        with open(pth, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_playouts(playouts):
    with open('num_playouts.json', 'w') as f:
        json.dump(playouts, f, indent=4)

def get_playouts_in_file(file_path):
    with bz2.open(file_path, 'rb') as f:
        chunk = torch.load(f)
        # min_weights = CPP_CONF['endgame_min_playout']


        min_weight = 0
        if CPP_CONF['use_endgame_playouts']:
            min_weight = CPP_CONF['endgame_min_playouts'] / CPP_CONF['search_depth']
        
        # abs for randomized cap
        return sum((max(abs(x.weight), min_weight) for x in chunk))
   

def get_playouts_in_gen(generation: int):
    db = DB()

    db.prefetch_generation(generation)

    path = CACHE_PATH / str(generation)

    if not path.exists():
        return None

    files = [x for x in path.iterdir() if x.is_file()]

    playouts = 0

    with Pool(cpu_count()) as p:
        playouts = sum(p.map(get_playouts_in_file, files))

    return playouts*CPP_CONF['search_depth']



def update_playouts(playouts):
    db = DB()

    # Get all folders in cached_data
    generations = sorted(db.generation_nums())


    # Get all files in each generation
    for generation in generations:


        if str(generation) not in playouts:

            n = get_playouts_in_gen(generation)
            if n is not None:
                playouts[generation] = n

    return playouts

def make_num_playouts(game_name, run_name):
    
    curr_dir = Path.cwd()
    set_run(run_name, game_name)
    global CPP_CONF

    with open('cpp_hyperparameters.json', 'r') as f:
        CPP_CONF = json.load(f)


    curr_playouts = get_curr_playouts()
    updated = update_playouts(curr_playouts)
    save_playouts(updated)

    os.chdir(curr_dir)

    return updated



def main():
    try:
        game_name = argv[1]
        run_name = argv[2]
    except IndexError:
        print('Usage: python make_num_playouts.py game_name run_name')
        return

    make_num_playouts(game_name, run_name)

if __name__ == '__main__':
    main()