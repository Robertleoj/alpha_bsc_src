from make_num_playouts import make_num_playouts
from pathlib import Path
from sys import argv
import json
from compete import compete_result
import pandas as pd

import warnings

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# VAULT: Path = Path('vault')
# GAME_PATH: Path = None
# RUN1_PATH: Path = None
# RUN2_PATH :Path = None

GAME_NAME: str = None
RUN1_NAME: str = None
RUN2_NAME: str = None



def get_run_cum_playouts(game_name, run1, run2):
    run1out = [0]
    run2out = [0]

    run1 = make_num_playouts(game_name, run1)
    # with open(RUN1_PATH / 'num_playouts.json', 'r') as f:
    #     run1 = json.load(f)

    run2 = make_num_playouts(game_name, run2)
    # with open(RUN2_PATH / 'num_playouts.json', 'r') as f:
    #     run2 = json.load(f)


    for key, val in sorted(run1.items(), key=lambda x: int(x[0])):
        key = int(key)
        if key != 0:
            val += run1out[key]
        run1out.append(val)
        
    for key, val in sorted(run2.items(), key=lambda x: int(x[0])):
        key = int(key)
        if key != 0:
            val += run2out[key]
        run2out.append(val)

    return run1out, run2out
        
    

def get_results(pth):
    
    if not pth.exists():
        return pd.DataFrame(columns=['run1', 'gen1', 'playouts1', 'run2', 'gen2', 'playouts2', 'r1_white_win_rate', 'r1_black_win_rate', 'r1_win_rate'])
    else:
        return pd.read_csv(pth)

def compete_experiments(game_name, run1, run2, run2_mult=1.0):
    pth = Path(f'../db/competitions/{game_name}/{run1}vs{run2}_mult{run2_mult}/results.csv')
    pth.parent.mkdir(parents=True, exist_ok=True)

    run1_playouts, run2_playouts = get_run_cum_playouts(game_name, run1, run2)
    run1_playouts: list[float]
    run2_playouts: list[float]

    curr_results = get_results(pth)

    # run1 is default, run2 is endgame
    for r1_gen, r1_playouts in enumerate(run1_playouts):
        r2_gen, r2_playouts = min(enumerate(run2_playouts), key=lambda x: abs(x[1] * run2_mult - r1_playouts))

        # check if already computed
        if curr_results.query(f'run1 == "{run1}" and gen1 == {r1_gen} and run2 == "{run2}" and gen2 == {r2_gen}').shape[0] != 0:
            continue

        # battle r1_gen and r2_gen
        res = compete_result(game_name, 800, run1, r1_gen, run2, r2_gen)
        mean_win = lambda x: ((sum(x) / len(x)) + 1) / 2

        r1_white_res = mean_win(res.p1.white)
        r1_black_res = mean_win(res.p1.black)
        r1_res = mean_win(res.p1.white + res.p1.black)

        # append row to df
        curr_results = curr_results.append({
            'run1': run1,
            'gen1': r1_gen,
            'playouts1': r1_playouts,
            'run2': run2,
            'gen2': r2_gen,
            'playouts2': r2_playouts,
            'r1_white_win_rate': r1_white_res,
            'r1_black_win_rate': r1_black_res,
            'r1_win_rate': r1_res
        }, ignore_index=True)

        curr_results.to_csv(pth, index=False)


    

def main():
    try:
        game_name = argv[1]
        run1 = argv[2]
        run2 = argv[3]
        r2_mult = float(argv[4])
        print(f'game_name: {game_name}, run1: {run1}, run2: {run2}')
    except IndexError:
        print("Usage: python make_competition_results.py <game_name> <run1> <run2> <r2_mult>")
        return
    
    # GAME_PATH = VAULT / game_name
    # RUN1_PATH = GAME_PATH / run1
    # RUN2_PATH = GAME_PATH / run2

    compete_experiments(game_name, run1, run2, r2_mult)

    
if __name__ == '__main__':
    main()


    
