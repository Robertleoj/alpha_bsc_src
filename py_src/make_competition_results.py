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

CONST_PLAY_EVERY = 3


NUM_PLAYOUTS = 800


def get_run_cum_playouts(game_name, run) -> list[int]:
    runout = [0]

    run = make_num_playouts(game_name, run)

    for key, val in sorted(run.items(), key=lambda x: int(x[0])):
        key = int(key)
        if key != 0:
            val += runout[key]
        runout.append(val)
        
    return runout
        
    

def get_results(pth, perfect=False):
    
    if not pth.exists():
        if perfect:
            columns = ['run', 'gen', 'playouts', 'white_win_rate', 'black_win_rate', 'win_rate']
        else:
            columns= ['run1', 'gen1', 'playouts1', 'run2', 'gen2', 'playouts2', 'r1_white_win_rate', 'r1_black_win_rate', 'r1_win_rate']

        return pd.DataFrame(columns=columns)
    else:
        return pd.read_csv(pth)


def get_competitions(run1_playouts, run2_playouts, mode, run2_mult=None, const_gen=None):
    if mode in ('gen', 'playout'):
        for r1_gen, r1_playouts in enumerate(run1_playouts):
            if mode == 'gen':
                lam = lambda x: abs(x[0] - r1_gen)
            else:
                lam = lambda x: abs(x[1] * run2_mult - r1_playouts)

            r2_gen, r2_playouts = min(enumerate(run2_playouts), key=lam)
            
            yield r1_gen, r1_playouts, r2_gen, r2_playouts

    elif mode == 'const':
        assert isinstance(const_gen, int)
        r1_gen = const_gen
        r1_playouts = run1_playouts[r1_gen]

        r2_matches = list(enumerate(run2_playouts))[::CONST_PLAY_EVERY]

        for r2_gen, r2_playouts in r2_matches:
            yield r1_gen, r1_playouts, r2_gen, r2_playouts


def perfect_compete(run, random):
    game_name = 'connect4'
    random_mod = 'random' if random else 'deterministic'
    pth = Path(f"../db/competitions/perfect/{run}_{random_mod}/results.csv")
    pth.parent.mkdir(parents=True, exist_ok=True)
    run_playouts = get_run_cum_playouts(game_name, run)

    curr_results = get_results(pth, perfect=True)

    for gen, playouts in list(enumerate(run_playouts))[::CONST_PLAY_EVERY]:
        if curr_results.query(f'gen == {gen}').shape[0] != 0:
            continue
        
        res = compete_result(game_name, NUM_PLAYOUTS, run, gen, 'perfect', 0, random)
        mean_win = lambda x: ((sum(x) / len(x)) + 1) / 2

        white_res = mean_win(res.p1.white)
        black_res = mean_win(res.p1.black)
        res = mean_win(res.p1.white + res.p1.black)

        # append row to df
        curr_results = curr_results.append({
            'run': run,
            'gen': gen,
            'playouts': playouts,
            'white_win_rate': white_res,
            'black_win_rate': black_res,
            'win_rate': res
        }, ignore_index=True)

        curr_results.to_csv(pth, index=False)


    

def compete_experiments(game_name, run1, run2, mode:str, **kwargs):
    run2_mult = 1.0
    if 'run2_mult' in kwargs:
        run2_mult = kwargs['run2_mult']
        if run2_mult is None:
            raise ValueError('run2_mult not specified')


    if mode == 'const':
        if 'r1_const_gen' not in kwargs or kwargs['r1_const_gen'] is None:
            raise ValueError('r1_const_gen not specified')

        r1_const_gen = kwargs['r1_const_gen']


    mult_modifier = '' if mode == 'gen' or mode == 'const' else f'_{run2_mult}'
    const_gen_modifier = '' if mode != 'const' else f'_{r1_const_gen}'

    pth = Path(f'../db/competitions/{game_name}/{run1}vs{run2}_{mode}{mult_modifier}{const_gen_modifier}/results.csv')
    pth.parent.mkdir(parents=True, exist_ok=True)

    run1_playouts, run2_playouts = get_run_cum_playouts(game_name, run1), get_run_cum_playouts(game_name, run2)

    run1_playouts: list[float]
    run2_playouts: list[float]

    curr_results = get_results(pth)

    competitions = get_competitions(run1_playouts, run2_playouts, mode, run2_mult, r1_const_gen)

    # run1 is default, run2 is endgame
    for r1_gen, r1_playouts, r2_gen, r2_playouts in competitions:

        # check if already computed
        if curr_results.query(f'run1 == "{run1}" and gen1 == {r1_gen} and run2 == "{run2}" and gen2 == {r2_gen}').shape[0] != 0:
            continue

        # battle r1_gen and r2_gen
        res = compete_result(game_name, NUM_PLAYOUTS, run1, r1_gen, run2, r2_gen)
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

        if run1 == 'perfect':
            assert game_name == 'connect4'
            random = argv[4]
            assert random in ('random', 'perfect')
            random = random == 'random'

            perfect_compete(run2, random)
            
        else:
            mode = argv[4]

            if mode == 'playout':
                r2_mult = float(argv[5])
            else:
                r2_mult = None

            if mode == 'const':
                r1_const_gen = int(argv[5])
            else:
                r1_const_gen = None


            print(f'game_name: {game_name}, run1: {run1}, run2: {run2}')
            compete_experiments(game_name, run1, run2, mode, r2_mult=r2_mult, r1_const_gen=r1_const_gen)

    except IndexError:

        print("Usage: python make_competition_results.py <game_name> <run1> <run2> [gen/playout/const] [<r2_mult>]")
        return
    

    
if __name__ == '__main__':
    main()


    
