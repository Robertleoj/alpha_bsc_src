import sys

from tqdm import tqdm
from utils import set_run
from pathlib import Path
import pandas as pd
from glob import glob
from conn4_solver import solve, evaluate_many
from DB import DB
import signal
import bson
import numpy as np
import os
from dataclasses import dataclass
import dataclasses
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random
import multiprocessing

import warnings

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

save_thread = None

"""
pandas fields

generation
moves_left
moves_played
gt_eval
game_outcome
"""

COLUMNS = ["generation", "moves_left", "moves_played", "gt_eval", "game_outcome"]
THREADS = 32
ILLEGAL_MOVE = -1000
# GEN_SAMPLES = 10000
FNAME = Path("./td_evaluations.csv")
BOOK_PATH = '../../../db/7x6.book'

@dataclass
class Position:
    moves_left: int
    moves_played: int
    game_outcome: float
    moves: str

def scores_to_pos_score(scores: list[int]) -> float:
    for i in range(len(scores)):
        if scores[i] == ILLEGAL_MOVE:
            scores[i] = -float('inf')
        elif scores[i] < 0:
            scores[i] = -1
        elif scores[i] > 0:
            scores[i] = 1

    return max(scores)


def get_eval_df() -> pd.DataFrame:
    if FNAME.exists():
        return pd.read_csv(FNAME)
    else:
        return pd.DataFrame(columns=COLUMNS)

def sample_to_pos(sample):
    moves_left = sample['moves_left']
    game_outcome = sample['outcome']
    moves_played = sample['moves'].count(';')
    moves = sample['moves'].replace(';', '')
    return Position(
        moves_left=moves_left,
        moves_played=moves_played,
        game_outcome=game_outcome,
        moves=moves
        )



def get_positions(generation) -> list[Position]:
    db = DB()

    db.uncompress_generation(generation)

    # get all files
    game_files = glob(f"./training_data/{generation}/*.bson")

    positions = []

    samples = []


    for game_file in tqdm(game_files, desc="Reading data files"):
        with open(game_file, "rb") as f:
            data = bson.loads(f.read())['samples']

        samples.extend(data)

            
    with Pool(THREADS) as p:
        positions = p.map(sample_to_pos, samples, chunksize=128)

    db.compress_generation(generation)

    return positions
 

# def eval_position(position: Position) -> float:
#     moves = position.moves.replace(';', '')
#     scores = solve(moves)
#     pos_score = scores_to_pos_score(scores)
#     return pos_score


# def clean_pos(pos: Position) -> Position:
#     moves = pos.moves.replace(';', '')


def eval_td_generation(generation:int) -> pd.DataFrame:
    print("Getting positions")
    positions = get_positions(generation)
    print("Got positions")
    # random.shuffle(positions)
    # positions = positions[:GEN_SAMPLES]

    # turn to dataframe
    print("Doing pandas stuff...")

    dict_list = [dataclasses.asdict(p) for p in positions]



    df = pd.DataFrame(
        dict_list,
        # columns=COLUMNS, 
        # dtype={
        #     'generation': pd.Int64Dtype(), 
        #     'moves_left': pd.Int64Dtype(),
        #     'moves_played': pd.Int64Dtype(),
        #     'gt_eval': pd.Float64Dtype(),
        #     'game_outcome': pd.Float64Dtype(),
        #     "moves": pd.StringDtype()
        # }
    )

    if 'moves' in df.columns:
        df.drop(['moves'], axis=1, inplace=True)

    df['gt_eval'] = np.nan
    df['gt_eval'] = df['gt_eval'].astype(float)
    df['generation'] = generation
    df['generation'] = df['generation'].astype(int)

    df['moves_played'] = df['moves_played'].astype(int)
    df['game_outcome'] = df['game_outcome'].astype(float)
    df['moves_left'] = df['moves_left'].astype(int)

    print("Done with pandas stuff...")

    # with Pool(THREADS) as p:
    #     scores = list(tqdm(p.imap(eval_position, positions, chunksize=16), total=len(positions), desc="Evaluating positions"))

    scores = evaluate_many([p.moves for p in positions], BOOK_PATH)
    print("Done with evaluation")
    df['gt_eval'] = scores
    print("Updated df")

    return df
    

def fill_evals(evals: pd.DataFrame) -> pd.DataFrame:
    global save_thread
    # get generations
    db = DB()
    generations = db.generation_nums()

    for gen in generations:
        # get games
        if evals.query("generation == @gen").empty:
            print("Evaluating generation", gen)
            evals = evals.append(eval_td_generation(gen), ignore_index=True)
            if save_thread is not None:
                save_thread.join()

            print("Saving evals... ", end="", flush=True)
            save_thread = multiprocessing.Process(target=save_evals, args=(evals,))
            save_thread.start()
            # save_thread = Thread(target=save_evals, args=(evals,))
            print("Done")
        else:
            print("Generation", gen, "already evaluated, continuing...")

    if save_thread is not None:
        save_thread.join()

    return evals

def save_evals(evals: pd.DataFrame):
    evals.to_csv(FNAME, index=False)

def gen_avg(plot_df, k, move_col):

    plot_df['generation_interval'] = (plot_df['generation']) // k * k

    plot_df = plot_df.groupby(['generation_interval', move_col])['eval_error'].mean().reset_index()

    plot_df['generation'] = plot_df['generation_interval'].astype(str) + '-' + (plot_df['generation_interval'] + k - 1).astype(str)

    plot_df = plot_df.drop('generation_interval', axis=1)

    # Reorder the columns
    plot_df = plot_df[['generation', move_col, 'eval_error']]

    return plot_df



def make_plots(evals: pd.DataFrame, max_gen=None, gen_every=1):
    fig_path = Path("./figures")
    
    fig_path.mkdir(exist_ok=True)
    print(evals.columns)

    if max_gen is not None:
        evals = evals.query("generation <= @max_gen")


    # x axis num moves left, y axis mean error, colored by generation
    # make new dataframe for the plot
    # add error column to df
    evals['eval_error'] = (evals['gt_eval'] - evals['game_outcome'])**2
    # evals['generation'] = evals['generation']#.round().astype(int)

    plot_df = evals.groupby(['generation', 'moves_left'])['eval_error'].mean().reset_index()

    plot_df = plot_df[['generation', 'moves_left', 'eval_error']]

    if gen_every > 1:
        plot_df = gen_avg(plot_df, gen_every, 'moves_left')

    
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=plot_df, x="moves_left", y="eval_error", hue="generation")
    plt.title("Moves left MSE")
    plt.xlabel("Moves Left")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(fig_path / "eval_vs_moves_left_mse.png")
    plt.clf()

    plt.figure(figsize=(7, 4))
    plot_df['eval_error'] = plot_df['eval_error']**0.5
    sns.lineplot(data=plot_df, x="moves_left", y="eval_error", hue="generation")
    plt.title("Moves Left RMSE")
    plt.xlabel("Moves Left")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(fig_path / "eval_vs_moves_left_rmse.png")
    plt.clf()

    # x axis num moves played, y axis mean error, colored by generation
    plot_df = evals.groupby(['generation', 'moves_played'])['eval_error'].mean().reset_index()
    plot_df = plot_df[['generation', 'moves_played','eval_error']]

    if gen_every > 1:
        plot_df = gen_avg(plot_df, gen_every, 'moves_played')

        


    
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=plot_df, x="moves_played", y="eval_error", hue="generation")
    plt.title("Moves Played MSE")
    plt.xlabel("Moves Played")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(fig_path / "eval_vs_moves_played_mse.png")
    plt.clf()

    plot_df['eval_error'] = plot_df['eval_error']**0.5
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=plot_df, x="moves_played", y="eval_error", hue="generation")
    plt.xlabel("Moves Played")
    plt.title("Moves Played RMSE")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(fig_path / "eval_vs_moves_played_rmse.png")
    plt.clf()


def sigint_handler(sig, frame):
    global save_thread
    if save_thread is not None:
        save_thread.join()
    sys.exit(0)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_training_data.py <run_name> <plot/eval>")

    signal.signal(signal.SIGINT, sigint_handler)

    game_name = "connect4"

    set_run(sys.argv[1], game_name)

    if sys.argv[2] == "eval":
        evals = get_eval_df()
        fill_evals(evals)

    elif sys.argv[2] == "plot":
        evals = get_eval_df()
        # mask = evals.isna().any(axis=1)
        # print(evals[mask])
        max_gen = None
        if len(sys.argv) > 3:
            max_gen = int(sys.argv[3])

        gen_every = 1
        if len(sys.argv) > 4:
            gen_every = int(sys.argv[4])

        make_plots(evals, max_gen, gen_every)

    else:
        print("Usage: python3 evaluate_training_data.py <run_name> <plot/eval>")

if __name__ == "__main__":
    main()



# for all states
## evaluate position
## connect to number of moves left

# where to write?

# Make plots



