import sys

from tqdm import tqdm
from utils import set_run
from pathlib import Path
import pandas as pd
from glob import glob
from conn4_solver import solve, evaluate_many
from DB import DB
import bson
import numpy as np
import os
from dataclasses import dataclass
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import random

import warnings

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
GEN_SAMPLES = 10000
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
        positions = p.map(sample_to_pos, samples)

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
    positions = get_positions(generation)
    random.shuffle(positions)

    positions = positions[:GEN_SAMPLES]

    df = pd.DataFrame(columns=COLUMNS)

    # turn to dataframe
    for position in positions:
        df = df.append({"generation": generation,
                        "moves_left": position.moves_left,
                        "moves_played": position.moves_played,
                        "gt_eval": np.nan,
                        "game_outcome": position.game_outcome}, ignore_index=True)

    # with Pool(THREADS) as p:
    #     scores = list(tqdm(p.imap(eval_position, positions, chunksize=16), total=len(positions), desc="Evaluating positions"))

    scores = evaluate_many([p.moves for p in positions], BOOK_PATH)

    df['gt_eval'] = scores

    return df
    

def fill_evals(evals: pd.DataFrame) -> pd.DataFrame:
    # get generations
    db = DB()
    generations = db.generation_nums()

    for gen in generations:
        # get games
        if evals.query("generation == @gen").empty:
            print("Evaluating generation", gen)
            evals = evals.append(eval_td_generation(gen), ignore_index=True)
            save_evals(evals)
        else:
            print("Generation", gen, "already evaluated, continuing...")

    return evals

def save_evals(evals: pd.DataFrame):
    evals.to_csv(FNAME, index=False)


def make_plots(evals: pd.DataFrame):
    fig_path = Path("./figures")
    
    fig_path.mkdir(exist_ok=True)

    # x axis num moves left, y axis mean error, colored by generation
    # make new dataframe for the plot
    # add error column to df
    evals['eval_error'] = (evals['gt_eval'] - evals['game_outcome'])**2
    evals['generation'] = evals['generation'].round().astype(int)

    plot_df = evals.groupby(['generation', 'moves_left'])['eval_error'].mean().reset_index()
    plot_df = plot_df[['generation', 'moves_left', 'eval_error']]
    
    sns.lineplot(data=plot_df, x="moves_left", y="eval_error", hue="generation")
    plt.title("Moves left MSE")
    plt.savefig(fig_path / "eval_vs_moves_left_mse.png")
    plt.clf()

    plot_df['eval_error'] = plot_df['eval_error']**0.5
    sns.lineplot(data=plot_df, x="moves_left", y="eval_error", hue="generation")
    plt.title("Moves Left RMSE")
    plt.savefig(fig_path / "eval_vs_moves_left_rmse.png")
    plt.clf()

    # x axis num moves played, y axis mean error, colored by generation
    plot_df = evals.groupby(['generation', 'moves_played'])['eval_error'].mean().reset_index()
    plot_df = plot_df[['generation', 'moves_played','eval_error']]
    
    sns.lineplot(data=plot_df, x="moves_played", y="eval_error", hue="generation")
    plt.title("Moves Played MSE")
    plt.savefig(fig_path / "eval_vs_moves_played_mse.png")
    plt.clf()

    plot_df['eval_error'] = plot_df['eval_error']**0.5
    sns.lineplot(data=plot_df, x="moves_played", y="eval_error", hue="generation")
    plt.title("Moves Played RMSE")
    plt.savefig(fig_path / "eval_vs_moves_played_rmse.png")
    plt.clf()




def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_training_data.py <run_name> <plot/eval>")

    game_name = "connect4"

    set_run(sys.argv[1], game_name)

    if sys.argv[2] == "eval":
        evals = get_eval_df()
        fill_evals(evals)

    elif sys.argv[2] == "plot":
        evals = get_eval_df()
        make_plots(evals)

    else:
        print("Usage: python3 evaluate_training_data.py <run_name> <plot/eval>")

if __name__ == "__main__":
    main()



# for all states
## evaluate position
## connect to number of moves left

# where to write?

# Make plots



