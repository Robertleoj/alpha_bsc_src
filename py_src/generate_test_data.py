from conn4_solver import solve
from multiprocessing import Pool, cpu_count
from sys import argv
import json
from torch import softmax
from random import randint
import torch
import os


ILLEGAL_MOVE = -1000
VALUE = 0.7
THREADS = 32

def fix_scores(scores: list[int]) -> torch.Tensor:
    for i in range(len(scores)):
        if scores[i] == ILLEGAL_MOVE:
            scores[i] = -float('inf')
        elif scores[i] < 0:
            scores[i] = -1
        elif scores[i] > 0:
            scores[i] = 1
    return torch.tensor(scores, dtype=torch.float32)


def target_dist(moves: str)-> list[float]:

    scores = solve(moves.split())
    scores = fix_scores(scores)
    best = max(scores)

    scores = (scores == best).float() 

    dist = scores / scores.sum()

    return dist.tolist(), best.item()


def select_move(scores):
    dist = softmax(scores, dim=0)
    torch.manual_seed(randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff))
    # Sample from the softmax distribution
    try:
        return torch.multinomial(dist, 1)[0].item() + 1
    except:
        print()
        print(scores)
        print(dist)
        raise Exception("Error in select_move")


def play_game() -> list[str]:
    move_list = []

    while True:

        try:
            scores = solve(move_list)
        except:
            break
        
        scores = fix_scores(scores) * VALUE
        move = select_move(scores)
        move_list.append(str(move))

    return move_list



def random_position(*args) -> list[str]:
    try:
        move_list = play_game()
    except Exception as e:
        print(e)
        return []

    print(".", end='', flush=True)
    cutoff = randint(0, len(move_list)-2)
    return move_list[:cutoff]

def generate(n: int, curr_games: list[dict])-> list[str]:

    games = [curr_games[i]['moves'] for i in range(len(curr_games))]

    new_games = []

    while len(new_games) < n:
        with Pool(THREADS) as p:
            sampled = p.map(random_position, range(n))

        sampled = [''.join(pos) for pos in sampled]
        new_games.extend([
            pos for pos in sampled 
            if pos not in games and pos not in new_games
        ])

    with Pool(THREADS) as p:
        targets = p.map(target_dist, new_games)

    print()


    new_games_evaluated =  [{'moves': game, 'target': target[0], 'value': target[1]} for game, target in zip(new_games, targets)]

    return curr_games + new_games_evaluated
    

def save_json(games: list[str], fname: str):
    with open(fname, 'w') as f:
        json.dump(games, f, indent=4)


def get_json(fname: str)-> list[str]: 
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return json.load(f)
    return []


def main():
    n_games = int(argv[1])
    fname = argv[2]

    curr_games = get_json(fname)
    n_games -= len(curr_games)
    n_games = max(n_games, 0)
    print("Loaded", len(curr_games), "positions from", fname, f"adding {n_games} new positions...")
    if n_games == 0:
        print("Nothing to do")
        return

    games = generate(n_games, curr_games)
    print(*games, sep='\n')
    print(f"Generated {len(games)} positions")
    save_json(games, fname)


if __name__ == '__main__':
    main()


    
    
