from conn4_solver import solve
from multiprocessing import Pool, cpu_count
from sys import argv
import json
from torch import softmax
from random import randint
import torch


ILLEGAL_MOVE = -1000
VALUE = 0.7

def fix_scores(scores: list[int])-> list[int]:
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

    return dist.tolist()


def select_move(scores):
    dist = softmax(scores, dim=0)
    torch.manual_seed(randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff))
    # Sample from the softmax distribution
    return torch.multinomial(dist, 1)[0].item() + 1


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
    move_list = play_game()
    print(".", end='', flush=True)
    cutoff = randint(0, len(move_list)-2)
    return move_list[:cutoff]

def generate(n: int)-> list[str]:

    games = []
    while len(games) < n:
        with Pool(cpu_count()) as p:
            sampled = p.map(random_position, range(n-len(games)))

        games.extend([''.join(pos) for pos in sampled])
        games = list(set(games))


    with Pool(cpu_count()) as p:
        targets = p.map(target_dist, games)

    print()

    return [{'moves': game, 'target': target} for game, target in zip(games, targets)]
    

def save_json(games: list[str], fname: str):
    with open(fname, 'w') as f:
        json.dump(games, f, indent=4)


def main():
    n_games = int(argv[1])
    fname = argv[2]

    games = generate(n_games)
    print(*games, sep='\n')
    print(f"Generated {len(games)} positions")
    save_json(games, fname)


if __name__ == '__main__':
    main()


    
    
