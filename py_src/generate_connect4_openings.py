import random
from sys import argv
from pathlib import Path
import player
import json

fpath = Path("../db/connect4_openings.json")

def make_opening():
    p = player.VanillaPlayer(3750, 'connect4')

    moves = []

    for _ in range(4):
        m = p.get_and_make_move()
        moves.append(m)

    print(moves)
    return tuple(moves)

def save_openings(openings):
    with open(fpath, "w") as f:
        json.dump(openings, f, indent=4)

def main():

    num_openings = int(argv[1])

    openings = set()

    while len(openings) < num_openings:
        openings.add(make_opening())
        print(f"{len(openings)=}")

    save_openings(list(openings))
    

if __name__ == "__main__":
    main()