import random
from sys import argv
from pathlib import Path
import json

fpath = Path("../db/connect4_openings.json")

def make_opening():
    return tuple(str(random.randint(1, 7)) for _ in range(4))

def save_openings(openings):
    with open(fpath, "w") as f:
        json.dump(openings, f, indent=4)

def main():
    num_openings = int(argv[1])

    openings = set()

    while len(openings) < num_openings:
        openings.add(make_opening())

    save_openings(list(openings))
    

if __name__ == "__main__":
    main()