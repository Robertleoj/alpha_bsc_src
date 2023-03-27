from DB import DB
from sys import argv
from utils import set_run
import os
from pathlib import Path
from tqdm import tqdm

def main():
    
    if len(argv) < 2:
        print("Usage: python3 train.py <run_name>")

    game_name = "connect4"
    run_name = argv[1]

    cpp_src_path = Path(os.getcwd()).parent / 'cpp_src'

    set_run(run_name, game_name)

    run_path = os.getcwd()

    db = DB()

    generations = db.generation_nums()

    tq = tqdm(generations)

    for gen in tq:

        tq.desc = f"Deleting"
        os.chdir(run_path)
        db.delete_eval(gen)

        os.chdir(cpp_src_path)
        tq.desc = f"Evaluating"
        ret_code = os.system(f"./eval_agent {run_name} {gen} > /dev/null")
        if ret_code != 0:
            print("Error in evaluation")
            return
    

if __name__ == "__main__":
    main()

