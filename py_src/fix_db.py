from utils import set_run
from DB import DB, read_tensors, tensor_from_blob
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
import bson
from pathlib import Path
from os import system
import os

print("Remember to comment out the magic line in db configuration file!")

input("Press enter to confirm that you have used magic.")


if len(sys.argv) < 2:
    print("Usage: python3 train.py <run_name> [<game>]")

run_name = sys.argv[1]

game = 'connect4'

if(len(sys.argv) == 3):
    game = sys.argv[2]

set_run(run_name, game)

def get_generation(generation:int):
    db = DB()

    query = f"""
        select 
            t.state, 
            t.policy, 
            t.outcome,
            t.moves,
            t.player,
            t.moves_left
        from 
            games g
            join generations gens
                on gens.game_id = g.id
            join training_data t
                on t.generation_id = gens.id
        where
            gens.generation_num = {generation}
            and g.game_name = "connect4"
    """

    res = db.query(query)

    print(f"Fetched {len(res)} rows")

    return res

def get_generation_nums():
    db = DB()
    query = f"""
        select 
            generation_num
        from 
            generations gens
            join games g
                on gens.game_id = g.id
        where 
            g.game_name = "connect4"
    """

    res = db.query(query)

    print(f"Num generations: {len(res)}")

    return [x[0] for x in res]


generations = get_generation_nums() 

for gen_num in generations:
    print(f"Processing generation {gen_num}")


    path = Path(f"./training_data/{gen_num}")
    path.mkdir(parents=True, exist_ok=True)

    file_name = path / "all.bson"

    if os.path.exists(file_name):
        print("Already copied this data. Continuing...")
        continue

    data = get_generation(gen_num)

    samples = []

    for row in data:
        
        samples.append({
            'state': row[0],
            'policy': row[1],
            'outcome': row[2],
            'moves': row[3],
            'player': row[4],
            'moves_left': row[5]
        })

    obj = {"samples": samples}
    dumped = bson.dumps(obj)
    with open(file_name, 'wb') as f:
        f.write(dumped)



# generations table
gens_fields = [
    'id',
    'generation_num',
    'created_at',
]
gens_query = f"select {','.join(gens_fields)} from generations where game_id = 1"
system(f"sqlite3 -csv db.db '{gens_query}' > generations.csv")

# losses table
loss_fields = [
    'id',
    'generation_id',
    'iteration',
    'loss',
    'created_at'
]

loss_query = f"select {','.join(loss_fields)} from losses"
system(f"sqlite3 -csv db.db '{loss_query}' > losses.csv")

# ground truth evals table
gt_fields = [
    "id",
    "generation_id",
    "moves",
    "search_depth",
    "policy_target",
    "value_target",
    "policy_prior",
    "policy_mcts",
    "nn_value",
    "nn_value_error",
    "mcts_value",
    "mcts_value_error",
    "prior_error",
    "mcts_error",
    "created_at",
]

gt_query = f"select {','.join(gt_fields)} from ground_truth_evals"
system(f"sqlite3 -csv  db.db '{gt_query}' > gt.csv")


system("sqlite3 db2.db < ../../../db/configure_db.sql")

def import_file(table_name, file_name):
    system(f"sqlite3 db2.db '.mode csv' '.import {file_name} {table_name}'")


import_file('generations', 'generations.csv')
import_file('losses', 'losses.csv')
import_file('ground_truth_evals', 'gt.csv')

system("mv db.db db_old.db")
system("mv db2.db db.db")

system("rm *.csv")

print("Check that everything is ok, and then delete db_old.db")
# sqlite3 database.db ".mode csv" ".import /path/to/csv/file.csv table_name"