import torch
import sqlite3
import io
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from glob import glob
import bson 


def read_tensors(arg):
    state, policy, outcome, moves_left, weights = arg

    return (
        np.array(tensor_from_blob(state)),
        np.array(tensor_from_blob(policy)),
        np.array(outcome),
        np.array(moves_left),
        np.array(weights)
    )

def tensor_from_blob(blob) -> torch.Tensor:

    with io.BytesIO(blob) as bytesio:
        module_wrapper = torch.jit.load(bytesio, 'cpu')

    return list(module_wrapper.parameters())[0]


class DB:

    def connect(self):

        return sqlite3.connect('./db.db')

    def add_generation(self, generation_num:int):

        query = f"""
            insert into generations (generation_num)
            values ({generation_num})
        """

        self.query(query, True)

    def query(self, query, no_data=False):
        conn = self.connect()
        cursor = conn.execute(query)

        res = None
        if not no_data:
            res = cursor.fetchall()
        else:
            conn.commit()

        conn.close()

        return res

    def evals(self, gen: int):
        query = f"""
            select
                gt.id,
                gt.policy_target, 
                gt.value_target, 
                gt.policy_prior,
                gt.policy_mcts,
                gt.nn_value,
                gt.nn_value_error,
                gt.mcts_value,
                gt.mcts_value_error,
                gt.prior_error,
                gt.mcts_error
            from
                join generations gens
                join ground_truth_evals gt
                    on gt.generation_id = gens.id
            where
                and gens.generation_num = {gen}
        """

        res = self.query(query)
        json_indices = [1, 3, 4]

        res = [
            [
                json.loads(x) if i in json_indices else x 
                for i, x in enumerate(row)
            ]
            for row in res
        ]

        # create dataframe
        df = pd.DataFrame(res, columns=[
                "id",
                "policy_target",
                "value_target",
                "policy_prior",
                "policy_mcts",
                "nn_value",
                "nn_value_error",
                "mcts_value",
                "mcts_value_error",
                "prior_error",
                "mcts_error"
            ])
        return df




    def newest_generation(self) -> int:
        query = f"""
            select 
                max(gens.generation_num)
            from
                generations gens
        """

        res = self.query(query)

        return res[0][0]

    def add_loss(self, generation, iteration, loss):

        query = f"""
            insert into losses (generation_id, iteration, loss)
            values (
                (select id from generations where generation_num = {generation}),
                {iteration},
                {loss}
            )
        """

        self.query(query, True)


    def prefetch_generation(self, generation:int):
        
        game_files = glob(f"./training_data/{generation}/*.bson")
        print(f"Found {len(game_files)} files")

        tuples = []

        for file in game_files:
            with open(file, "rb") as f:
                data = bson.loads(f.read())['samples']

            for row in data:
                tuples.append((
                    row["state"],
                    row["policy"],
                    row["outcome"],
                    row["moves_left"],
                    row["weight"] if "weight" in row else 1
                ))

        print(f"Fetched {len(tuples)} positions")

        with Pool(cpu_count()) as p:
            result = list(tqdm(p.imap_unordered(read_tensors, tuples, chunksize=16), total=len(tuples), desc="Making tensors"))
        
        result = tuple(
            map(np.stack, zip(*result))
        )
      
        return tuple(map(torch.tensor, result))

    def generation_nums(self):
        query = """
            select generation_num from generations
        """

        res = self.query(query)
        return [x[0] for x in res]
        
if __name__ == "__main__":
    db = DB()

    cursor = db.query("""
        select id, generation_id, state, policy, outcome, created_at
        from training_data
    """)

    for (id, g_id, state, policy, outcome, created_at) in cursor:
        policy = tensor_from_blob(policy)
        state = tensor_from_blob(state)

        print(f"{id=}, {g_id=}, {outcome=}, {created_at=}")
        
        print(f"Policy shape: {policy.shape}")

        print(f"State shape: {state.shape}")
