import torch
import sqlite3
import io
import os
import json
import numpy as np
import gc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from glob import glob
import bson 
from utils import sevenzip_cmd

def read_tensors(arg):
    state, policy, outcome, moves_left, weights = arg

    return (
        np.array(tensor_from_blob(state)),
        np.array(tensor_from_blob(policy).reshape(-1)),
        np.array(outcome),
        np.array(moves_left),
        np.array(weights)
    )

def tensor_from_blob(blob) -> torch.Tensor:

    with io.BytesIO(blob) as bytesio:
        module_wrapper = torch.jit.load(bytesio, 'cpu')

    return list(module_wrapper.parameters())[0]


class DB:

    def __connect(self):

        return sqlite3.connect('./db.db')

    def add_generation(self, generation_num:int):

        query = f"""
            insert into generations (generation_num)
            values ({generation_num})
        """

        self.query(query, True)

    def query(self, query, no_data=False):
        conn = self.__connect()
        cursor = conn.execute(query)

        res = None
        if not no_data:
            res = cursor.fetchall()
        else:
            conn.commit()

        conn.close()

        return res

    def generation_id(self, gen):
        query = f"""
            select id
            from generations
            where generation_num = {gen}
        """

        query_res = self.query(query)
        try:
            return query_res[0][0]
        except:
            print("Query result:")
            print(query_res)
            raise Exception(f"Could not find generation {gen}")

    def delete_eval(self, gen):
        gen_id = self.generation_id(gen)
        query = f"""
            delete from ground_truth_evals where generation_id = {gen_id}
        """
        self.query(query, True)

    def evals(self, gen: int):

        with open(f"./evals/{gen}.json", "r") as f:
            return json.load(f)['evals']


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

    def compress_generation(self, generation):
        sz_cmd = sevenzip_cmd()

        if not os.path.exists(f"./training_data/{generation}.7z"):
            print("Compressing training data...")

            ret_code = os.system(f"{sz_cmd} a -bso0 -bsp0 ./training_data/{generation}.7z ./training_data/{generation}")

            if ret_code != 0:
                print("WARNING: failed to compress, will not delete uncompressed training_data")
            else:
                os.system(f"rm -r ./training_data/{generation}")
        else:
            os.system(f"rm -r ./training_data/{generation}")

    def uncompress_generation(self, generation):
        sz_cmd = sevenzip_cmd()
        if os.path.exists(f"./training_data/{generation}.7z"):
            print("Found compressed data, extracting...")
            ret_code = os.system(f"{sz_cmd} x -bso0 -bsp0 ./training_data/{generation}.7z -o./training_data/ -y")
            if ret_code != 0:
                raise Exception("Could not extract data")

    def prefetch_generation(self, generation:int):
        
        self.uncompress_generation(generation)
       
        game_files = glob(f"./training_data/{generation}/*.bson")
        print(f"Found {len(game_files)} files")

        tuples = []

        for file in game_files:
            with open(file, "rb") as f:
                try:
                    data = bson.loads(f.read())['samples']
                except:
                    raise Exception(f"Could not read file\n{file}")

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

        del tuples
        gc.collect()
        
        result = tuple(
            map(np.stack, zip(*result))
        )

        self.compress_generation(generation)
      
        return tuple(map(torch.tensor, result))


    def generation_nums(self) -> list[int]:
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
