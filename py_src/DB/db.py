import bz2
import lzma
import torch
import sqlite3
import io
import os
import json
import numpy as np
import uuid
import random
import gc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from glob import glob
import bson 
from utils import sevenzip_cmd, Data
from pathlib import Path
from config import config

# need to fit (dl_threads) * (chunk_size) * (chunk_sampled) into memory

def read_tensors(arg):
    state, policy, outcome, moves_left, weights = arg

    # return (
    #     np.array(tensor_from_blob(state)),
    #     np.array(tensor_from_blob(policy).reshape(-1)),
    #     np.array(outcome),
    #     np.array(moves_left),
    #     np.array(weights)
    # )

    return (
        tensor_from_blob(state),
        tensor_from_blob(policy).reshape(-1),
        outcome,
        moves_left,
        weights
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

    def save_file_tensors(self, arg):
        file, generation = arg
        chunk_name = Path(file).stem


        with open(file, "rb") as f:
            try:
                data = bson.loads(f.read())['samples']
            except:
                raise Exception(f"Could not read file\n{file}")

        # save all the data
        for i, row in enumerate(data):

            row =(
                row["state"],
                row["policy"],
                row["outcome"],
                row["moves_left"],
                row["weight"] if "weight" in row else 1
            )

            row = read_tensors(row)

            row = Data(
                state=row[0],
                policy=row[1],
                outcome=row[2],
                moves_left=row[3],
                weight=row[4]
            )

            state_path = Path(f"./cached_data/tmp/{generation}/{chunk_name}/{i}.pt")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            torch.save(row, buffer)
            with bz2.open(state_path, "wb") as f:
                f.write(buffer.getvalue())


        return len(data)


    def prefetch_generation(self, generation:int):

        if os.path.exists(f"./cached_data/{generation}"):
            return

        self.uncompress_generation(generation)
       
        game_files = glob(f"./training_data/{generation}/*.bson")
        print(f"Found {len(game_files)} files")

        # tuples = []

        total_positions = 0
        # with Pool(cpu_count()) as p:
        #     result = list(tqdm(p.imap_unordered(read_tensors, tuples, chunksize=16), total=len(tuples), desc="Making tensors"))

        tasks = [(file, generation) for file in game_files]
            
        with Pool(cpu_count()//2) as p:
            total_positions = sum(tqdm(p.imap_unordered(self.save_file_tensors, tasks, chunksize=16), total=len(game_files), desc="Making tensors"))
            

        print(f"Fetched {total_positions} positions")

        self.compress_generation(generation)

        # make chunky data
        self.make_chunked(generation)

        self.delete_unchunked(generation)
        
      
        # return tuple(map(torch.tensor, result))

    def delete_unchunked(self, generation):
        
        os.system(f'rm -r ./cached_data/tmp/{generation}')

    def make_chunked(self, generation):
        file_names = glob(f"./cached_data/tmp/{generation}/*/*.pt")
        random.shuffle(file_names)
        
        chunk_size = config['cache_chunk_size']
        chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]
        
        args = [(chunk, generation) for chunk in chunks]

        with Pool(cpu_count()) as p:
            list(tqdm(p.imap_unordered(self.make_chunk, args), total=len(chunks), desc="Making chunks"))


    def make_chunk(self, args):

        files, generation = args

        if len(files) == 0:
            return

        # uuid chunk name
        chunk_name = uuid.uuid4().hex
        chunk_path = Path(f"./cached_data/{generation}/{chunk_name}.pt")
        chunk_path.parent.mkdir(parents=True, exist_ok=True)

        if chunk_path.exists():
            return

        chunk = []
        for file in files:
            with bz2.open(file, "rb") as f2:
                chunk.append(torch.load(f2))

        # buffer = io.BytesIO()
        # torch.save(chunk, buffer)                   
        with bz2.open(chunk_path, "wb") as f:
            torch.save(chunk, f)
            # f.write(buffer.getvalue())


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
