import torch
import sqlite3
import io
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd


def read_tensors(arg):
    state, policy, outcome, moves_left = arg
    return (
        np.array(tensor_from_blob(state)),
        np.array(tensor_from_blob(policy)),
        np.array(outcome),
        np.array(moves_left)
    )

def tensor_from_blob(blob) -> torch.Tensor:

    with io.BytesIO(blob) as bytesio:
        module_wrapper = torch.jit.load(bytesio, 'cpu')

    return list(module_wrapper.parameters())[0]


class DB:

    def connect(self):

        return sqlite3.connect('./db.db')

    def add_generation(self, game:str, generation_num:int):

        game_id = self.get_game_id(game)

        query = f"""
            insert into generations (game_id, generation_num)
            values ({game_id}, {generation_num})
        """

        self.query(query, True)

    def get_game_id(self, game:str):

        query = f"""
            select id from games where game_name = "{game}"
        """

        res = self.query(query)
        return res[0][0]

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

    def evals(self, game:str, gen: int):
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
                games g
                join generations gens
                    on gens.game_id = g.id
                join ground_truth_evals gt
                    on gt.generation_id = gens.id
            where
                g.game_name = "{game}"
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




    def newest_generation(self, game:str) -> int:
        query = f"""
            select 
                max(gens.generation_num)
            from
                generations gens
                join games g
                    on g.id = gens.game_id
            where
                g.game_name = "{game}"
        """

        res = self.query(query)

        return res[0][0]

    def add_loss(self, game: str, generation, iteration, loss):

        game_id = self.get_game_id(game)

        query = f"""
            insert into losses (generation_id, iteration, loss)
            values (
                (select id from generations where game_id = {game_id} and generation_num = {generation}),
                {iteration},
                {loss}
            )
        """

        self.query(query, True)


    def prefetch_generation(self, game:str, generation:int):
        query = f"""
            select 
                t.state, 
                t.policy, 
                t.outcome,
                t.moves_left
            from 
                games g
                join generations gens
                    on gens.game_id = g.id
                join training_data t
                    on t.generation_id = gens.id
            where
                g.game_name = "{game}"
                and gens.generation_num = {generation}
        """

        res = self.query(query)

        print(f"Fetched {len(res)} rows")

        with Pool(cpu_count()) as p:
            result = list(tqdm(p.imap_unordered(read_tensors, res, chunksize=16), total=len(res), desc="Making tensors"))
        
        result = tuple(
            map(np.stack, zip(*result))
        )
      
        return tuple(map(torch.tensor, result))
        
    def get_ids(self, generation: int, game: str) -> list[int]:

        min_gen = max(generation - config['buffer_generations'], 0)
        
        query = f"""
            select t.id
            from 
                games g
                join generations gens
                    on gens.game_id = g.id
                join training_data t
                    on t.generation_id = gens.id
            where
                gens.generation_num 
                    between {min_gen} and {generation} 
                and g.game_name = "{game}"
        """

        res = self.query(query)
        res = [c[0] for c in res]

        return res

    def get_training_sample(self, id):
        query = f"""
            select 
                state, policy, outcome
            from 
                training_data 
            where
                id = {id}
        """

        res = self.query(query)

        for (state, policy, outcome) in res:
            state_t = tensor_from_blob(state)
            policy_t = tensor_from_blob(policy)
            outcome_t = torch.tensor(outcome)
            break

        return state_t, policy_t, outcome_t


    def get_training_samples(self, ids):
        query = f"""
            select 
                state, policy, outcome
            from 
                training_data 
            where
                id in (
                    {",".join(map(str,ids))}
                )
        """


        res = self.query(query)

        states = []
        policies = []
        outcomes = []

        for (state, policy, outcome) in res:
            state_t = tensor_from_blob(state)
            policy_t = tensor_from_blob(policy)
            outcome_t = torch.tensor(outcome)

            states.append(state_t)
            policies.append(policy_t)
            outcomes.append(outcome_t)

        states = torch.stack(states, 0)
        policies = torch.stack(policies, 0)
        outcomes = torch.stack(outcomes, 0)

        return states, policies, outcomes

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
