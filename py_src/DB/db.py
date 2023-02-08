import mariadb
import torch
import io
import config



def tensor_from_blob(blob) -> torch.Tensor:

    bytesio = io.BytesIO(blob)

    module_wrapper = torch.jit.load(bytesio)

    return list(module_wrapper.parameters())[0]


class DB:

    def connect(self):
        return mariadb.connect(
            user='user',
            password='password',
            host='127.0.0.1',
            port=3306,
            database='self_play'
        )


    def __init__(self):
        pass
        # self.conn = mariadb.connect(
        #     user='user',
        #     password='password',
        #     host='127.0.0.1',
        #     port=3306,
        #     database='self_play'
        # )

    def get_ids(self, generation: int, game: str) -> list[int]:

        min_gen = max(generation - config.buffer_generations, 0)
        
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

        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query)

        # print(results)

        res = [c[0] for c in cursor]
        cursor.close()
        conn.close()
        return res

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


        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query)

        states = []
        policies = []
        outcomes = []

        # print("making tensors")
        for (state, policy, outcome) in cursor:
            state_t = tensor_from_blob(state)
            policy_t = tensor_from_blob(policy)
            outcome_t = torch.tensor(outcome)

            states.append(state_t)
            policies.append(policy_t)
            outcomes.append(outcome_t)

        cursor.close()
        conn.close()
        # print("made tensors")


        states = torch.stack(states, 0)
        policies = torch.stack(policies, 0)
        outcomes = torch.stack(outcomes, 0)

        return states, policies, outcomes

if __name__ == "__main__":
    db = DB()

    cursor = db.conn.cursor()
    cursor.execute("""
        select id, generation_id, state, policy, outcome, created_at
        from training_data
    """)

    for (id, g_id, state, policy, outcome, created_at) in cursor:
        policy = tensor_from_blob(policy)
        state = tensor_from_blob(state)

        print(f"{id=}, {g_id=}, {outcome=}, {created_at=}")
        
        print(f"Policy shape: {policy.shape}")

        print(f"State shape: {state.shape}")

