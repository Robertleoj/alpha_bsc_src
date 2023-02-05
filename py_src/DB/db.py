import mariadb
import torch
import io


def tensor_from_blob(blob):

    bytesio = io.BytesIO(blob)

    module_wrapper = torch.jit.load(bytesio)

    return list(module_wrapper.parameters())[0]


class DB:
    def __init__(self):
        self.conn = mariadb.connect(
            user='user',
            password='password',
            host='127.0.0.1',
            port=3306,
            database='self_play'
        )
    


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

