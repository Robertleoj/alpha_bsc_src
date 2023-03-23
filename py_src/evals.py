from DB import DB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import set_run
import sys

if len(sys.argv) < 2:
    print("Usage: python3 train.py <run_name>")

game_name = "connect4"

set_run(sys.argv[1], game_name)

db = DB()

max_gen = db.newest_generation()  

evals = [db.evals(game_name, i) for i in range(max_gen + 1)]

# eval_df = pd.DataFrame(cols = ["gen", "nn_value_error", "mcts_value_error", "prior_error", "mcts_pol_error"])
eval_df = pd.DataFrame(columns = ["gen", "error_type", "error"])

for i, eval in enumerate(evals):
    for col in eval.columns:
        if "error" in col:
            eval_df = eval_df.append({
                "gen": i,
                "error_type": col,
                "error": eval[col].mean()
            }, ignore_index=True)

print(eval_df)
# for i, eval in enumerate(evals):
#     eval_df = eval_df.append({
#         "gen": i,
#         "nn_value_error": eval.nn_value_error.mean(),
#         "mcts_value_error": eval.mcts_value_error.mean(),
#         "prior_error": eval.prior_error.mean(),
#         "mcts_pol_error": eval.mcts_error.mean()
#     }, ignore_index=True)

sns.lineplot(data=eval_df.query("error_type == 'mcts_value_error' or error_type == 'nn_value_error'"), x="gen", y="error",hue="error_type")
plt.savefig("value_err.png")
plt.clf()

sns.lineplot(data=eval_df.query("error_type == 'prior_error' or error_type == 'mcts_error'"), x="gen", y="error",hue="error_type")
plt.savefig("pol_evals.png")
