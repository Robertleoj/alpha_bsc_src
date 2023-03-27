from DB import DB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import set_run
import sys
import json
from pathlib import Path

import warnings

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if len(sys.argv) < 2:
    print("Usage: python3 train.py <run_name>")

game_name = "connect4"

set_run(sys.argv[1], game_name)

db = DB()

max_gen = db.newest_generation()  

# evals = [db.evals(i) for i in range(max_gen + 1)]

eval_files = Path("./evals").glob("*.json")
evals = {}
for file in eval_files:
    gen = int(file.stem)
    with open(file, "r") as f:
        evals[gen] = json.load(f)['evals']

cols = list(evals[0][0].keys())

if len(evals) != max_gen + 1:
    print("WARNING: evals not found for all generations")

# eval_df = pd.DataFrame(cols = ["gen", "nn_value_error", "mcts_value_error", "prior_error", "mcts_pol_error"])
eval_df = pd.DataFrame(columns = ["gen", "error_type", "error"])

mean = lambda j, k: sum([obj[k] for obj in j]) / len(j)

for i, eval in evals.items():
    # print(eval)
    for col in cols:
        if "error" in col:
            
            eval_df = eval_df.append({
                "gen": i,
                "error_type": col,
                "error": mean(eval, col)
            }, ignore_index=True)

    eval_df = eval_df.append({
        "gen": i,
        "error_type": "rmse_nn_value_error",
        "error": mean(eval, "nn_value_error") ** 0.5
    }, ignore_index=True)

    eval_df = eval_df.append({
        "gen": i,
        "error_type": "rmse_mcts_value_error",
        "error": mean(eval, "mcts_value_error") ** 0.5
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

fig_path = Path("./figures")
fig_path.mkdir(exist_ok=True)

palette = {'mcts_value_error': "red", 'nn_value_error': 'blue'}
sns.lineplot(data=eval_df.query("error_type == 'mcts_value_error' or error_type == 'nn_value_error'"), x="gen", y="error",hue="error_type", palette=palette)
plt.title("Value Error MSE")
plt.savefig(fig_path / "value_err_mse.png")
plt.clf()

palette = {'rmse_mcts_value_error': "red", 'rmse_nn_value_error': 'blue'}
rmse_df = eval_df.query("error_type == 'rmse_mcts_value_error' or error_type == 'rmse_nn_value_error'")
plt.title("Value Error RMSE")
sns.lineplot(data=rmse_df, x="gen", y="error",hue="error_type", palette=palette)
plt.savefig(fig_path / "value_err_rmse.png")
plt.clf()


palette = {'policy_mcts_error': "red", 'policy_prior_error': 'blue'}
sns.lineplot(data=eval_df.query("error_type == 'policy_prior_error' or error_type == 'policy_mcts_error'"), x="gen", y="error",hue="error_type", palette=palette)
plt.title("Policy Error Cross Entropy")
plt.savefig(fig_path / "pol_evals.png")
