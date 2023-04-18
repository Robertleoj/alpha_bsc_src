from DB import DB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import set_run
from make_num_playouts import make_num_playouts
import sys
import json
from pathlib import Path
import config

import warnings

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if len(sys.argv) < 2:
    print("Usage: python3 evals.py <run_name>")

game_name = "connect4"
run_name = sys.argv[1]

playouts_dict = make_num_playouts(game_name, run_name)

set_run(run_name, game_name)


db = DB()

max_gen = db.newest_generation()

# evals = [db.evals(i) for i in range(max_gen + 1)]

with open("./cpp_hyperparameters.json", "r") as f:
    cpp_config = json.load(f)

max_playouts = cpp_config["search_depth"]


eval_files = Path("./evals").glob("*.json")

gen_playouts = {}
evals = {}
for file in eval_files:
    gen = int(file.stem)
    if gen != 0:

        # db.prefetch_generation(gen - 1)

        gen_playouts[gen] = playouts_dict[gen-1]

        # gen_playouts[gen] = (
        #     load_generation(gen - 1).weights.sum() * max_playouts
        # ).item()
    else:
        gen_playouts[gen] = 0

    with open(file, "r") as f:
        evals[gen] = json.load(f)["evals"]

cols = list(evals[0][0].keys())

if len(evals) != max_gen + 1:
    print("WARNING: evals not found for all generations")

# eval_df = pd.DataFrame(cols = ["gen", "nn_value_error", "mcts_value_error", "prior_error", "mcts_pol_error"])
eval_df = pd.DataFrame(columns=["gen", "error_type", "error"])

mean = lambda j, k: sum([obj[k] for obj in j]) / len(j)

for i, eval in evals.items():
    # print(eval)
    for col in cols:
        if "error" in col:

            eval_df = eval_df.append(
                {"gen": i, "error_type": col, "error": mean(eval, col)},
                ignore_index=True,
            )

    eval_df = eval_df.append(
        {
            "gen": i,
            "error_type": "rmse_nn_value_error",
            "error": mean(eval, "nn_value_error") ** 0.5,
        },
        ignore_index=True,
    )

    eval_df = eval_df.append(
        {
            "gen": i,
            "error_type": "rmse_mcts_value_error",
            "error": mean(eval, "mcts_value_error") ** 0.5,
        },
        ignore_index=True,
    )

print(eval_df.sort_values(by="gen").tail(6))
# for i, eval in enumerate(evals):
#     eval_df = eval_df.append({
#         "gen": i,
#         "nn_value_error": eval.nn_value_error.mean(),
#         "mcts_value_error": eval.mcts_value_error.mean(),
#         "prior_error": eval.prior_error.mean(),
#         "mcts_pol_error": eval.mcts_error.mean()
#     }, ignore_index=True)


def plot_errors(df, x, val_col, mcts_col, title, fname) -> None:
    mcts_df = df.query("error_type == @mcts_col")
    sns.lineplot(
        data=mcts_df, x=x, y="error", linestyle="--", color="blue", label="MCTS"
    )
    val_df = df.query("error_type == @val_col")
    sns.lineplot(
        data=val_df, x=x, y="error", linestyle="-", color="blue", label="NN"
    )  # dashes=[(2, 2), (2, 2)]
    plt.title(title)
    plt.savefig(fig_path / fname)
    plt.clf()


fig_path = Path("./figures")
fig_path.mkdir(exist_ok=True)

eval_df = eval_df.sort_values(by=["gen"])
eval_df["playouts"] = eval_df["gen"].apply(lambda x: gen_playouts[x])
# change to cumulative sum
eval_df["playouts"] = eval_df["playouts"].cumsum()


plot_errors(
    eval_df,
    "gen",
    "nn_value_error",
    "mcts_value_error",
    "Value Error MSE",
    "value_err_mse.png",
)
plot_errors(
    eval_df,
    "playouts",
    "nn_value_error",
    "mcts_value_error",
    "Value Error MSE (Playouts)",
    "value_err_mse_playouts.png",
)

plot_errors(
    eval_df,
    "gen",
    "rmse_nn_value_error",
    "rmse_mcts_value_error",
    "Value Error RMSE",
    "value_err_rmse.png",
)
plot_errors(
    eval_df,
    "playouts",
    "rmse_nn_value_error",
    "rmse_mcts_value_error",
    "Value Error RMSE (Playouts)",
    "value_err_rmse_playouts.png",
)

plot_errors(
    eval_df,
    "gen",
    "policy_prior_error",
    "policy_mcts_error",
    "Policy Error Cross Entropy",
    "pol_err.png",
)
plot_errors(
    eval_df,
    "playouts",
    "policy_prior_error",
    "policy_mcts_error",
    "Policy Error Cross Entropy (Playouts)",
    "pol_err_playouts.png",
)
# eval_df.to_csv(fig_path / "evals2.csv")
