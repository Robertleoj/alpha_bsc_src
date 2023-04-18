from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT = "default_high_lr_connect4.csv"
ENDGAME = "endgame_playouts_big_window_connect4.csv"


def plot_errors(df, x, val_col, mcts_col, title, fname) -> None:

    default_mcts = df.query("error_type == @mcts_col and category == 'default'")
    default_val = df.query("error_type == @val_col and category == 'default'")
    endgame_mcts = df.query("error_type == @mcts_col and category == 'endgame'")
    endgame_val = df.query("error_type == @val_col and category == 'endgame'")

    sns.lineplot(
        data=default_mcts, x=x, y="error", linestyle="--", color="red", label="default MCTS"
    )
    sns.lineplot(
        data=default_val, x=x, y="error", linestyle="-", color="red", label="default NN"
    )

    sns.lineplot(
        data=endgame_mcts, x=x, y="error", linestyle="--", color="blue", label="endgame MCTS"
    )

    sns.lineplot(
        data=endgame_val, x=x, y="error", linestyle="-", color="blue", label="endgame NN"
    )

    plt.title(title)
    plt.savefig(fig_path / fname)
    plt.clf()



default_df = pd.read_csv(DEFAULT)
default_df["category"] = "default"

endgame_df = pd.read_csv(ENDGAME)
endgame_df["category"] = "endgame"

eval_df = pd.concat([default_df, endgame_df])


fig_path = Path("./figures")
fig_path.mkdir(exist_ok=True)

plot_errors(eval_df, "gen", "nn_value_error", "mcts_value_error", "Value Error MSE", "value_err_mse.png")
plot_errors(eval_df, "playouts", "nn_value_error", "mcts_value_error", "Value Error MSE (Playouts)", "value_err_mse_playouts.png")

plot_errors(eval_df, "gen", "rmse_nn_value_error", "rmse_mcts_value_error", "Value Error RMSE", "value_err_rmse.png")
plot_errors(eval_df, "playouts", "rmse_nn_value_error", "rmse_mcts_value_error", "Value Error RMSE (Playouts)", "value_err_rmse_playouts.png")

plot_errors(eval_df, "gen", "policy_prior_error", "policy_mcts_error", "Policy Error Cross Entropy", "pol_evals.png")
plot_errors(eval_df, "playouts", "policy_prior_error", "policy_mcts_error", "Policy Error Cross Entropy (Playouts)", "pol_evals_playouts.png")


# palette = {"default": "red", "endgame": "blue"}

# plot_df = eval_df.query(
#     "error_type == 'mcts_value_error' or error_type == 'nn_value_error'"
# )

# sns.lineplot(
#     data=plot_df, 
#     x="gen", 
#     y="error", 
#     style="error_type", 
#     hue='category',
#     palette=palette
# )

# plt.title("Value Error MSE")
# plt.savefig(fig_path / "value_err_mse.png")
# plt.clf()


# # palette = {"rmse_mcts_value_error": "red", "rmse_nn_value_error": "blue"}
# rmse_df = eval_df.query(
#     "error_type == 'rmse_mcts_value_error' or error_type == 'rmse_nn_value_error'"
# )
# plt.title("Value Error RMSE")

# sns.lineplot(
#     data=rmse_df, 
#     x="gen", 
#     y="error", 
#     style="error_type", 
#     hue='category',
#     palette=palette
# )
# plt.savefig(fig_path / "value_err_rmse.png")
# plt.clf()


# # palette = {"policy_mcts_error": "red", "policy_prior_error": "blue"}
# plot_df = eval_df.query(
#     "error_type == 'policy_prior_error' or error_type == 'policy_mcts_error'"
# )

# sns.lineplot(
#     data=plot_df,
#     x="gen",
#     y="error",
#     style="error_type",
#     hue='category',
#     palette=palette,
# )
# plt.title("Policy Error Cross Entropy")
# plt.savefig(fig_path / "pol_evals.png")
# plt.clf()

# # palette = {"mcts_value_error": "red", "nn_value_error": "blue"}
# sns.lineplot(
#     data=eval_df.query(
#         "error_type == 'mcts_value_error' or error_type == 'nn_value_error'"
#     ),
#     x="playouts",
#     y="error",
#     style="error_type",
#     hue='category',
#     palette=palette,
# )
# plt.title("Value Error MSE Playouts")
# plt.savefig(fig_path / "value_err_mse_playouts.png")
# plt.clf()

# # palette = {"rmse_mcts_value_error": "red", "rmse_nn_value_error": "blue"}
# rmse_df = eval_df.query(
#     "error_type == 'rmse_mcts_value_error' or error_type == 'rmse_nn_value_error'"
# )
# plt.title("Value Error RMSE Playouts")
# sns.lineplot(
#     data=rmse_df,
#     x="playouts", 
#     y="error", 
#     style="error_type", 
#     hue='category', 
#     palette=palette
# )
# plt.savefig(fig_path / "value_err_rmse_playouts.png")
# plt.clf()


# # palette = {"policy_mcts_error": "red", "policy_prior_error": "blue"}
# sns.lineplot(
#     data=eval_df.query(
#         "error_type == 'policy_prior_error' or error_type == 'policy_mcts_error'"
#     ),
#     x="playouts",
#     y="error",
#     style="error_type",
#     hue='category',
#     palette=palette,
# )
# plt.title("Policy Error Cross Entropy Playouts")
# plt.savefig(fig_path / "pol_evals_playouts.png")
# plt.clf()

# eval_df.to_csv(fig_path / "evals2.csv")
