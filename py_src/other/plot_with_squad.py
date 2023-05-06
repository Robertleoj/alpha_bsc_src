from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

filenames = {
    'Default 20W': 'default_hlr.csv',
    'Default 40W': 'default_big_window.csv',

    'Endgame 20W': 'endgame_normal_window.csv',
    'Endgame 40W': 'endgame_big_window.csv',
}

color_map = {
    "Endgame 20W": "purple",
    "Endgame 40W": "magenta",
    "Default 20W": "blue",
    "Default 40W": "cyan"
    # randomized_cap: green
    # "Dynamic Window": "orange",
    # "Monotone": "purple"
}

# color_map = {
#     "Endgame": "red",
#     # Endgame bw: pink
#     "Default": "blue",
#     # Default bw: cyan
#     # randomized_cap: green
#     "Dynamic Window": "orange",
#     "Monotone": "purple"
# }


fig_path = Path("./squad_figures")
fig_path.mkdir(exist_ok=True)

def plot_errors(df, x, val_col, title, fname) -> None:

    # lineplots by error type
    df = df.query("error_type == @val_col")

    # lineplots by category
    # plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df, 
        x=x, 
        y="error", 
        hue="category",
        palette=color_map,
        linewidth=1
    )

    xlabel = "Generation" if x == "gen" else "Simulations"
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path / fname)
    plt.clf()


dfs = {name: pd.read_csv(filename) for name, filename in filenames.items()}

for name, df in dfs.items():
    df['category'] = name

df = pd.concat(dfs.values())

plot_errors(df, "gen", "rmse_nn_value_error", "Value Error RMSE", "value_error_rmse.png")

plot_errors(df, "gen", "rmse_mcts_value_error", "MCTS Value Error RMSE", "mcts_value_error_rmse.png")

plot_errors(df, "playouts", "rmse_nn_value_error", "Value Error RMSE (Simulations)", "value_error_rmse_playouts.png")

plot_errors(df, "playouts", "rmse_mcts_value_error", "MCTS Value Error RMSE (Simulations)", "mcts_value_error_rmse_playouts.png")







