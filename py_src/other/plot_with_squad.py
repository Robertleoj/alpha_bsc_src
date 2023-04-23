from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

filenames = {
    'default': 'default_high_lr.csv',
    'endgame': 'endgame_playouts_big_window_connect4.csv',
    'randomized_cap': 'random_cap.csv',
    'dyn_window': 'dyn_window_high_lr.csv',
    'monotone': "monotone.csv"
}


fig_path = Path("./squad_figures")
fig_path.mkdir(exist_ok=True)

def plot_errors(df, x, val_col, title, fname) -> None:

    # lineplots by error type
    df = df.query("error_type == @val_col")

    # lineplots by category
    sns.lineplot(
        data=df, x=x, y="error", hue="category"
    )

    plt.title(title)
    plt.savefig(fig_path / fname)
    plt.clf()


dfs = {name: pd.read_csv(filename) for name, filename in filenames.items()}

for name, df in dfs.items():
    df['category'] = name

df = pd.concat(dfs.values())

plot_errors(df, "gen", "rmse_nn_value_error", "Value Error RMSE", "value_error_rmse.png")

plot_errors(df, "gen", "rmse_mcts_value_error", "MCTS Value Error RMSE", "mcts_value_error_rmse.png")

plot_errors(df, "playouts", "rmse_nn_value_error", "Value Error RMSE (Playouts)", "value_error_rmse_playouts.png")

plot_errors(df, "playouts", "rmse_mcts_value_error", "MCTS Value Error RMSE (Playouts)", "mcts_value_error_rmse_playouts.png")







