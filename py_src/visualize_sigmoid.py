import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')

phi = (5 ** 0.5 + 1) / 2



# endgame rerun
# uniat = 100
# unicons = -4
# base = 16
# power = 2
# slop = 5.1
# min_playouts = 50
# max_playouts = 800
# game = "Breakghrough"

# connect4 canon
uniat = 100
unicons = -4
base = 10.5
power = 2
slop = 3.0
b = 0.595
min_playouts = 20
max_playouts = 600
game = "Connect4"


# breakthrough
# max_moves = 110
# max_gen = 100

# connect4
max_moves = 42
max_gen = 100

def shift(gen):

    return (unicons  - base) * (gen / uniat) ** power + base

def f(x, gen):
    return 1 / (1 + np.exp(shift(gen) - (x / slop)))

def plot():
    xs = np.arange(0, max_moves, 0.5)

    gens = range(0, max_gen + 1, 10)

    data = {
        "x" : xs,
    }

    for gen in gens:
        data[str(gen)] = f(xs, gen).clip(min_playouts/max_playouts,1)

    df = pd.DataFrame(data)

    # Melt the DataFrame
    df_melted = pd.melt(df, id_vars=['x'], value_vars=[str(gen) for gen in gens], var_name='line', value_name='weight')

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='x', y='weight', hue='line', style='line', dashes=False)
    # set ylim to 0 to 1
    plt.ylim(0, 1)

    # sns.lineplot(x=xs, y=50/800, color='r', linestyle='-')
    plt.xlabel("Moves Played")
    plt.ylabel("Weight")
    plt.title(f"Endgame Weighting for {game}")
    plt.legend(title='Generation')
    # plt.show()

    plt.savefig(f"endgame_weighting_{game}.png")

def main():
    plot()

if __name__ == '__main__':
    main()
