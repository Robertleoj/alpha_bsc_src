from sys import argv
from pathlib import Path
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

COMP_PATH = Path("../db/competitions")

def plot_competition_results(csv: pd.DataFrame, dir):
    # make lineplot with seaborn
    # playouts on x axis, 1 - win_rate on y axis
    # Columns: run1,gen1,playouts1,run2,gen2,playouts2,r1_white_win_rate,r1_black_win_rate,r1_win_rate

    csv['Total'] = 1 - csv['r1_win_rate']
    csv['White'] = 1 - csv['r1_black_win_rate']
    csv['Black'] = 1 - csv['r1_white_win_rate']

    df_melted = pd.melt(csv, id_vars=['playouts2'], value_vars=['White', 'Black', 'Total'], var_name='rate_type', value_name='win_rate')



    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='playouts2', y='win_rate', hue='rate_type', style='rate_type', markers=True, dashes=False)

    # name legend
    plt.legend(title='')

    # set y axis to 0-1
    plt.ylim(0, 1)

    # add line for 50% win rate
    plt.axhline(y=0.5, color='r', linestyle='-')
    # Make vertical line at playouts1
    plt.axvline(x=csv['playouts1'][0], color='g', linestyle='-')

    plt.xlabel("Playouts")
    plt.ylabel("Win Rate")
    title_path = dir / 'title.txt'
    if title_path.exists():
        with open(title_path, 'r') as f:
            title = f.read()
            plt.title(title)
    else:
        plt.title("Win Rates vs Playouts")
    plt.show()

    # save plot
    save_path = dir / "plot.png"
    plt.savefig(save_path)


def main():
    runs = COMP_PATH.glob("**/results.csv")
    print(runs)
    
    for run in runs:
        save_dir = Path(run).parent
        name = save_dir.name
        print("Generating graph for", name)
        csv = pd.read_csv(run)

        plot_competition_results(csv, save_dir)



if __name__ == "__main__":
    main()