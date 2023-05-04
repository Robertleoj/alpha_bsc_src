from sys import argv
from pathlib import Path
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# COMP_PATH = Path("../db/competitions/breakthrough")
# PERFECT = True
# GAME = 'Connect4'
# COMP_PATH = Path("../db/competitions/perfect")
# NUM_GAMES = 100
# VS = 'Perfect Agent'

# runs_dirnames = {
#     'Default': 'DEFAULT_high_lr_deterministic',
#     'Monotone': 'MONOTONE_deterministic',
#     'Endgame': 'ENDGAME_normal_window_deterministic',
#     'Endgame big window': 'ENDGAME_PLAYOUTS_BIG_WINDOW_deterministic',
#     'Dynamic Window': 'DYN_WINDOW_high_lr_deterministic',
# }



PERFECT = False
GAME = 'Breakghrough'
COMP_PATH = Path("../db/competitions/breakthrough")
NUM_GAMES = 100
VS = 'Default generation 100'


# runs_dirnames = {
#     "Default":"DEFAULT_5000_gamesvsDEFAULT_5000_games_const_100" ,
#     "Dynamic Window":"DEFAULT_5000_gamesvsDYN_WINDOW_const_1.0", 
#     "Endgame":"DEFAULT_5000_gamesvsENDGAME_PLAYOUTS_uniat100_few_games_const_100" ,
#     "Monotone":"DEFAULT_5000_gamesvsMONOTONE_const_1.0" ,
#     # "DEFAULT_5000_gamesvsrandomized_cap_const_1.0"
# }

runs_dirnames = {
    "Default": "DEFAULT_5000_gamesvsDEFAULT_5000_games_const_100" ,
    "Dynamic Window":"DEFAULT_5000_gamesvsDYN_WINDOW_const_1.0", 
    "Endgame":"DEFAULT_5000_gamesvsENDGAME_PLAYOUTS_uniat100_few_games_const_100" ,
    "Monotone":"DEFAULT_5000_gamesvsMONOTONE_const_1.0" ,
    # "DEFAULT_5000_gamesvsrandomized_cap_const_1.0"
}



# conf = 0.95

# title = f'Breakthrough: Win Rates ({int(conf * 100)}% CI) vs Playouts'


def z_value_from_confidence(confidence_level):
    alpha = 1 - confidence_level
    z_value = norm.ppf(1 - alpha / 2)
    return z_value

def ci_bounds(p_hat):
    se = np.sqrt((p_hat * (1 - p_hat)) / n)
    lower = p_hat - z_alpha_2 * se
    upper = p_hat + z_alpha_2 * se
    return lower, upper

def plot_competition_results(csv: pd.DataFrame, title):
    # make lineplot with seaborn
    # playouts on x axis, 1 - win_rate on y axis

    if not PERFECT:
        csv['win_rate'] = 1 - csv['r1_win_rate']


    plt.figure(figsize=(15, 6))
    p_key = 'playouts' if PERFECT else 'playouts2'
    sns.lineplot(data=csv, x=p_key, y='win_rate', hue='run_name', style='run_name',markers=True, dashes=False)

    # name legend
    plt.legend(title='')

    # set y axis to 0-1
    ytop = 0.5 if PERFECT else 1
    plt.ylim(0, ytop)


    # add line for 50% win rate
    plt.axhline(y=0.5, color='r', linestyle='-')
    # Make vertical line at playouts1

    # get first value of playouts1
    if not PERFECT:
        pl1 = csv['playouts1'][0].iloc[0]
        print(pl1)
        plt.axvline(x=pl1, color='g', linestyle='-')


    plt.xlabel("Playouts")
    plt.ylabel("Win Rate")
    plt.title(title)
    plt.show()

    # save plot
    plt.savefig(COMP_PATH / "together.png")

def plot_competition_results_conf(csv: pd.DataFrame, conf, title):
    # make lineplot with seaborn
    # playouts on x axis, 1 - win_rate on y axis
    if not PERFECT:
        csv['win_rate'] = 1 - csv['r1_win_rate']
    
    # Confidence level (e.g., 95%)
    # z_alpha_2 = 1.96

    z_alpha_2 = z_value_from_confidence(conf)

    # Compute the confidence intervals
    csv['se'] = np.sqrt((csv['win_rate'] * (1 - csv['win_rate'])) / NUM_GAMES)
    csv['lower'] = csv['win_rate'] - z_alpha_2 * csv['se']
    csv['upper'] = csv['win_rate'] + z_alpha_2 * csv['se']

    plt.figure(figsize=(15, 6))
    x_key = 'playouts' if PERFECT else 'playouts2'
    sns.lineplot(data=csv, x=x_key, y='win_rate', hue='run_name', markers=True, dashes=False)

    # Plot the confidence intervals
    for run_name, color in zip(csv['run_name'].unique(), sns.color_palette()):
        temp_df = csv[csv['run_name'] == run_name]
        plt.fill_between(temp_df[x_key], temp_df['lower'], temp_df['upper'], color=color, alpha=0.2)

    # name legend
    plt.legend(title='')

    # set y axis to 0-1
    ytop = 0.5 if PERFECT else 1
    plt.ylim(0, ytop)

    # add line for 50% win rate
    plt.axhline(y=0.5, color='r', linestyle='-')
    
    # get first value of playouts1
    if not PERFECT:
        pl1 = csv['playouts1'].iloc[0]

        # Make vertical line at playouts1
        plt.axvline(x=pl1, color='g', linestyle='-')

    plt.xlabel("Playouts")
    plt.ylabel("Win Rate")
    plt.title(title)
    plt.show()

    # save plot
    plt.savefig(COMP_PATH / f"together_{int(conf * 100)}_conf.png")


def main():
    # global conf
    # global title

    use_conf = False
    if len(argv) > 1:
        use_conf = True
        conf = float(argv[1])
        title = f'{GAME}: Win Rates ({int(conf * 100)}% CI) vs {VS}'
    else:
        title = f'{GAME}: Win Rates vs {VS}'


    # get csv path for each run
    csv_paths = {}
    for name, dirname in runs_dirnames.items():
        csv_paths[name] = COMP_PATH / dirname / "results.csv"

    # get csv data for each run
    csvs = {}
    for name, path in csv_paths.items():
        csvs[name] = pd.read_csv(path)
    
    # make single dataframe, adding column for run name
    df = pd.DataFrame()
    for name, csv in csvs.items():
        csv['run_name'] = name
        df = df.append(csv)

    
    if use_conf:
        plot_competition_results_conf(df, conf, title)
    else:
        plot_competition_results(df, title)
   

if __name__ == "__main__":
    main()