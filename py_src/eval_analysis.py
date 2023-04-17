from utils import set_run
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

set_run("ENDGAME_PLAYOUTS_BIG_WINDOW", 'connect4')

gen = 100
df = pd.read_csv('td_evaluations.csv').query("generation == @gen")

figpath = Path('figures')
figpath.mkdir(exist_ok=True)


sns.lineplot(
    data=df.groupby(['moves_left'])['gt_eval'].mean().reset_index(),
    x='moves_left',
    y='gt_eval',
    color='red',
    label='Ground Truth'
)
sns.lineplot(
    data=df.groupby(['moves_left'])['game_outcome'].mean().reset_index(),
    x='moves_left',
    y='game_outcome',
    color='blue',
    label='Game Outcome'
)

plt.title(f"Ground Truth vs Game Outcome (Gen {gen})")
plt.savefig(figpath / 'moves_left_gt_vs_outcome.png')

# df.groupby('moves_played')['gt_eval'].mean().reset_index().plot(x='moves_left', y='gt_eval', kind='line')
