import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, 'data.csv'))
panels = list(dict.fromkeys(df['panel']))
fig, axs = plt.subplots(1, len(panels), figsize=(W, H), sharey=True)
if len(panels) == 1:
    axs = [axs]
for i, p in enumerate(panels):
    ax = axs[i]
    for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']:
        v = df[(df['panel'] == p) & (df['series'] == a)]['y'].values
        if len(v) == 0:
            continue
        j = ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'].index(a)
        rng = np.random.RandomState(11 + j)
        ax.scatter(j + rng.uniform(-0.30, 0.30, len(v)), v, s=1.6,
                   color=COL.get(a, GRAY), alpha=0.6, lw=0)
        ax.scatter(j, np.median(v), s=20, marker=MARK.get(a, "o"), color=COL.get(a, GRAY),
                   edgecolors='white', linewidths=0.4, zorder=4)
    e = df[(df['panel'] == p) & (df['series'] == '__eps__')]
    if len(e):
        ax.axhline(float(e['y'].iloc[0]), color=INK, lw=0.9, ls='--')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'])))
    ax.set_xticklabels(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'], rotation=90)
    ax.text(0.03, 0.96, '(%s) %s' % ('ab'[i], p), transform=ax.transAxes,
            va='top', ha='left')
    if i == 0:
        ax.set_ylabel('Best risk value $f$ (s)')
    clean(ax, 'y')
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.28, wspace=0.10)
save(fig, D)
