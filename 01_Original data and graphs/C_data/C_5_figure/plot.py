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
fig, axs = plt.subplots(1, 2, figsize=(W, H))
ax = axs[0]
cost = df[df['panel'] == 'cost']
for j, a in enumerate(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']):
    s = cost[cost['series'] == a]
    if len(s) == 0:
        continue
    ax.plot([j, j], [s['y'].min(), s['y'].max()], color=COL.get(a, GRAY), lw=0.8)
    for _, r in s.iterrows():
        ax.scatter(j, r['y'], s=14, marker='o' if r['x'] == 'Multi-vehicle' else 's',
                   color=COL.get(a, GRAY), edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks(np.arange(len(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'])))
ax.set_xticklabels(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'], rotation=90)
ax.set_ylabel('Cost saved')
clean(ax, 'y')
panel(ax, 'a')

ax = axs[1]
sens = df[df['panel'] == 'sensitivity']
for sc, ls in zip(['Multi-vehicle', 'Intersection'], ['-', '--']):
    for h, mk in zip([30, 60, 120], ['o', 's', '^']):
        s = sens[(sens['series'] == sc) & (sens['value'] == h)].sort_values('x')
        if len(s):
            ax.plot(s['x'], s['y'], color=COL['CCSO'] if sc == 'Multi-vehicle' else '#c56526',
                    ls=ls, marker=mk, ms=3, lw=0.9)
ax.set_xlabel('Noise level $\\sigma$')
ax.set_ylabel('Change (%)')
clean(ax, 'both')
panel(ax, 'b')
fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.32, wspace=0.40)
save(fig, D)
