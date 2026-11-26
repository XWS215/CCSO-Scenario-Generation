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
metrics = ['Fast', 'Hit', 'Many', 'Efficiency', 'Coverage', 'Stability']
scenes = list(dict.fromkeys(df['panel']))
algos = [a for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'] if a in set(df['series'])]
M = np.full((len(metrics), len(algos) * len(scenes)), np.nan)
for j, a in enumerate(algos):
    for si, sc in enumerate(scenes):
        for i, k in enumerate(['快', '准', '多', '效率', '全', '稳']):
            q = df[(df['panel'] == sc) & (df['column'] == k) & (df['series'] == a)]
            if len(q):
                M[i, j * len(scenes) + si] = float(q['y'].iloc[0])
fig, ax = plt.subplots(figsize=(W, H))
im = ax.imshow(M, cmap=SEQ, aspect='auto', vmin=1, vmax=len(algos))
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)
ax.set_xticks([j * len(scenes) + (len(scenes) - 1) / 2 for j in range(len(algos))])
ax.set_xticklabels(algos, rotation=90, fontsize=5)
for t, a in zip(ax.get_xticklabels(), algos):
    t.set_color(COL.get(a, INK2))
for j in range(len(algos) + 1):
    ax.axvline(j * len(scenes) - 0.5, color='white', lw=0.8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label('Rank')
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.13, right=0.93, top=0.95, bottom=0.44)
save(fig, D)
