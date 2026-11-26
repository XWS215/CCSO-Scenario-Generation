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
order = ['只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵',
         '只保留开局', '只保留中局', '只保留残局']
vals = {r['series']: r['x'] for _, r in df.iterrows()}
base = vals['完整CCSO']
fig, ax = plt.subplots(figsize=(W, H))
x = np.arange(len(order))
for i, a in enumerate(order):
    ax.plot([i, i], [base, vals[a]], color=VCOL[a], lw=0.8)
    ax.scatter(i, vals[a], s=16, color=VCOL[a], edgecolors='white',
               linewidths=0.4, zorder=3)
ax.axhline(base, color=MUTED, lw=0.8, ls='--')
ax.set_xticks(x)
ax.set_xticklabels([VSHORT[a] for a in order], rotation=90)
for t, a in zip(ax.get_xticklabels(), order):
    t.set_color(VCOL[a])
ax.set_ylabel('Avg. rank')
clean(ax, 'y')
fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.34)
save(fig, D)
