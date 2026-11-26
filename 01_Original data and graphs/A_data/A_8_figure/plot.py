import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
fig, axs = plt.subplots(1, 2, figsize=(W, H))
ax = axs[0]
for a in ALGOS:
    s = df[(df['panel'] == 'profile') & (df['series'] == a)].sort_values('x')
    ax.plot(s['x'], s['y'], color=COL[a], lw=1.3 if a == 'CCSO' else 0.8,
            marker=MARK[a], ms=2.0, mew=0.3)
ax.set_xlabel('Parameter axis')
ax.set_ylabel('Coverage')
clean(ax, 'y')
panel(ax, 'a')

ax = axs[1]
s = df[df['panel'] == 'total'].sort_values('y')
x = np.arange(len(s))
ax.vlines(x, 0, s['y'], color=[COL[a] for a in s['series']], lw=0.8)
ax.scatter(x, s['y'], s=12, c=[COL[a] for a in s['series']],
           edgecolors='white', linewidths=0.3)
ax.set_xticks(x)
ax.set_xticklabels(s['series'], rotation=90)
for t, a in zip(ax.get_xticklabels(), s['series']):
    t.set_color(COL[a])
ax.set_ylabel('Total coverage')
clean(ax, 'y')
panel(ax, 'b')
fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.32, wspace=0.35)
save(fig, D)
