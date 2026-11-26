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
for i, p in enumerate(['a', 'b']):
    ax = axs[i]
    s = df[df['panel'] == p]
    for _, r in s.iterrows():
        ax.scatter(r['x'], r['y'], s=24 if r['series'] == 'CCSO' else 14,
                   marker=MARK.get(r['series'], 'o'), color=COL.get(r['series'], GRAY),
                   edgecolors='white', linewidths=0.3, zorder=3)
        ax.annotate(r['series'], (r['x'], r['y']), fontsize=4.5,
                    xytext=(2, 2), textcoords='offset points', color=COL.get(r['series'], INK2))
    ax.set_xlabel('Efficiency' if p == 'a' else 'Many')
    ax.set_ylabel('Stability' if p == 'a' else 'Efficiency')
    clean(ax, 'both')
    panel(ax, p)
fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.22, wspace=0.35)
save(fig, D)
