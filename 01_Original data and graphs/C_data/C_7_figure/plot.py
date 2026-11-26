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
for si, p in enumerate(panels):
    ax = axs[si]
    order = sorted(['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA'], key=lambda a: df[(df['panel'] == p) & (df['series'] == a)]['x'].mean())
    ys = np.linspace(0.4, 10.6, 200)
    for i, a in enumerate(order):
        v = df[(df['panel'] == p) & (df['series'] == a)]['x'].values
        if len(v) == 0:
            continue
        if np.std(v) > 0:
            d = stats.gaussian_kde(v, bw_method=0.55)(ys)
            d = (d / d.max()) ** 1.8 * 0.40
            d[d < 0.10] = np.nan
            ax.fill_betweenx(ys, i, i + d, color=COL.get(a, GRAY), alpha=0.55, lw=0)
        ax.plot([i - 0.30, i - 0.30], [np.percentile(v, 25), np.percentile(v, 75)],
                color=COL.get(a, GRAY), lw=1.8)
        ax.plot([i - 0.30], [np.median(v)], marker='|', ms=5, color=INK)
        rng = np.random.RandomState(200 + si * 20 + i)
        ax.scatter(i + rng.uniform(-0.34, -0.06, len(v)), v, s=1.6,
                   color=COL.get(a, GRAY), alpha=0.5, lw=0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for t, a in zip(ax.get_xticklabels(), order):
        t.set_color(COL.get(a, INK2))
    ax.text(0.03, 0.96, '(%s) %s' % ('ab'[si], p), transform=ax.transAxes,
            va='top', ha='left')
    if si == 0:
        ax.set_ylabel('Rank (1 = best)')
    clean(ax, 'y')
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.30, wspace=0.10)
save(fig, D)
