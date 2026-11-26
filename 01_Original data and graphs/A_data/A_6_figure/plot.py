import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
fig, ax = plt.subplots(figsize=(W, H))
order = ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']
means = {a: df[df['series'] == a]['x'].mean() for a in order}
order = sorted(order, key=lambda a: means[a])
ys = np.linspace(0.4, len(order) + 0.6, 220)
for i, a in enumerate(order):
    v = df[df['series'] == a]['x'].values
    c = COL[a]
    if len(v) > 1 and np.std(v) > 0:
        d = stats.gaussian_kde(v, bw_method=0.35)(ys)
        d = (d / d.max()) ** 1.8 * 0.40
        d[d < 0.10] = np.nan
        ax.fill_betweenx(ys, i, i + d, color=c, alpha=0.55, lw=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([i - 0.28, i - 0.28], [q1, q3], color=c, lw=2.0, solid_capstyle='butt')
    ax.plot([i - 0.28], [med], marker='|', ms=5, color=INK, mew=1.0)
    rng = np.random.RandomState(100 + i)
    ax.scatter(i + rng.uniform(-0.32, -0.06, len(v)), v, s=1.8,
               color=c, alpha=0.45, lw=0)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([a for a in order], rotation=90)
for t, a in zip(ax.get_xticklabels(), order):
    t.set_color(COL[a])
ax.set_ylabel('Rank')
clean(ax, 'y')
fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.34)
save(fig, D)
