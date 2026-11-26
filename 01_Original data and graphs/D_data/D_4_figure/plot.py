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
df = df.sort_values('x')
fig, ax = plt.subplots(figsize=(W, H))
x = np.arange(len(df))
ax.vlines(x, df['lo'], df['hi'], color='#c3c2b7', lw=1.0)
ax.scatter(x, df['x'], s=18, c=[VCOL[s] for s in df['series']],
           edgecolors='white', linewidths=0.4, zorder=3)
ax.axhline(0, color=INK2, lw=0.6, ls='--')
ax.set_xticks(x)
ax.set_xticklabels([VSHORT[s] for s in df['series']], rotation=90)
for t, s in zip(ax.get_xticklabels(), df['series']):
    t.set_color(VCOL[s])
ax.set_ylabel('$\\Delta$ mean rank')
clean(ax, 'y')
fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.34)
save(fig, D)
