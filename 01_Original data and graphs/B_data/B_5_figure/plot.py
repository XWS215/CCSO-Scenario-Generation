import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
df = df.sort_values('x')
cd = float(df['lo'].iloc[0])
k = len(df)
fig, ax = plt.subplots(figsize=(W, H))
y = np.arange(k)[::-1]
ax.hlines(y, 1, df['x'], color='#e2e1da', lw=0.6)
ax.scatter(df['x'], y, s=18, c=[color_of(s, True) for s in df['series']],
           edgecolors='white', linewidths=0.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([VSHORT.get(s, s) for s in df['series']])
for t, s in zip(ax.get_yticklabels(), df['series']):
    t.set_color(color_of(s, True))
    if s in ('完整CCSO',):
        t.set_fontweight('bold')
ax.set_xlim(1, k + 0.15)
ax.set_xlabel('Average rank (1 = best)')
ax.plot([1, 1 + cd], [k - 0.15, k - 0.15], color=INK, lw=1.0)
ax.text(1 + cd / 2, k - 0.05, 'CD = %.2f' % cd, ha='center', va='bottom')
clean(ax, 'x')
fig.subplots_adjust(left=0.22, right=0.99, top=0.96, bottom=0.20)
save(fig, D)
