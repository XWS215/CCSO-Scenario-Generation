import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
algos = list(dict.fromkeys(df['series']))
mean_rank = {a: df[df['series'] == a]['y'].mean() for a in algos}
algos = sorted(algos, key=lambda a: mean_rank[a])
panels = list(dict.fromkeys(df['panel']))
blocks = []
for p in panels:
    cols = list(dict.fromkeys(df[df['panel'] == p]['column']))
    blocks.append((p, cols))
M = []
for a in algos:
    row = []
    for p, cols in blocks:
        for c in cols:
            q = df[(df['panel'] == p) & (df['column'] == c) & (df['series'] == a)]
            row.append(float(q['y'].iloc[0]) if len(q) else np.nan)
        row.append(np.nan)
    M.append(row[:-1])
M = np.array(M, float)
fig, ax = plt.subplots(figsize=(W, H))
im = ax.imshow(M, cmap=SEQ, aspect='auto', vmin=1, vmax=len(algos))
ax.set_yticks(np.arange(len(algos)))
ax.set_yticklabels(algos)
for t, a in zip(ax.get_yticklabels(), algos):
    t.set_color(color_of(a))
ax.set_xticks([])
pos = 0
for p, cols in blocks:
    ax.axvline(pos - 0.5, color='white', lw=0.8)
    ax.text(pos + (len(cols) - 1) / 2, -0.65, p,
            ha='center', va='bottom', fontsize=6, clip_on=False)
    pos += len(cols)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Rank')
cb.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.subplots_adjust(left=0.12, right=0.92, top=0.94, bottom=0.16)
save(fig, D)
