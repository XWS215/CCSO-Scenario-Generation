import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, 'data.csv'))
panels = list(dict.fromkeys(df['panel']))
fig, axs = plt.subplots(1, len(panels), figsize=(W, H), sharey=True)
if len(panels) == 1:
    axs = [axs]
for i, p in enumerate(panels):
    ax = axs[i]
    for a in ['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵', '只保留开局', '只保留中局', '只保留残局']:
        s = df[(df['panel'] == p) & (df['series'] == a)].sort_values('x')
        if len(s) == 0:
            continue
        ax.fill_between(s['x'], s['lo'], s['hi'], color=VCOL[a], alpha=0.14, lw=0)
        ax.plot(s['x'], s['y'], color=VCOL[a], lw=1.4 if a == '完整CCSO' else 0.8)
    ax.text(0.03, 0.95, '(%s) %s' % ('abcd'[i], p), transform=ax.transAxes,
            va='top', ha='left')
    clean(ax, 'both')
    if i == len(panels) // 2:
        ax.set_xlabel('Test time ($10^3$ s)')
    if i == 0:
        ax.set_ylabel('Scenarios found')
h = [Line2D([], [], color=VCOL[a], lw=1.0, label=VSHORT.get(a, a)) for a in ['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵', '只保留开局', '只保留中局', '只保留残局']]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.01), handlelength=1.2, columnspacing=1.0,
           handletextpad=0.3, fontsize=5.2)
fig.subplots_adjust(left=0.10, right=0.995, top=0.98, bottom=0.47, wspace=0.10)
save(fig, D)
