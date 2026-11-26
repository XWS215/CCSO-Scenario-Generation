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
fig, ax = plt.subplots(figsize=(W, H))
for a in ['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵', '只保留开局', '只保留中局', '只保留残局']:
    s = df[df['series'] == a].sort_values('x')
    ax.plot(s['x'], s['y'], color=VCOL[a], lw=1.4 if a == '完整CCSO' else 0.8,
            marker=VMARK[a], ms=2.0, mew=0.3)
ax.set_xlim(0.6, len(['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵', '只保留开局', '只保留中局', '只保留残局']) + 0.4)
ax.set_ylim(-0.04, 1.06)
ax.set_xlabel('Rank $r$')
ax.set_ylabel('P(rank $\\leq$ r)')
clean(ax, 'both')
h = [Line2D([], [], color=VCOL[a], lw=1.0, marker=VMARK[a], ms=2.0,
            label=VSHORT[a]) for a in ['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮', '只保留兵', '只保留开局', '只保留中局', '只保留残局']]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.34)
save(fig, D)
