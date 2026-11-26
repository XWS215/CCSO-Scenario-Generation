import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
fig, axs = plt.subplots(1, 3, figsize=(W, H), sharey=True)
panels = sorted(df['panel'].unique(), key=lambda s: int(s.split('=')[-1]))
for i, p in enumerate(panels):
    ax = axs[i]
    for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']:
        s = df[(df['panel'] == p) & (df['series'] == a)].sort_values('x')
        ax.plot(s['x'], s['y'], color=COL[a],
                lw=1.3 if a == 'CCSO' else 0.8,
                ls='-' if a == 'CCSO' else '--')
    ax.set_title('(%s) %s' % ('abc'[i], p), loc='left')
    clean(ax, 'both')
    ax.set_xlabel('Billed evaluations')
    if i == 0:
        ax.set_ylabel('P(hit > t)')
h = [Line2D([], [], color=COL[a], lw=1.0, label=a) for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.30, wspace=0.12)
save(fig, D)
