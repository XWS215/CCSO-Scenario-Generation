import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__))
df = read(D)
metrics = ['Fast', 'Hit', 'Many', 'Efficiency', 'Coverage', 'Stability']
fig, axs = plt.subplots(2, 3, figsize=(W, H))
axes = axs.ravel()
for i, m in enumerate(metrics):
    ax = axes[i]
    for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']:
        s = df[(df['panel'] == m) & (df['series'] == a)].sort_values('x')
        ax.plot(s['x'], s['y'], color=color_of(a), ls='-',
                lw=1.4 if a == 'CCSO' else 0.8,
                marker=mark_of(a), ms=1.8, mew=0.3)
    ax.text(0.04, 0.96, '(%s) %s' % ('abcdef'[i], m),
            transform=ax.transAxes, va='top', ha='left')
    ax.set_xlim(0.6, 10 + 0.4)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 1])
    ax.set_xticks([1, 5, 10])
    clean(ax, 'both')
fig.text(0.012, 0.5, 'P(rank $\\leq$ $x$)', rotation=90, va='center')
h = [Line2D([], [], color=color_of(a), lw=1.0, marker=mark_of(a), ms=2.0,
            label=a) for a in ['CCSO', 'DE', 'SHADE', 'WOA', 'CMA-ES', 'PSO', 'GWO', 'CCO', 'HHO', 'SA']]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.0,
           handletextpad=0.3)
fig.subplots_adjust(left=0.11, right=0.995, top=0.97, bottom=0.34,
                    wspace=0.32, hspace=0.42)
save(fig, D)
