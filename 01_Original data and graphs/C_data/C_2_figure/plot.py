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
fig, axs = plt.subplots(1, len(panels), figsize=(W, H))
if len(panels) == 1:
    axs = [axs]
vmax = df[df['series'] == 'density']['value'].max()
for i, p in enumerate(panels):
    ax = axs[i]
    den = df[(df['panel'] == p) & (df['series'] == 'density')]
    xs = np.sort(den['x'].unique())
    ys = np.sort(den['y'].unique())
    H = den.pivot_table(index='y', columns='x', values='value').values
    pc = ax.pcolormesh(xs, ys, H, cmap=DENS, norm=LogNorm(vmin=1, vmax=vmax),
                       rasterized=True)
    crit = df[(df['panel'] == p) & (df['series'] == 'critical')]
    ax.scatter(crit['x'], crit['y'], s=1.0, color='#c56526', alpha=0.7, lw=0)
    e = df[(df['panel'] == p) & (df['series'] == 'eps')]
    if len(e):
        ax.axhline(float(e['y'].iloc[0]), color=INK, lw=0.8, ls='--')
    ax.text(0.03, 0.96, '(%s) %s' % ('ab'[i], p), transform=ax.transAxes,
            va='top', ha='left')
    ax.set_xlabel('Evaluation duration (s)')
    if i == 0:
        ax.set_ylabel('$\\log_{10} f$')
    else:
        ax.tick_params(labelleft=False)
    clean(ax, 'both')
cb = fig.colorbar(pc, ax=axs, fraction=0.025, pad=0.015)
cb.set_label('Density')
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.10, right=0.93, top=0.94, bottom=0.20, wspace=0.12)
save(fig, D)
