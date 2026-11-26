import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotstyle import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, 'data.csv'))
metrics = ['Fast', 'Hit', 'Many', 'Efficiency', 'Coverage', 'Stability']
variants = ['完整CCSO', '只保留帅', '只保留车', '只保留马', '只保留炮',
            '只保留兵', '只保留开局', '只保留中局', '只保留残局']
M = np.full((len(metrics), len(variants)), np.nan)
for i, m in enumerate(metrics):
    for j, a in enumerate(variants):
        q = df[(df['panel'] == m) & (df['series'] == a)]
        if len(q):
            M[i, j] = float(q['y'].iloc[0])
fig, ax = plt.subplots(figsize=(W, H))
im = ax.imshow(M, cmap=SEQ, aspect='auto', vmin=1, vmax=len(variants))
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)
ax.set_xticks(np.arange(len(variants)))
ax.set_xticklabels([VSHORT[a] for a in variants], rotation=90)
for t, a in zip(ax.get_xticklabels(), variants):
    t.set_fontweight('bold' if a == '完整CCSO' else 'normal')
for i in range(len(metrics)):
    for j in range(len(variants)):
        v = M[i, j]
        ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=5.5,
                color='white' if v > 0.62 * len(variants) else INK)
ax.set_xticks(np.arange(-0.5, len(variants), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
ax.grid(which='minor', color='white', lw=0.8)
ax.tick_params(which='minor', length=0)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Avg. rank')
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.26)
save(fig, D)
