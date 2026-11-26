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
fig, ax = plt.subplots(figsize=(W, H))
y = np.arange(len(df))
ax.barh(y, df['x'], height=0.62, color=[VCOL.get(s, GRAY) for s in df['series']],
        edgecolor='white', linewidth=0.3)
for yy, v in zip(y, df['x']):
    ax.text(v + df['x'].max() * 0.02, yy, '%.2f' % v, va='center', fontsize=5.4)
ax.set_yticks(y)
ax.set_yticklabels([VSHORT.get(s, s) for s in df['series']])
for t, s in zip(ax.get_yticklabels(), df['series']):
    t.set_color(VCOL.get(s, GRAY))
ax.set_xlim(0, df['x'].max() * 1.22)
ax.set_xlabel('Duplicate rate (%)')
ax.annotate('lower is better', xy=(0.99, 1.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=5.4, color=MUTED)
clean(ax, 'x')
fig.subplots_adjust(left=0.22, right=0.98, top=0.93, bottom=0.20)
save(fig, D)
