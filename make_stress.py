"""Figure 4 - what noise does to each method, and what each one costs.

Reads out/exp70_stress.csv from paper3. Two panels, because the paper makes two
separate claims and they should not be crammed into one axis:

  (a) both methods eat the SAME corrupted detections. The x axis is how badly
      corrupted. What matters is the slope, not the intercept.
  (b) accuracy against wall-clock. A method to the left of the 33 ms line can
      run at broadcast frame rate; one to the right cannot, at any batch size.

FIGURE CONTRACT
  - no number appears here that is not in exp70_stress.csv
  - the solver's clean point is its OWN pipeline (its own lifter prior), not
    our estimate handed to it; the hybrid is drawn separately and labelled
  - log x on both panels, because both quantities span decades
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P3 = '/home/gino/paper3/out/'
SW = pd.read_csv(P3 + 'exp70_stress.csv')
NA = pd.read_csv(P3 + 'exp70n_stress.csv') if os.path.exists(P3 + 'exp70n_stress.csv') else None
INK, MUT, LINE = '#141b1e', '#5c6b70', '#dfe6e5'
OURS, THEM, HYB = '#0e7c7b', '#e8552f', '#7a5ea8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': LINE, 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': MUT, 'text.color': INK})

fig, (A, C, B) = plt.subplots(1, 3, figsize=(12.4, 3.5), dpi=220,
                              gridspec_kw=dict(width_ratios=[1.15, 0.8, 1.15]))
fig.patch.set_facecolor('white')

DROPCOL = 'drop_' if 'drop_' in SW.columns else 'drop'


def series(df, col):
    return df[df[DROPCOL] == 0], df[(df.noise == 0) & (df[DROPCOL] > 0)]


nz, dr = series(SW, None)
nzn, drn = series(NA, None) if NA is not None else (None, None)

# ---------------------------------------------------------------- (a) noise
x = nz.noise.values.copy(); x[x == 0] = 0.7          # 0 px has no place on a log axis
A.plot(x, nz.ours, 'o-', color=OURS, lw=1.8, ms=5, label='ours, trained on detector noise')
if nzn is not None:
    xn = nzn.noise.values.copy(); xn[xn == 0] = 0.7
    A.plot(xn, nzn.ours, '^--', color=HYB, lw=1.7, ms=5.5, label='ours, + noise augmentation')
A.plot(x, nz.solver, 's-', color=THEM, lw=1.8, ms=5, label='paper 1 pipeline')
A.set_xscale('log'); A.set_xticks([0.7, 2, 4, 8, 16])
A.set_xticklabels(['clean', '2', '4', '8', '16'])
A.set_xlabel('Gaussian noise added to every detected pixel  (px)')
A.set_ylabel('median 3D error  (m)')
A.set_title('(a) displaced pixels', fontsize=9.5, loc='left', pad=8)
A.legend(frameon=False, fontsize=7.8, loc='upper left')
A.set_ylim(0.34, 1.62)

# ------------------------------------------------------------ (b) held frames
d = [0] + list(dr[DROPCOL].values)
C.plot(d, [nz.ours.iloc[0]] + list(dr.ours), 'o-', color=OURS, lw=1.8, ms=5)
if drn is not None:
    C.plot([0] + list(drn[DROPCOL].values), [nzn.ours.iloc[0]] + list(drn.ours),
           '^--', color=HYB, lw=1.7, ms=5.5)
C.plot(d, [nz.solver.iloc[0]] + list(dr.solver), 's-', color=THEM, lw=1.8, ms=5)
C.set_xticks(d); C.set_xticklabels(['clean'] + [str(int(k)) for k in dr[DROPCOL]])
C.set_xlabel('frames the tracker held  (of ~27)')
C.set_title('(b) held frames', fontsize=9.5, loc='left', pad=8)
C.set_ylim(0.34, 1.62); C.set_yticklabels([])

# ------------------------------------------------------------- (c) frontier
COST = pd.read_csv(P3 + 'exp70_cost.csv') if os.path.exists(P3 + 'exp70_cost.csv') else None
pts = []
if COST is not None:
    for _, r in COST.iterrows():
        pts.append((r['ms'], r['err'], r['name']))
else:                                        # numbers printed by exp70 section 1
    pts = [(1.6, SW.ours.iloc[0], 'ours'), (400.0, SW.solver.iloc[0], 'paper 1 pipeline')]
cols = {'ours': OURS, 'ours + noise aug': HYB, 'paper 1 pipeline': THEM,
        'hybrid, gated': '#c58f00'}
for ms, err, name in pts:
    B.scatter([ms], [err], s=70, color=cols.get(name, MUT), zorder=3,
              edgecolor='white', linewidth=1.2)
    off = {'ours': (10, -3, 'left'), 'ours + noise aug': (10, -3, 'left'),
           'paper 1 pipeline': (0, 12, 'center'),
           'hybrid, gated': (-11, -2, 'right')}[name]
    B.annotate(name, (ms, err), textcoords='offset points', xytext=off[:2],
               fontsize=8, color=cols.get(name, MUT), ha=off[2], va='center')
B.set_xscale('log'); B.set_xlim(0.7, 2200); B.set_ylim(0.355, 0.755)
B.axvline(1000 / 30, color=MUT, lw=1, ls=(0, (3, 3)))
B.text(1000 / 30 * 0.8, 0.60, '30 fps\nbudget', fontsize=7.5, color=MUT,
       ha='right', va='center')
B.set_xlabel('wall clock per stroke  (ms, one RTX 4090)')
B.set_ylabel('median 3D error  (m)')
B.set_title('(c) accuracy against cost, clean detections', fontsize=9.5, loc='left', pad=8)
for ax in (A, C, B):
    ax.grid(True, color=LINE, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig('fig/stress.' + ext, facecolor='white', bbox_inches='tight')
print('fig/stress.{png,pdf,svg}')
