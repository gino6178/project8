"""Stage strip: what the data actually looks like at each step.

FIGURE CONTRACT
---------------
Core conclusion: the pipeline turns a 7 px smudge into a metric arc, and every
    intermediate is a real thing that can be shown rather than asserted.
Evidence chain: one broadcast frame -> the heatmap the trunk actually produces
    on it -> the detections joined into a stroke -> the height the head reads
    off that stroke -> the 3D arc beside paper 1's, from above, which is the
    only view where the depth direction is not foreshortened.
Archetype: sequential stage strip after MonoTrack's Fig. 4, where each panel
    is the real output of the stage above it, not an icon standing in for one.
Review risk: schematic-only architecture figures let a reader assume every
    stage works; showing the intermediates lets them see the shuttle is barely
    visible and that our arc and paper 1's separate in depth.

One held-out stroke, chosen at the median of the per-stroke error so the
picture is not the best case: A2655, 18 frames, 0.336 m median.
"""
import matplotlib as mpl
mpl.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7})
INK, MUT, LINE = '#141b1e', '#5c6b70', '#c9d6d4'
ACC, ACC2, SOFT = '#0e7c7b', '#c8481f', '#e9f3f2'
W_IN, H_IN = 7.2, 1.82
S = 'fig/src/'
uv = np.load(S + 'uv_track.npy'); P3 = np.load(S + 'p3.npy')
GT = np.load(S + 'gt3.npy'); TT = np.load(S + 't.npy')
n, emed = np.load(S + 'stage.npy')

fig, AX = plt.subplots(figsize=(W_IN, H_IN))
AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')
NP, GAP = 5, 0.022
PWD = (1 - (NP - 1) * GAP) / NP
PHT = PWD * 0.86 * (W_IN / H_IN)          # every cell the same box
PYT = 0.830


def cell(i):
    ax = AX.inset_axes([i * (PWD + GAP), PYT - PHT, PWD, PHT])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE); sp.set_linewidth(0.8)
    return ax


def label(i, t, sub):
    x = i * (PWD + GAP) + PWD / 2
    AX.text(x, PYT + 0.075, t, ha='center', va='bottom', fontsize=6.9,
            color=INK, fontweight='bold')
    AX.text(x, PYT + 0.022, sub, ha='center', va='bottom', fontsize=6.0, color=MUT)


def between(i):
    x = i * (PWD + GAP) + PWD + GAP / 2
    AX.add_patch(FancyArrowPatch((x - GAP * 0.32, PYT - PHT / 2), (x + GAP * 0.32, PYT - PHT / 2),
                 arrowstyle='-|>', mutation_scale=7, lw=1.0, color=ACC, zorder=5))


sc = uv * np.array([512 / 1280.0, 288 / 720.0])

# 1 --------------------------------------------------------------- the frame
ax = cell(0); ax.imshow(plt.imread(S + 's1_frame.png'), aspect='auto')
ax.add_patch(Circle(tuple(sc[len(sc) // 2]), 17, fill=False, ec=ACC2, lw=1.3))
ax.set_xlim(0, 512); ax.set_ylim(288, 0)
label(0, 'broadcast frame', 'shuttle circled, 7 px')
between(0)

# 2 -------------------------------------------------------------- the heatmap
ax = cell(1); ax.imshow(plt.imread(S + 's2_heat.png'), aspect='auto')
ax.set_xlim(0, 512); ax.set_ylim(288, 0)
label(1, 'trunk heatmap', 'the detector answers')
between(1)

# 3 ------------------------------------------------------- the assembled track
ax = cell(2); ax.imshow(plt.imread(S + 's1_frame.png'), alpha=0.40, aspect='auto')
ax.plot(sc[:, 0], sc[:, 1], '-', color=ACC2, lw=1.2, zorder=3)
ax.scatter(sc[:, 0], sc[:, 1], s=6, color=ACC2, zorder=4, edgecolors='white', linewidths=0.35)
ax.set_xlim(0, 512); ax.set_ylim(288, 0)
label(2, 'one stroke', '%d detections joined' % int(n))
between(2)

# 4 -------------------------------------------------------------- the height
ax = cell(3)
ax.plot(TT, GT[:, 2], color=MUT, lw=1.3)
ax.plot(TT, P3[:, 2], color=ACC, lw=1.6)
ax.scatter(TT, P3[:, 2], s=4, color=ACC, zorder=3)
ax.set_ylim(-0.15, max(3.6, GT[:, 2].max() * 1.22))
ax.set_xlim(TT.min() - 0.02, TT.max() + 0.02)
ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['0', '1', '2', '3 m'], fontsize=5.1)
for _tl in ax.get_yticklabels(): _tl.set_horizontalalignment('left')
ax.tick_params(axis='y', pad=-9, labelcolor=MUT)
ax.tick_params(length=1.6, pad=0.6)
ax.text(0.97, 0.94, 'ours', transform=ax.transAxes, ha='right', va='top',
        fontsize=5.8, color=ACC, fontweight='bold')
ax.text(0.97, 0.79, 'paper 1', transform=ax.transAxes, ha='right', va='top',
        fontsize=5.8, color=MUT)
label(3, 'height per frame', 'the only learned number')
between(3)

# 5 ---------------------------------------- the 3D arc from above, court lying down
ax = cell(4)
W, L = 6.10, 13.40
ax.add_patch(Rectangle((0, 0), L, W, fill=False, ec=MUT, lw=0.9))
ax.plot([L / 2, L / 2], [0, W], color=MUT, lw=0.9)
for yy in (0.76, L - 0.76, L / 2 - 1.98, L / 2 + 1.98):
    ax.plot([yy, yy], [0, W], color=LINE, lw=0.6)
for xx in (0.46, W - 0.46, W / 2):
    ax.plot([0, L], [xx, xx], color=LINE, lw=0.6)
ax.plot(GT[:, 1], GT[:, 0], color=MUT, lw=1.4)
ax.plot(P3[:, 1], P3[:, 0], color=ACC, lw=1.6)
ax.scatter(P3[:, 1], P3[:, 0], s=4, color=ACC, zorder=3)
ax.set_xlim(-0.5, L + 0.5); ax.set_ylim(-1.6, W + 1.6)
label(4, 'court, from above', 'median gap %.2f m' % emed)

AX.text(0.5, PYT - PHT - 0.075,
        'One held-out stroke at the median of the per-stroke error, not the best case. '
        'The view from above is the only one in which the depth direction is not foreshortened.',
        ha='center', va='center', fontsize=6.2, color=MUT, style='italic')
for ext in ('svg', 'pdf'):
    fig.savefig('fig/stages.' + ext, bbox_inches='tight')
fig.savefig('fig/stages.png', dpi=400, bbox_inches='tight', facecolor='white')
print('fig/stages.*  (stroke %d frames, %.3f m)' % (int(n), emed))
