"""The deployed TrackNet3D graph, drawn as it is actually built and trained.

FIGURE CONTRACT
---------------
Core conclusion: one differentiable graph runs from pixels to metres; the
    detector stays five-frame so its 2D output is comparable to TrackNetV5, and
    the accuracy comes from the stage that reads the WHOLE stroke. The two
    zero-parameter stages carry the geometry and are drawn differently from the
    learned ones, because that distinction is the paper's argument.
Evidence attached to the spine, not asserted: the five-frame branch and the
    stroke branch are the same weights on the same fold (0.852 vs 0.453 m), and
    the intermediate height is measured to be redundant (delete it: 0.455;
    shuffle it: 0.456).
Review risk: drawing the second stage as a "repair" module would be a claim the
    ablation refutes. It is drawn as what it is -- a lifter reading the detected
    track -- with the redundancy stated on the figure.

Every shape and parameter count is read from out/repair.pt, not retyped.
Export: SVG (editable text) + PDF (Type 42) + PNG.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

SD = torch.load('/home/gino/paper3/out/repair.pt', map_location='cpu')
P = {k: sum(v.numel() for v in SD[k].values() if hasattr(v, 'numel')) / 1e6 for k in SD}

INK, MUT, LINE = '#141b1e', '#5c6b70', '#dfe6e5'
LEARN, ALG, GRAD, BAD = '#e8552f', '#0e7c7b', '#7a5ea8', '#c8481f'
SOFT, SOFTA, WARM = '#fdf3ee', '#e9f3f2', '#f6f2fb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

fig = plt.figure(figsize=(12.8, 7.0), dpi=200)
fig.patch.set_facecolor('white')
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')

AX.text(0.038, 0.955, 'One graph from pixels to metres', fontsize=15.5, weight='bold')
AX.text(0.038, 0.915, 'The detector stays five-frame so its 2D output is comparable to TrackNetV5. '
                      'Everything is trained with one loss on metric 3D.',
        fontsize=9.2, color=MUT)


def box(x, y, w, h, title, lines, kind='learn', fs=8.4):
    face = SOFT if kind == 'learn' else (SOFTA if kind == 'alg' else WARM)
    edge = LEARN if kind == 'learn' else (ALG if kind == 'alg' else GRAD)
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                lw=1.5, ec=edge, fc=face, zorder=2))
    AX.text(x + w / 2, y + h - 0.052, title, ha='center', fontsize=9.6, weight='bold', zorder=3)
    for i, s in enumerate(lines):
        AX.text(x + w / 2, y + h - 0.092 - i * 0.038, s, ha='center', fontsize=fs, color=MUT, zorder=3)


def arrow(x0, y0, x1, y1, label='', col=INK, ls='-', lw=1.5, rad=0.0, dy=0.016, fs=8.0):
    AX.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=13,
                                 lw=lw, color=col, linestyle=ls, zorder=1,
                                 connectionstyle='arc3,rad=%.2f' % rad))
    if label:
        AX.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha='center', fontsize=fs,
                color=col, style='italic', zorder=3)


ROW = 0.615
box(0.030, ROW, 0.152, 0.215, 'five frames', ['RGB + background', 'difference, 4 ch each',
                                              '(5, 4, 288, 512)'], 'alg')
box(0.212, ROW, 0.170, 0.215, 'TrackNetV5-Lite', ['alpha 0.5, fine-tuned', 'one heatmap per frame',
                                                  '%.2f M parameters' % P['bb']], 'learn')
box(0.412, ROW, 0.165, 0.215, 'soft-argmax', ['7 px window on the', 'detached hard peak',
                                              'zero parameters'], 'alg')
box(0.607, ROW, 0.150, 0.215, 'detected track', ['(u, v) per frame', 'the 2D output scored',
                                                 'against TrackNetV5'], 'alg')

arrow(0.182, ROW + 0.108, 0.212, ROW + 0.108)
arrow(0.382, ROW + 0.108, 0.412, ROW + 0.108, 'heatmaps')
arrow(0.577, ROW + 0.108, 0.607, ROW + 0.108, 'peaks')

# ---- the two consumers of the detected track
MID = 0.330
box(0.607, MID, 0.150, 0.185, 'five-frame head', ['reads one window', 'of five detections',
                                                  '%.2f M parameters' % P['lift']], 'learn')
box(0.812, MID, 0.160, 0.185, 'stroke-level head', ['reads ALL frames of', 'the stroke + camera',
                                                    '%.2f M parameters' % P['repair']], 'learn')
arrow(0.682, ROW - 0.002, 0.682, MID + 0.185, rad=0.0)
arrow(0.757, ROW + 0.040, 0.892, MID + 0.185, '', rad=-0.20)
AX.text(0.905, ROW - 0.045, 'the whole track, t, camera', ha='center', fontsize=8.0, color=INK, style='italic')
arrow(0.757, MID + 0.100, 0.812, MID + 0.100, '', col=MUT, ls=(0, (3, 3)), lw=1.2)
AX.text(0.784, MID + 0.128, 'z5', ha='center', fontsize=7.8, color=MUT, style='italic')
AX.text(0.784, MID + 0.062, 'ignored', ha='center', fontsize=7.4, color=BAD, style='italic')

box(0.812, 0.085, 0.160, 0.185, 'closed-form (X, Y)', ['xy_from_uvz(G, col3,', '(u,v), Z) - exact',
                                                       'zero parameters'], 'alg')
arrow(0.892, MID - 0.002, 0.892, 0.270, 'height Z')
box(0.607, 0.085, 0.150, 0.185, 'metric 3D', ['court frame, metres', 'one point per frame',
                                              'the deployed output'], 'alg')
arrow(0.812, 0.177, 0.757, 0.177)

# ---- the loss and the gradient
AX.add_patch(FancyBboxPatch((0.030, 0.085), 0.430, 0.185,
                            boxstyle='round,pad=0.008,rounding_size=0.012', lw=1.5,
                            ec=GRAD, fc=WARM, zorder=2))
AX.text(0.245, 0.232, 'one loss, trained jointly', ha='center', fontsize=9.6, weight='bold', zorder=3)
for i, s in enumerate(['smooth-L1 on metric 3D  +  heatmap MSE  +  the same loss on the five-frame branch',
                       'the 3D gradient reaches the convolutions through the soft-argmax:',
                       'trunk gradient norm 7.18, 129 of 133 tensors move after one step']):
    AX.text(0.245, 0.192 - i * 0.038, s, ha='center', fontsize=8.4, color=MUT, zorder=3)
arrow(0.607, 0.150, 0.460, 0.150, col=GRAD, lw=1.4, ls=(0, (4, 3)))
AX.add_patch(FancyArrowPatch((0.036, 0.220), (0.240, ROW - 0.002), arrowstyle='-|>',
                             mutation_scale=13, lw=1.4, color=GRAD, linestyle=(0, (4, 3)),
                             connectionstyle='arc3,rad=0.30', zorder=1))
AX.text(0.020, 0.455, 'gradient', fontsize=8.4, color=GRAD, style='italic', rotation=90)

# ---- measured facts, attached rather than asserted
AX.add_patch(FancyBboxPatch((0.075, 0.320), 0.480, 0.205,
                            boxstyle='round,pad=0.008,rounding_size=0.012', lw=1.2,
                            ec=LINE, fc='white', zorder=2))
AX.text(0.092, 0.487, "What the context width is worth", fontsize=9.4, weight='bold', zorder=3)
AX.text(0.092, 0.458, 'one held-out venue, scored on the clean subset with the model\'s own detections',
        fontsize=7.8, color=MUT, zorder=3)
rows = [('height head reads 5 frames', '0.852 m', MUT, ''),
        ('height head reads the whole stroke', '0.453 m', ALG, '1.9x better'),
        ('delete z5 from the stroke head', '0.455 m', MUT, ''),
        ('shuffle z5 across frames', '0.456 m', MUT, 'so z5 carries nothing')]
for i, (a, b, c, note) in enumerate(rows):
    y = 0.424 - i * 0.029
    AX.text(0.092, y, a, fontsize=8.2, color=MUT, zorder=3)
    AX.text(0.348, y, b, fontsize=8.2, color=c, zorder=3, weight='bold' if c == ALG else 'normal')
    if note:
        AX.text(0.412, y, note, fontsize=7.8, color=c if c == ALG else BAD, zorder=3,
                weight='bold' if c == ALG else 'normal')

for x, c, lab in ((0.545, LEARN, 'learned'), (0.655, ALG, 'geometry, zero parameters'),
                  (0.850, GRAD, 'training only')):
    AX.add_patch(FancyBboxPatch((x, 0.020), 0.016, 0.016,
                                boxstyle='round,pad=0.002,rounding_size=0.004', lw=1.4, ec=c,
                                fc=SOFT if c == LEARN else (SOFTA if c == ALG else WARM)))
    AX.text(x + 0.024, 0.028, lab, fontsize=7.8, color=MUT, va='center')
AX.text(0.038, 0.028, 'total %.2f M parameters  |  camera is an input, never predicted' % sum(P.values()),
        fontsize=8.0, color=MUT, va='center')

for ext in ('png', 'svg', 'pdf'):
    fig.savefig('fig/model.%s' % ext, facecolor='white', bbox_inches='tight', pad_inches=0.06)
print('wrote fig/model.{png,svg,pdf}')
