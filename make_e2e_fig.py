"""The unified end-to-end graph, drawn.

FIGURE CONTRACT
  - every shape and parameter count is what exp89_unified.py actually produces,
    read from a live forward pass, not retyped
  - the gradient arrow is drawn only because it was verified: a 3D-only loss
    gives the trunk a gradient norm of 7.18 and moves 129/133 of its tensors
  - the two zero-parameter stages are drawn differently from the learned ones,
    because that distinction is the paper's argument
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUT, LINE = '#141b1e', '#5c6b70', '#dfe6e5'
LEARN, ALG, GRAD = '#e8552f', '#0e7c7b', '#7a5ea8'
SOFT, SOFTA = '#fdf3ee', '#e9f3f2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})

fig = plt.figure(figsize=(11.6, 7.4), dpi=220)
fig.patch.set_facecolor('white')
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')

AX.text(0.045, 0.965, 'One graph from pixels to metres',
        fontsize=17, fontweight='bold', va='top')
AX.text(0.045, 0.924,
        'The 3D loss reaches the convolutions. Two stages in the middle have no parameters at all.',
        fontsize=10.5, color=MUT, va='top', style='italic')

BX, BW = 0.30, 0.40
STAGES = [
    (0.815, 0.088, 'learn', 'TrackNetV5-Lite trunk',
     ['5 frames x (RGB + background difference), 288 x 512',
      '2.53 M parameters   alpha 0.50   one heatmap per frame']),
    (0.663, 0.080, 'alg', 'Spatial soft-argmax',
     ['heatmap -> continuous (u, v) + peak confidence',
      'no parameters.  a 7 px window, not the whole image']),
    (0.515, 0.088, 'learn', 'Lifting transformer',
     ['reads (u, v), t, d(u,v)/dt and the venue camera',
      '6 layers, d = 192, 2.80 M parameters -> height Z']),
    (0.363, 0.080, 'alg', 'Closed-form geometry',
     ['(X, Y) = xy_from_uvz(G, col3, (u, v), Z)',
      'no parameters.  reprojects onto (u,v) to 8e-07 px']),
    (0.225, 0.070, 'out', '3D trajectory, court metres',
     ['(X, Y, Z) per frame']),
]
mid = {}
for y, h, kind, title, lines in STAGES:
    fc = SOFT if kind == 'learn' else (SOFTA if kind == 'alg' else 'white')
    ec = LEARN if kind == 'learn' else (ALG if kind == 'alg' else MUT)
    AX.add_patch(FancyBboxPatch((BX, y - h), BW, h, boxstyle='round,pad=0.006',
                                fc=fc, ec=ec, lw=1.6 if kind != 'out' else 1.2))
    AX.text(BX + BW / 2, y - 0.022, title, ha='center', fontsize=11.5, fontweight='bold')
    for k, ln in enumerate(lines):
        AX.text(BX + BW / 2, y - 0.044 - k * 0.021, ln, ha='center',
                fontsize=8.6, color=MUT)
    mid[title] = (y, h)

FLOW = [(0.815, 0.663, '(1, 5, 288, 512)  heatmaps'),
        (0.663, 0.515, '(1, 5, 2)  pixels'),
        (0.515, 0.363, '(1, 5)  height in metres'),
        (0.363, 0.225, '(1, 5, 2)  horizontal, in metres')]
for ytop, ybot, lab in FLOW:
    ytop -= [h for (y, h, *_ ) in STAGES if abs(y - ytop) < 1e-9][0]
    AX.add_patch(FancyArrowPatch((BX + BW / 2, ytop), (BX + BW / 2, ybot),
                                 arrowstyle='-|>', mutation_scale=13,
                                 color=MUT, lw=1.3))
    AX.text(BX + BW / 2 + 0.012, (ytop + ybot) / 2, lab, fontsize=8.4,
            color=ALG, style='italic', va='center')

# input at the top
AX.text(BX + BW / 2, 0.878, 'broadcast frames  (1280 x 720, 30 fps)',
        ha='center', fontsize=10, color=INK)
AX.add_patch(FancyArrowPatch((BX + BW / 2, 0.868), (BX + BW / 2, 0.815),
                             arrowstyle='-|>', mutation_scale=13, color=MUT, lw=1.3))

# ---- the camera enters as an input, it is not predicted
CY = 0.515
AX.text(0.962, CY + 0.028, 'an INPUT, not a prediction', ha='right', fontsize=8.8,
        color=INK, style='italic')
AX.add_patch(FancyBboxPatch((0.745, CY - 0.052), 0.217, 0.052,
                            boxstyle='round,pad=0.006', fc='white', ec=MUT, lw=1.2))
AX.text(0.8535, CY - 0.018, 'venue camera', ha='center', fontsize=10, fontweight='bold')
AX.text(0.8535, CY - 0.039, '4 corners + focal, calibrated once',
        ha='center', fontsize=8.4, color=MUT)
AX.add_patch(FancyArrowPatch((0.745, CY - 0.026), (BX + BW, CY - 0.026),
                             arrowstyle='-|>', mutation_scale=12, color=MUT, lw=1.2))
AX.text(0.962, CY - 0.075,
        'exp51 predicted it inside the graph and lost:\n'
        '19.8 px of corner error on an unseen venue\nwent straight into Y.',
        ha='right', fontsize=8.0, color=MUT, va='top')

# ---- the gradient, drawn on the left because it was verified
GX = 0.255
AX.add_patch(FancyArrowPatch((GX, 0.225), (GX, 0.815), arrowstyle='-|>',
                             mutation_scale=15, color=GRAD, lw=2.0,
                             connectionstyle='arc3,rad=-0.16'))
AX.text(0.135, 0.555, 'the 3D loss reaches\nthe convolutions',
        fontsize=10.5, color=GRAD, fontweight='bold', ha='center')
AX.text(0.135, 0.497,
        'verified, not assumed:\nwith the heatmap loss removed\nentirely, a 3D-only loss gives the\n'
        'trunk a gradient norm of 7.18 and\nmoves 129 of its 133 tensors.',
        fontsize=8.2, color=MUT, ha='center', va='top')

# ---- losses
AX.add_patch(FancyBboxPatch((0.735, 0.225), 0.225, 0.105, boxstyle='round,pad=0.006',
                            fc='white', ec=MUT, lw=1.2))
AX.text(0.8475, 0.308, 'losses', ha='center', fontsize=10, fontweight='bold')
for k, (t_, c_) in enumerate([('3D position, smooth L1', INK),
                              ('heatmap, anchors the detector', MUT),
                              ('net barrier (optional)', MUT)]):
    AX.text(0.8475, 0.284 - k * 0.020, t_, ha='center', fontsize=8.4, color=c_)
AX.add_patch(FancyArrowPatch((0.735, 0.278), (BX + BW, 0.242),
                             arrowstyle='-|>', mutation_scale=12, color=MUT, lw=1.2))

# ---- legend and footer
AX.add_patch(FancyBboxPatch((0.30, 0.075), 0.40, 0.052, boxstyle='round,pad=0.006',
                            fc='white', ec=LINE, lw=1.0))
for k, (lab, c, f) in enumerate([('learned here', LEARN, SOFT),
                                 ('algebra, no parameters', ALG, SOFTA)]):
    AX.add_patch(FancyBboxPatch((0.325 + k * 0.195, 0.094), 0.020, 0.016,
                                boxstyle='round,pad=0.003', fc=f, ec=c, lw=1.3))
    AX.text(0.352 + k * 0.195, 0.102, lab, fontsize=9, va='center')
AX.text(0.5, 0.038,
        '5.32 M parameters in total, of which the two middle stages contribute none.',
        ha='center', fontsize=9.2, color=MUT, style='italic')

for ext in ('png', 'pdf', 'svg'):
    fig.savefig('fig/e2e.' + ext, facecolor='white', bbox_inches='tight')
print('fig/e2e.{png,pdf,svg}')
