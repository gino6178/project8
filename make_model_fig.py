"""TrackNet3D overview -- paper-figure style: modules, shapes, one accent, the one result that matters.
Parameter counts are read from out/repair.pt. Export: SVG + PDF + PNG.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

SD = torch.load('/home/gino/paper3/out/repair.pt', map_location='cpu')
P = {k: sum(v.numel() for v in SD[k].values() if hasattr(v, 'numel')) / 1e6 for k in SD}
INK, MUT, LINE = '#1a1a1a', '#6b6b6b', '#d9d9d9'; ACC, ACC2, GRAD = '#e8552f', '#0e7c7b', '#7a5ea8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(14.0, 5.2), dpi=200); fig.patch.set_facecolor('white')
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')


def blk(x, y, w, h, name, spec='', shape='', col=ACC, fs=8.2, lw=1.4):
    face = {ACC: '#fdeee9', ACC2: '#e6f2f1', GRAD: '#f6f2fb'}[col]
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.003,rounding_size=0.008', lw=lw, ec=col, fc=face, zorder=2))
    AX.text(x + w / 2, y + h / 2 + (0.022 if spec else 0), name, ha='center', va='center', fontsize=fs + 0.8, weight='bold', zorder=3)
    if spec: AX.text(x + w / 2, y + h / 2 - 0.022, spec, ha='center', va='center', fontsize=fs - 0.6, color=MUT, zorder=3, linespacing=1.3)
    if shape: AX.text(x + w / 2, y - 0.03, shape, ha='center', va='top', fontsize=fs - 1.0, color=MUT, zorder=3, family='DejaVu Sans Mono')


def arrow(x0, y0, x1, y1, col=INK, lw=1.4, ls='-', rad=0.0):
    AX.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11, lw=lw, color=col, linestyle=ls,
                                 connectionstyle='arc3,rad=%.2f' % rad, zorder=1))


AX.text(0.03, 0.94, 'TrackNet3D: one graph from pixels to metres', fontsize=14, weight='bold')
AX.text(0.03, 0.885, 'The detector stays five-frame, so its 2D output is compared with TrackNetV5 like for like; the height is read from the whole stroke.', fontsize=8.6, color=MUT)

Y = 0.52; H = 0.24
blk(0.030, Y, 0.110, H, 'five frames', 'RGB + bg diff', '(5, 4, 288, 512)', ACC2)
blk(0.170, Y, 0.130, H, 'TrackNetV5-Lite', 'U-Net, motion gate\n%.2f M' % P['bb'], 'heatmaps (5, 288, 512)', ACC)
blk(0.330, Y, 0.120, H, 'soft-argmax', 'windowed, 0 params', '(u, v) per frame', ACC2)
blk(0.480, Y, 0.140, H, 'stroke transformer', '6 layers, d 192\n%.2f M' % P['repair'], 'height z per frame', ACC)
blk(0.650, Y, 0.130, H, 'closed-form (X, Y)', 'ray ∩ plane at z\n0 params', 'xy_from_uvz(G, col3, uv, z)', ACC2)
blk(0.810, Y, 0.110, H, 'metric 3D', 'court frame', '(n, 3)', ACC2)
for a, b in ((0.140, 0.170), (0.300, 0.330), (0.450, 0.480), (0.620, 0.650), (0.780, 0.810)):
    arrow(a, Y + H / 2, b, Y + H / 2)
AX.text(0.390, Y + H + 0.03, 'scored vs TrackNetV5', ha='center', fontsize=7.2, color=MUT, style='italic')
AX.text(0.550, Y + H + 0.03, 'reads all n frames + camera', ha='center', fontsize=7.2, color=MUT, style='italic')

# auxiliary five-frame head, below the token stream
blk(0.480, 0.17, 0.140, 0.15, 'five-frame head', 'same transformer, 1 window\n%.2f M, training aux' % P['lift'], 'z5 — ignored downstream', ACC)
arrow(0.390, Y, 0.550, 0.32, rad=0.25)
arrow(0.550, 0.32, 0.550, Y, col=MUT, lw=1.0, ls=(0, (2, 2)))

# loss + gradient
AX.add_patch(FancyBboxPatch((0.810, 0.17), 0.110, 0.15, boxstyle='round,pad=0.003,rounding_size=0.008', lw=1.3, ec=GRAD, fc='#f6f2fb', zorder=2))
AX.text(0.865, 0.27, 'loss', ha='center', fontsize=9, weight='bold', zorder=3)
AX.text(0.865, 0.225, 'smooth-L1 on 3D\n+ 0.5 × on z5 + heatmap', ha='center', va='center', fontsize=7.4, color=MUT, zorder=3)
arrow(0.865, Y, 0.865, 0.32, col=GRAD, lw=1.1, ls=(0, (3, 2)))
AX.add_patch(FancyArrowPatch((0.810, 0.245), (0.235, Y), arrowstyle='-|>', mutation_scale=11, lw=1.1, color=GRAD, linestyle=(0, (3, 2)),
                             connectionstyle='arc3,rad=0.22', zorder=1))
AX.text(0.50, 0.075, 'gradient of the 3D loss reaches the convolutions through the soft-argmax', ha='center', fontsize=7.2, color=GRAD, style='italic')

# the one result
AX.add_patch(FancyBboxPatch((0.030, 0.13), 0.400, 0.23, boxstyle='round,pad=0.003,rounding_size=0.008', lw=1.1, ec=LINE, fc='white', zorder=2))
AX.text(0.045, 0.325, 'context the height head reads', fontsize=7.4, color=MUT, weight='bold')
AX.text(0.300, 0.325, '3D median', fontsize=7.4, color=MUT, weight='bold')
rows = [('5 frames', '0.852 m', MUT), ('5 frames + pixels around the shuttle', '0.852 m', MUT),
        ('whole stroke', '0.453 m', ACC2), ('whole stroke, z5 deleted / shuffled', '0.455 / 0.456 m', MUT)]
for i, (a, b, c) in enumerate(rows):
    AX.text(0.045, 0.283 - i * 0.045, a, fontsize=7.6, color=INK if c == ACC2 else MUT, zorder=3)
    AX.text(0.300, 0.283 - i * 0.045, b, fontsize=7.6, color=c, weight='bold' if c == ACC2 else 'normal', zorder=3)
AX.text(0.380, 0.283 - 2 * 0.045, '1.9×', fontsize=8.6, color=ACC2, weight='bold', zorder=3)
AX.text(0.045, 0.095, 'one held-out venue, the model\'s own detections, clean subset', fontsize=6.6, color=MUT, style='italic')

for i, (c, f, t) in enumerate([(ACC, '#fdeee9', 'learned'), (ACC2, '#e6f2f1', 'algebra, 0 parameters'), (GRAD, '#f6f2fb', 'training only')]):
    AX.add_patch(FancyBboxPatch((0.480 + i * 0.130, 0.025), 0.012, 0.012, boxstyle='round,pad=0.001,rounding_size=0.002', lw=1.1, ec=c, fc=f))
    AX.text(0.497 + i * 0.130, 0.031, t, fontsize=6.8, color=MUT, va='center')
AX.text(0.97, 0.031, 'total %.2f M' % sum(P.values()), fontsize=6.8, color=MUT, va='center', ha='right')
for ext in ('png', 'svg', 'pdf'):
    fig.savefig('fig/model.%s' % ext, facecolor='white', bbox_inches='tight', pad_inches=0.05)
print('wrote fig/model.{png,svg,pdf}')
