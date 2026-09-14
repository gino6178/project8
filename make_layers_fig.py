"""TrackNet3D, layer by layer -- paper-figure style (per the paper-comic skill's spec):
short labels, module blocks with tensor shapes, one accent colour for the learned path, no prose.
Every kernel/channel/shape is read from the instantiated tnv5 TrackNetV5 (alpha 0.5) and out/repair.pt.
Export: SVG (editable text) + PDF (Type 42) + PNG.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK, MUT, LINE = '#1a1a1a', '#6b6b6b', '#d9d9d9'
ACC, ACC2 = '#e8552f', '#0e7c7b'           # learned / algebra
GATE = '#b8860b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

fig = plt.figure(figsize=(16.0, 7.6), dpi=200); fig.patch.set_facecolor('white')
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')


def blk(x, y, w, h, name, spec='', shape='', col=ACC, fill=True, fs=7.6, lw=1.3, z=2):
    face = {ACC: '#fdeee9', ACC2: '#e6f2f1', GATE: '#faf3dc', LINE: 'white'}[col] if fill else 'white'
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.002,rounding_size=0.006', lw=lw, ec=col, fc=face, zorder=z))
    AX.text(x + w / 2, y + h / 2 + (0.014 if spec else 0), name, ha='center', va='center', fontsize=fs + 0.6, weight='bold', zorder=3)
    if spec:
        AX.text(x + w / 2, y + h / 2 - 0.012, spec, ha='center', va='center', fontsize=fs - 0.6, color=MUT, zorder=3)
    if shape:
        AX.text(x + w / 2, y - 0.013, shape, ha='center', va='top', fontsize=fs - 0.8, color=MUT, zorder=3, family='DejaVu Sans Mono')


def arrow(x0, y0, x1, y1, col=INK, lw=1.3, ls='-', rad=0.0, z=1):
    AX.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, lw=lw, color=col,
                                 linestyle=ls, connectionstyle='arc3,rad=%.2f' % rad, zorder=z))


def lab(x, y, s, col=MUT, fs=7.0, ha='center', rot=0, style='italic'):
    AX.text(x, y, s, ha=ha, va='center', fontsize=fs, color=col, style=style, rotation=rot, zorder=3)


AX.text(0.025, 0.965, 'TrackNet3D', fontsize=15, weight='bold')
AX.text(0.025, 0.930, 'five-frame TrackNetV5-Lite  ->  differentiable soft-argmax  ->  stroke-level transformer  ->  closed-form geometry.   Shapes: (C, H, W) at 288x512; (n, d) tokens.',
        fontsize=8.4, color=MUT)

# =============================================================== A. detector, drawn as a U-Net (encoder heights follow resolution)
Y0 = 0.58                         # baseline of the U
AX.text(0.025, 0.885, 'A   detector  (TrackNetV5-Lite, 2.54 M, released weights, fine-tuned)', fontsize=9.2, weight='bold', color=ACC)

blk(0.025, Y0 + 0.04, 0.062, 0.18, 'input', '5 frames x\nRGB + bg diff', '(20, 288, 512)', ACC2)
blk(0.104, Y0 + 0.13, 0.066, 0.09, 'R(2+1)D', 'Conv3d 1x3x3\nConv3d 3x1x1', '(4, 5, H, W) -> (20, H, W)', ACC)
blk(0.104, Y0 + 0.02, 0.066, 0.085, 'motion\nprompt', 'gated |Δ frame|', 'M  (1, H, W)', GATE)
arrow(0.087, Y0 + 0.175, 0.104, Y0 + 0.175); arrow(0.087, Y0 + 0.0625, 0.104, Y0 + 0.0625)

enc = [(0.190, 0.180, 'down 1', '3x3 20->64\n+ FMB', '(64, 288, 512)'),
       (0.262, 0.135, 'down 2', '3x3 64->64\n+ FMB', '(64, 144, 256)'),
       (0.334, 0.095, 'down 3', '3x3 64->64\n+ 2 FMB', '(64, 72, 128)'),
       (0.406, 0.060, 'bottleneck', '3x3 64->128, 2 FMB', '(128, 36, 64)')]
dec = [(0.478, 0.095, 'up 1', 'cat, 3x3 192->64\n+ MB', '(64, 72, 128)'),
       (0.550, 0.135, 'up 2', 'cat, 3x3 128->64\n+ MB', '(64, 144, 256)'),
       (0.622, 0.180, 'up 3', 'cat, 3x3 128->64\n+ MB', '(64, 288, 512)')]
for x, h, n, sp, sh in enc:
    blk(x, Y0 + 0.02, 0.058, h, n, sp, sh, ACC, fs=7.2)
for x, h, n, sp, sh in dec:
    blk(x, Y0 + 0.02, 0.058, h, n, sp, sh, ACC, fs=7.2)
# main path arrows
arrow(0.170, Y0 + 0.175, 0.190, Y0 + 0.16)
xs = [0.190, 0.262, 0.334, 0.406, 0.478, 0.550, 0.622]
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + 0.058, Y0 + 0.035, b, Y0 + 0.035)
lab(0.190, Y0 - 0.022, 'max-pool /2 between encoder blocks;  nearest upsample x2 before each decoder block', fs=6.4, ha='left')
# skip connections over the top
for k, ((xe, he, *_), (xd, hd, *_)) in enumerate(zip(enc[:3], dec[::-1])):
    y = Y0 + 0.02 + 0.180 + 0.022 + k * 0.022
    AX.plot([xe + 0.029, xe + 0.029, xd + 0.029], [Y0 + 0.02 + he, y, y], color=MUT, lw=0.9, ls=(0, (3, 2)), zorder=1)
    arrow(xd + 0.029, y, xd + 0.029, Y0 + 0.02 + hd + 0.002, col=MUT, lw=0.9, ls=(0, (3, 2)))
lab(0.406 + 0.029, Y0 + 0.02 + 0.180 + 0.090, 'skip concatenations', fs=6.6)
# motion prompt gates into the decoder
for x, tag in ((0.478, 'M/4'), (0.550, 'M/2'), (0.622, 'M')):
    blk(x + 0.004, Y0 - 0.055, 0.050, 0.038, 'SCG', tag, '', GATE, fs=6.6, lw=1.1)
    arrow(x + 0.029, Y0 - 0.017, x + 0.029, Y0 + 0.02, col=GATE, lw=1.0)
AX.add_patch(FancyArrowPatch((0.137, Y0 + 0.02), (0.478, Y0 - 0.036), arrowstyle='-|>', mutation_scale=8, lw=1.0, color=GATE,
                             linestyle=(0, (3, 2)), connectionstyle='arc3,rad=0.28', zorder=1))
AX.plot([0.532, 0.536], [Y0 - 0.036, Y0 - 0.036], color=GATE, lw=1.0); AX.plot([0.604, 0.608], [Y0 - 0.036, Y0 - 0.036], color=GATE, lw=1.0)
lab(0.300, Y0 - 0.075, 'M pooled to each decoder resolution', GATE, fs=6.6)

blk(0.694, Y0 + 0.02, 0.058, 0.180, 'predictor', '1x1 64->5, σ', '(5, 288, 512)', ACC, fs=7.2)
arrow(0.680, Y0 + 0.035, 0.694, Y0 + 0.035)
lab(0.723, Y0 + 0.215, 'one heatmap per frame', fs=6.6)

# block internals, one line each
AX.text(0.025, Y0 - 0.115, 'FMB  FusedMBConv:  3x3 C->4C, BN, SiLU  |  1x1 4C->C, BN  |  + x', fontsize=6.9, color=MUT)
AX.text(0.025, Y0 - 0.140, 'MB   MBConv:  1x1 C->4C, BN, SiLU  |  dw 3x3, BN, SiLU  |  1x1 4C->C, BN  |  + x', fontsize=6.9, color=MUT)
AX.text(0.025, Y0 - 0.165, 'SCG  spatial-channel gate:  V · σ(MLP[GAP V, GAP M]) · σ(3x3 M)  ->  dw 3x3, pw 1x1  |  + V', fontsize=6.9, color=MUT)

# =============================================================== B. soft-argmax and tokens
AX.text(0.780, 0.885, 'B   peak -> pixel  (0 parameters)', fontsize=9.2, weight='bold', color=ACC2)
blk(0.780, Y0 + 0.13, 0.095, 0.085, 'windowed\nsoft-argmax', '7 px window at the\ndetached argmax, T=40', '', ACC2, fs=7.2)
AX.text(0.883, Y0 + 0.16, '(5, 2) px\n@ 1280x720', fontsize=6.6, color=MUT, va='center', family='DejaVu Sans Mono')
arrow(0.752, Y0 + 0.17, 0.780, Y0 + 0.17)
blk(0.780, Y0 + 0.02, 0.095, 0.075, 'token', 'u, v, t, Δuv/Δt,\ncamera (9)', '', ACC2, fs=7.2)
AX.text(0.883, Y0 + 0.057, '(n, 14)', fontsize=6.6, color=MUT, va='center', family='DejaVu Sans Mono')
arrow(0.8275, Y0 + 0.13, 0.8275, Y0 + 0.095)
lab(0.883, Y0 + 0.205, 'the 2D output scored\nagainst TrackNetV5', fs=6.6, ha='left')

# =============================================================== C. the two heads
YB = 0.10
AX.text(0.025, 0.36, 'C   lifting heads  (2 x 2.80 M)', fontsize=9.2, weight='bold', color=ACC)
# one expanded encoder layer
X1 = 0.025
blk(X1, YB + 0.02, 0.070, 0.055, 'Linear', '14 -> 192  + pos', '(n, 192)', ACC, fs=7.0)
lx = X1 + 0.088
AX.add_patch(Rectangle((lx, YB), 0.215, 0.215, lw=1.1, ec=ACC, fc='white', ls=(0, (4, 2)), zorder=1))
AX.text(lx + 0.006, YB + 0.20, 'x 6   TransformerEncoderLayer, pre-norm', fontsize=7.0, weight='bold', va='center')
blk(lx + 0.010, YB + 0.115, 0.058, 0.055, 'LN + MHA', '6 heads, d 192', '', ACC, fs=6.8)
blk(lx + 0.078, YB + 0.115, 0.058, 0.055, 'LN + FFN', '192 -> 768 -> 192', '', ACC, fs=6.8)
blk(lx + 0.146, YB + 0.115, 0.058, 0.055, 'dropout', '0.1', '', ACC, fs=6.8, fill=False)
arrow(lx + 0.068, YB + 0.1425, lx + 0.078, YB + 0.1425); arrow(lx + 0.136, YB + 0.1425, lx + 0.146, YB + 0.1425)
for x0 in (lx + 0.010, lx + 0.078):
    AX.add_patch(FancyArrowPatch((x0, YB + 0.115), (x0 + 0.058, YB + 0.115), arrowstyle='-|>', mutation_scale=7, lw=0.9, color=MUT,
                                 connectionstyle='arc3,rad=0.6', zorder=1))
lab(lx + 0.107, YB + 0.055, 'residual', fs=6.4)
arrow(X1 + 0.070, YB + 0.0475, lx, YB + 0.0475)
blk(lx + 0.230, YB + 0.02, 0.078, 0.055, 'readout', 'LN, 192->128, GELU, 128->1', 'z = 1.5 + out', ACC, fs=6.8)
arrow(lx + 0.215, YB + 0.0475, lx + 0.230, YB + 0.0475)

# the two instantiations
X2 = 0.445
blk(X2, YB + 0.105, 0.120, 0.075, 'five-frame head', 'n = 5, one window\nt from the window start', 'z5  (5,)', ACC, fs=7.0)
blk(X2, YB + 0.005, 0.120, 0.075, 'stroke-level head', 'n = whole stroke\n+ Linear 2->192 on (z5, Δz5)', 'z  (n,)', ACC, fs=7.0)
lab(X2 - 0.008, YB + 0.150, 'same layers,\nseparate weights', fs=6.6, ha='right')
BUS = YB + 0.245
AX.plot([0.8275, 0.8275, X2 + 0.060], [Y0 + 0.02, BUS, BUS], color=INK, lw=1.3, zorder=1)
arrow(X2 + 0.060, BUS, X2 + 0.060, YB + 0.182, col=INK, lw=1.3)
AX.plot([X2 + 0.100, X2 + 0.100], [BUS, YB + 0.082], color=INK, lw=1.3, zorder=1); AX.plot([X2 + 0.060, X2 + 0.100], [BUS, BUS], color=INK, lw=1.3, zorder=1)
arrow(X2 + 0.100, YB + 0.095, X2 + 0.100, YB + 0.082, col=INK, lw=1.3)
lab(0.8275 + 0.006, (Y0 + 0.02 + BUS) / 2, 'tokens', fs=6.6, ha='left')
arrow(X2 + 0.030, YB + 0.105, X2 + 0.030, YB + 0.082, col=MUT, lw=0.9, ls=(0, (2, 2)))
lab(X2 - 0.008, YB + 0.093, 'z5: zero-init input,\nignored (ablation)', ACC, fs=6.4, ha='right')

# =============================================================== D. geometry
X3 = 0.640
AX.text(X3, 0.36, 'D   geometry  (0 parameters)', fontsize=9.2, weight='bold', color=ACC2)
blk(X3, YB + 0.005, 0.120, 0.075, 'closed-form (X, Y)', 'ray through (u,v) meets\nthe plane at height z', 'xy_from_uvz(G, col3, uv, z)', ACC2, fs=7.0)
arrow(X2 + 0.120, YB + 0.0425, X3, YB + 0.0425)
blk(X3 + 0.140, YB + 0.005, 0.090, 0.075, 'metric 3D', 'court frame, per frame', '(n, 3)', ACC2, fs=7.0)
arrow(X3 + 0.120, YB + 0.0425, X3 + 0.140, YB + 0.0425)
lab(X3 + 0.060, YB + 0.20, 'camera G, col3 are inputs\n(calibrated once per venue)', fs=6.6)

# =============================================================== loss
AX.add_patch(FancyBboxPatch((X3 + 0.250, YB + 0.005), 0.100, 0.075, boxstyle='round,pad=0.002,rounding_size=0.006', lw=1.1, ec='#7a5ea8', fc='#f6f2fb', zorder=2))
AX.text(X3 + 0.300, YB + 0.055, 'loss', ha='center', fontsize=8.0, weight='bold', zorder=3)
AX.text(X3 + 0.300, YB + 0.030, 'smooth-L1 on 3D\n+ 0.5 x on z5 branch\n+ heatmap MSE', ha='center', va='center', fontsize=6.6, color=MUT, zorder=3)
arrow(X3 + 0.230, YB + 0.0425, X3 + 0.250, YB + 0.0425, col='#7a5ea8', lw=1.0, ls=(0, (3, 2)))
lab(X3 + 0.300, YB + 0.20, 'gradient reaches the\ndetector through B', '#7a5ea8', fs=6.6)

# legend
for i, (c, f, t) in enumerate([(ACC, '#fdeee9', 'learned'), (ACC2, '#e6f2f1', 'algebra'), (GATE, '#faf3dc', 'motion gate'), ('#7a5ea8', '#f6f2fb', 'training only')]):
    AX.add_patch(FancyBboxPatch((0.025 + i * 0.075, 0.025), 0.011, 0.011, boxstyle='round,pad=0.001,rounding_size=0.002', lw=1.1, ec=c, fc=f))
    AX.text(0.040 + i * 0.075, 0.0305, t, fontsize=6.8, color=MUT, va='center')
AX.text(0.985, 0.0305, 'total 8.13 M parameters', fontsize=6.8, color=MUT, va='center', ha='right')

for ext in ('png', 'svg', 'pdf'):
    fig.savefig('fig/layers.%s' % ext, facecolor='white', bbox_inches='tight', pad_inches=0.05)
print('wrote fig/layers.{png,svg,pdf}')
