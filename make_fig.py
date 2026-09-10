"""Architecture schematic for the TrackNet3D page.

FIGURE CONTRACT
---------------
Core conclusion: the height head is what makes this work, and it is what does
    NOT look at the image. Everything the picture shows is arranged to answer
    "then why is there a trunk at all?" -- the trunk finds the shuttle, the
    height head reads the shape of the whole flight, and the homography does
    the rest without being asked.
Evidence chain: five frames cannot fix a height (the arc is straight over
    0.167 s) -> a whole stroke can (27 frames at the median) -> horizontal
    then follows in closed form, so it is not predicted -> a solver can refine
    what the model produced, but only where the solver itself says so.
Archetype: schematic-led composite, pipeline on the right, the three claims
    that justify its shape on the left.
Review risk: a reader assumes an "end-to-end from pixels" system feeds pixels
    to every stage. The figure has to say, in the figure, that feeding the
    trunk's features to the height head was measured and made it worse.

Supersedes the earlier launch-state schematic. That figure drew a six-number
launch state pushed through an RK4 drag integrator; the design was implemented
and measured (0.294 m against 0.204 m for free per-frame heights) and is not
what this system does.

Export: SVG (editable text) + PDF (Type 42) + PNG, 183 mm double-column width.
"""
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
    'axes.linewidth': 0.8, 'legend.frameon': False,
})
INK, MUT, LINE, ACC, ACC2, SOFT = '#141b1e', '#5c6b70', '#c9d6d4', '#0e7c7b', '#c8481f', '#e9f3f2'
GOLD, BLUE, WARM = '#a8791b', '#2f6ea8', '#fdf3ee'
W_IN, H_IN = 7.2, 6.35
TITLE_PT, BODY_PT, HEAD_PT = 7.6, 6.4, 8.6


class Sheet:
    def __init__(s, w, h, title):
        s.fig, s.ax = plt.subplots(figsize=(w, h))
        s.h = h
        s.ax.set_xlim(0, 1); s.ax.set_ylim(0, 1); s.ax.axis('off')
        s.ls = BODY_PT * 1.62 / (72 * h)
        s.t0 = TITLE_PT * 1.35 / (72 * h)
        s.pad = BODY_PT * 1.15 / (72 * h)
        s.ax.text(0.0, 1.0, title, ha='left', va='top', fontsize=HEAD_PT,
                  color=INK, fontweight='bold')

    def box(s, x, ytop, w, title, lines, fc='white', ec=LINE, lw=1.0):
        h = 2 * s.pad + s.t0 + s.ls * len(lines)
        s.ax.add_patch(FancyBboxPatch((x, ytop - h), w, h,
                       boxstyle='round,pad=0.006,rounding_size=0.014',
                       fc=fc, ec=ec, lw=lw, zorder=2))
        s.ax.text(x + w / 2, ytop - s.pad - s.t0 / 2, title, ha='center', va='center',
                  fontsize=TITLE_PT, color=INK, fontweight='bold', zorder=3)
        for i, t in enumerate(lines):
            s.ax.text(x + w / 2, ytop - s.pad - s.t0 - s.ls * (i + .5), t, ha='center',
                      va='center', fontsize=BODY_PT, color=MUT, zorder=3)
        return ytop - h

    def arrow(s, p, q, color=ACC, rad=0.0, ls='-', lw=1.2):
        s.ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=lw,
                       color=color, shrinkA=1.5, shrinkB=1.5, zorder=1,
                       linestyle=ls, connectionstyle='arc3,rad=%.2f' % rad))

    def note(s, x, y, t, color=MUT, fs=None, style='italic', ha='center', weight='normal'):
        s.ax.text(x, y, t, ha=ha, va='center', fontsize=fs or BODY_PT - .4, color=color,
                  style=style, zorder=4, fontweight=weight)

    def panel(s, x, ytop, w, h):
        ax = s.ax.inset_axes([x, ytop - h, w, h])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE); sp.set_linewidth(0.8)
        return ax

    def save(s, stem):
        s.fig.savefig(stem + '.svg', bbox_inches='tight')
        s.fig.savefig(stem + '.pdf', bbox_inches='tight')
        s.fig.savefig(stem + '.png', dpi=400, bbox_inches='tight', facecolor='white')
        plt.close(s.fig)


S = Sheet(W_IN, H_IN, 'How a height is recovered, and what never sees a pixel')
LX, LW = 0.0, 0.300          # left evidence column
RX, RW = 0.400, 0.600        # right pipeline column
top = 0.925

# ---------------------------------------------------------------- pipeline
y = S.box(RX, top, RW, 'Broadcast frames',
          ['5 consecutive frames, 512 x 288, RGB + background difference'], fc=SOFT, ec=ACC)
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
y = S.box(RX, y - 0.022, RW, 'TrackNetV5-Lite trunk, fine-tuned',
          ['heatmap head kept and supervised  ->  soft-argmax (u, v)',
           'F1 95.58 against the released 96.77: the cost is recall, not precision'])
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
y = S.box(RX, y - 0.022, RW, 'One stroke of detections',
          ['the whole flight, 27 frames at the median, plus the venue camera',
           '(four court corners and a focal length, calibrated once)'])
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
yh = y - 0.022
y = S.box(RX, yh, RW, 'Height head  -  one number per frame',
          ['6-layer transformer, 192-d, over the stroke',
           'reads the track and the camera.  It does not read the image.'],
          fc=WARM, ec=ACC2, lw=1.3)
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
y = S.box(RX, y - 0.022, RW, 'Horizontal position  -  not predicted',
          ['(X, Y) = xy_from_uvz(G, col3, (u, v), Z),  closed form',
           'exact to 0.031 m on the labels; regressing it costs 0.152 vs 0.113 m'])
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
y = S.box(RX, y - 0.022, RW, 'Optional refinement, gated',
          ['paper 1 ’s constrained solver, started from OUR estimate',
           'accepted only where its own fit residual is  <=  20 px  (80% of strokes)'],
          fc=SOFT, ec=ACC)
S.arrow((RX + RW / 2, y), (RX + RW / 2, y - 0.022))
y = S.box(RX, y - 0.022, RW, '3D trajectory in court metres',
          ['0.351 m median against paper 1, 6.9% of frames within 5 cm',
           'no camera is estimated by the trajectory head at inference'],
          fc=SOFT, ec=ACC)

# ------------------------------------------------- left column: three claims
def arc(t, z0, vz, drag=0.9):
    """A shuttle-like arc: gravity plus the strong drag that flattens the tail."""
    return z0 + vz * t - 0.5 * 9.81 * t ** 2 - drag * t ** 3


ph = 0.185
py = 0.925

# (a) five frames cannot fix a height
S.note(LX, py + 0.012, 'a.  Why five frames are not enough', color=INK, style='normal',
       ha='left', weight='bold', fs=BODY_PT)
ax = S.panel(LX, py, LW, ph)
t = np.linspace(0, 0.95, 300)
TC, HALF = 0.3835, 0.0835                          # five frames at 30 fps, centred
base = arc(t, 2.4, 5.2)
# Perturbations that vanish to third order at the window centre: inside the
# window the three flights are the same curve to within a pixel, outside they
# are metres apart. That is the ambiguity, drawn.
for k in (14.0, -14.0):
    ax.plot(t, base + k * (t - TC) ** 3, color=LINE, lw=1.0)
ax.plot(t, base, color=ACC, lw=1.3)
ax.axvspan(TC - HALF, TC + HALF, color=ACC2, alpha=0.13, lw=0)
tw = np.linspace(TC - HALF, TC + HALF, 30)
ax.plot(tw, arc(tw, 2.4, 5.2), color=ACC2, lw=2.6)
ax.text(TC, 0.42, '5 frames\n0.167 s', ha='center', va='bottom', fontsize=BODY_PT - 1.1,
        color=ACC2, linespacing=1.25)
ax.annotate('', xy=(0.90, arc(np.array([0.90]), 2.4, 5.2)[0] + 14 * (0.90 - TC) ** 3),
            xytext=(0.90, arc(np.array([0.90]), 2.4, 5.2)[0] - 14 * (0.90 - TC) ** 3),
            arrowprops=dict(arrowstyle='<->', color=MUT, lw=0.8, shrinkA=0, shrinkB=0))
ax.text(0.845, 4.35, 'metres\napart', ha='right', va='center',
        fontsize=BODY_PT - 1.1, color=MUT, linespacing=1.25)
ax.text(0.045, 5.55, 'identical inside the window', ha='left', va='top',
        fontsize=BODY_PT - 1.1, color=INK)
ax.set_xlim(0, 0.97); ax.set_ylim(0, 6.0)
S.note(LX, py - ph - 0.026,
       'Over 0.167 s the arc is straight, so its height', ha='left', fs=BODY_PT - 0.8)
S.note(LX, py - ph - 0.026 - S.ls,
       'is unobservable. A stroke is long enough to bend.', ha='left', fs=BODY_PT - 0.8)

# (b) pixels do not help the height head
py2 = py - ph - 0.105
S.note(LX, py2 + 0.012, 'b.  Adding pixels makes it worse', color=INK, style='normal',
       ha='left', weight='bold', fs=BODY_PT)
ax = S.panel(LX, py2, LW, ph * 0.78)
bars = [('track only', 0.321, ACC), ('+ trunk features', 0.510, ACC2)]
for i, (lab, v, c) in enumerate(bars):
    ax.barh(i, v, height=0.42, color=c, alpha=0.85)
    ax.text(v + 0.018, i, '%.3f m' % v, va='center', fontsize=BODY_PT - 1.0, color=INK)
    ax.text(0.012, i - 0.30, lab, va='bottom', fontsize=BODY_PT - 1.0, color=MUT)
ax.set_xlim(0, 0.72); ax.set_ylim(-0.85, 1.72); ax.invert_yaxis()
S.note(LX, py2 - ph * 0.78 - 0.026,
       'The 64-d trunk feature at the detected point', ha='left', fs=BODY_PT - 0.8)
S.note(LX, py2 - ph * 0.78 - 0.026 - S.ls,
       'drowns the 14-d track it was meant to help.', ha='left', fs=BODY_PT - 0.8)

# (c) the model cannot gate itself, the solver can
py3 = py2 - ph * 0.78 - 0.105
S.note(LX, py3 + 0.012, 'c.  Who knows when the answer is wrong', color=INK, style='normal',
       ha='left', weight='bold', fs=BODY_PT)
ax = S.panel(LX, py3, LW, ph * 0.80)
sig = [('detector conf.', 0.018), ('trajectory jerk', 0.099), ('closed-form scale', 0.025),
       ('solver residual', 0.62)]
for i, (lab, r) in enumerate(sig):
    c = ACC if r > 0.3 else LINE
    ax.barh(i, r, height=0.44, color=c, alpha=0.9 if r > 0.3 else 1.0)
    ax.text(0.012, i - 0.30, lab, va='bottom', fontsize=BODY_PT - 1.1, color=MUT)
ax.axvline(0.0, color=MUT, lw=0.6)
ax.set_xlim(0, 0.88); ax.set_ylim(-0.85, 3.75); ax.invert_yaxis()
ax.text(0.70, 3.45, 'usable', ha='center', va='center', fontsize=BODY_PT - 1.1, color=ACC)
S.note(LX, py3 - ph * 0.80 - 0.026,
       'Nothing the model reports predicts its own', ha='left', fs=BODY_PT - 0.8)
S.note(LX, py3 - ph * 0.80 - 0.026 - S.ls,
       'error (rank corr. <= 0.10). The solver measures', ha='left', fs=BODY_PT - 0.8)
S.note(LX, py3 - ph * 0.80 - 0.026 - 2 * S.ls,
       'its fit after the fact; the gate reads that.', ha='left', fs=BODY_PT - 0.8)

# ---------------------------------------------------------------- footer
S.note(0.5, 0.030,
       'Height error reaches 3D error through the closed form with a measured gain of 4.1x:',
       color=INK, fs=BODY_PT - 0.2)
S.note(0.5, 0.030 - S.ls, 'one centimetre of height is four centimetres of position.',
       color=INK, fs=BODY_PT - 0.2)
S.save('fig/arch')
print('fig/arch.svg / .pdf / .png')
