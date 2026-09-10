"""Architecture schematic for the TrackNet3D page.

FIGURE CONTRACT
---------------
Core conclusion: the stage that recovers the height is the stage that does NOT
    look at the image, and each of the four design choices that sounds wrong
    was measured. The figure has to make the pipeline readable on its own and
    then attach, to each stage, the measurement that forced its shape.
Evidence chain: a 7 px shuttle needs the difference channel to be found at all
    -> five frames cannot fix a height, a whole stroke can -> pixels fed to the
    height head make it worse -> the model cannot gate itself but the solver
    can.
Archetype: schematic spine with attached evidence. Each panel on the left is
    tied by a leader line to the one stage it justifies; arrows between stages
    are labelled with what actually flows, so the reader never has to guess
    whether a stage passes pixels, coordinates or metres.
Review risk: two columns that do not touch read as two figures. The leader
    lines and the flow labels are what stop that, and the legend has to say
    which stages are learned and which are algebra.

Supersedes the earlier launch-state schematic, whose six-number launch state
and RK4 drag integrator were implemented, measured (0.294 m against 0.204 m
for free per-frame heights) and are not what this system does.

Export: SVG (editable text) + PDF (Type 42) + PNG, 183 mm double-column width.
"""
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
    'axes.linewidth': 0.8, 'legend.frameon': False,
})
INK, MUT, LINE = '#141b1e', '#5c6b70', '#c9d6d4'
ACC, ACC2, SOFT, WARM = '#0e7c7b', '#c8481f', '#e9f3f2', '#fdf3ee'
W_IN, H_IN = 7.2, 7.1
TITLE_PT, BODY_PT, HEAD_PT, FLOW_PT = 7.8, 6.5, 9.0, 6.0

fig, AX = plt.subplots(figsize=(W_IN, H_IN))
AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis('off')
LS = BODY_PT * 1.60 / (72 * H_IN)
PAD = BODY_PT * 1.10 / (72 * H_IN)
T0 = TITLE_PT * 1.35 / (72 * H_IN)

AX.text(0.0, 0.995, 'How a height is recovered, and what never sees a pixel',
        ha='left', va='top', fontsize=HEAD_PT, color=INK, fontweight='bold')
AX.text(0.0, 0.963, 'Each panel on the left is the measurement that forced the stage it points to.',
        ha='left', va='top', fontsize=BODY_PT, color=MUT, style='italic')

# geometry -------------------------------------------------------------------
PX, PW = 0.000, 0.315                 # evidence column
BX, BW = 0.430, 0.570                 # pipeline column
LEAD = 0.020                          # leader-line inset


def box(ytop, title, lines, fc='white', ec=LINE, lw=1.0):
    h = 2 * PAD + T0 + LS * len(lines)
    AX.add_patch(FancyBboxPatch((BX, ytop - h), BW, h,
                 boxstyle='round,pad=0.006,rounding_size=0.013',
                 fc=fc, ec=ec, lw=lw, zorder=2))
    AX.text(BX + BW / 2, ytop - PAD - T0 / 2, title, ha='center', va='center',
            fontsize=TITLE_PT, color=INK, fontweight='bold', zorder=3)
    for i, t in enumerate(lines):
        AX.text(BX + BW / 2, ytop - PAD - T0 - LS * (i + .5), t, ha='center',
                va='center', fontsize=BODY_PT, color=MUT, zorder=3)
    return ytop - h, (ytop + ytop - h) / 2


def flow(y0, y1, label):
    """An arrow between stages, labelled with what travels along it."""
    x = BX + BW / 2
    AX.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>',
                 mutation_scale=8, lw=1.1, color=ACC, shrinkA=1.5, shrinkB=1.5, zorder=1))
    AX.text(x + 0.012, (y0 + y1) / 2, label, ha='left', va='center',
            fontsize=FLOW_PT, color=ACC, style='italic', zorder=4)


def lead(py, by):
    """Dotted leader from an evidence panel to the stage it explains."""
    AX.plot([PX + PW + 0.004, BX - LEAD, BX - 0.004], [py, by, by],
            color=LINE, lw=0.8, ls=(0, (1.6, 1.6)), zorder=0, solid_capstyle='butt')
    AX.add_patch(Circle((BX - 0.004, by), 0.0035, fc=LINE, ec='none', zorder=1))


def panel(ytop, h, w=PW, x=PX):
    ax = AX.inset_axes([x, ytop - h, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE); sp.set_linewidth(0.8)
    return ax


def head(y, t):
    AX.text(PX, y, t, ha='left', va='bottom', fontsize=BODY_PT + 0.3,
            color=INK, fontweight='bold')


def cap(y, lines):
    for i, t in enumerate(lines):
        AX.text(PX, y - i * LS, t, ha='left', va='top', fontsize=BODY_PT - 0.7,
                color=MUT, style='italic')
    return y - len(lines) * LS

# ======================================================= pipeline, with flows
TOP = 0.918
y, c_frames = box(TOP, 'Broadcast frames',
                  ['5 consecutive frames, 512 x 288', 'RGB + background difference'],
                  fc=SOFT, ec=ACC)
flow(y, y - 0.034, '4 channels x 5 frames')
y, c_trunk = box(y - 0.034, 'TrackNetV5-Lite trunk  (learned)',
                 ['heatmap head kept and supervised, soft-argmax to (u, v)',
                  'F1 95.58 against the released 96.77 - the cost is recall'])
flow(y, y - 0.034, 'one (u, v) per frame')
y, c_stroke = box(y - 0.034, 'One stroke, assembled',
                  ['the whole flight: 27 frames at the median',
                   'plus the venue camera, calibrated once'])
flow(y, y - 0.034, '27 x (u, v)  +  camera')
y, c_head = box(y - 0.034, 'Height head  (learned)',
                ['6-layer transformer, 192-d, over the whole stroke',
                 'reads the track and the camera.  Never the image.'],
                fc=WARM, ec=ACC2, lw=1.4)
flow(y, y - 0.034, 'one height Z per frame')
y, c_solve = box(y - 0.034, 'Horizontal position  (algebra, not learned)',
                 ['(X, Y) = xy_from_uvz(G, col3, (u, v), Z)',
                  'exact to 0.031 m; regressing it instead costs 0.152 vs 0.113 m'])
flow(y, y - 0.034, 'a 3D track, in metres')
y, c_gate = box(y - 0.034, 'Refinement, gated  (optional)',
                ["paper 1 's solver, started from OUR estimate, kept only",
                 'where its own fit residual is <= 20 px  (80% of strokes)'],
                fc=SOFT, ec=ACC)
flow(y, y - 0.034, 'accepted or discarded')
y, c_out = box(y - 0.034, '3D trajectory in court metres',
               ['0.351 m median against paper 1, 6.9% of frames within 5 cm',
                'the trajectory head estimates no camera at inference'],
               fc=SOFT, ec=ACC)
BOT = y

# legend
AX.add_patch(FancyBboxPatch((BX, BOT - 0.062), BW, 0.040,
             boxstyle='round,pad=0.004,rounding_size=0.010', fc='white', ec=LINE, lw=0.8))
# labels sized to the third of the box they sit in, or they collide
for i, (col, fcol, lab) in enumerate([
        (ACC2, WARM, 'learned here'), (LINE, 'white', 'released weights'),
        (ACC, SOFT, 'algebra / optional')]):
    x0 = BX + 0.020 + i * (BW - 0.040) / 3
    AX.add_patch(FancyBboxPatch((x0, BOT - 0.050), 0.018, 0.013,
                 boxstyle='round,pad=0.002,rounding_size=0.005', fc=fcol, ec=col, lw=1.1))
    AX.text(x0 + 0.027, BOT - 0.0435, lab, ha='left', va='center',
            fontsize=BODY_PT - 0.7, color=MUT)


def arc(t, z0, vz, drag=0.9):
    return z0 + vz * t - 0.5 * 9.81 * t ** 2 - drag * t ** 3


# ============================== a. what the camera gives  ->  broadcast frames
head(TOP + 0.006, 'a.  What the camera gives')
FH = PW * (288 / 512) * (W_IN / H_IN)
axf = panel(TOP, FH)
_su, _sv = np.load('fig/src/meta.npy')[:2]
axf.imshow(plt.imread('fig/src/frame.png'))
axf.add_patch(Circle((_su, _sv), 13, fill=False, ec=ACC2, lw=1.3))
axf.set_xlim(0, 512); axf.set_ylim(288, 0)
CW = PW * 0.455
CH = CW * (W_IN / H_IN)
for i, (fn, lab) in enumerate([('fig/src/crop_rgb.png', 'RGB'),
                               ('fig/src/crop_diff.png', 'background difference')]):
    axc = panel(TOP - FH - 0.012, CH, w=CW, x=PX + i * (PW - CW))
    axc.imshow(plt.imread(fn))
    AX.text(PX + i * (PW - CW) + CW / 2, TOP - FH - 0.012 - CH - 0.009, lab,
            ha='center', va='top', fontsize=BODY_PT - 0.9, color=MUT)
ya = cap(TOP - FH - CH - 0.040,
         ['The shuttle is circled. At 7 px across it is a smudge',
          'in RGB and unambiguous in the difference channel -',
          'which is why the input has a fourth channel.'])
lead(TOP - FH / 2, c_frames)

# ============================ b. five frames are not enough  ->  one stroke
head(ya - 0.024, 'b.  Why one stroke, not five frames')
PH = 0.132
ax = panel(ya - 0.030, PH)
t = np.linspace(0, 0.95, 300)
TC, HALF = 0.3835, 0.0835
base = arc(t, 2.4, 5.2)
for k in (14.0, -14.0):
    ax.plot(t, base + k * (t - TC) ** 3, color=LINE, lw=1.0)
ax.plot(t, base, color=ACC, lw=1.3)
ax.axvspan(TC - HALF, TC + HALF, color=ACC2, alpha=0.13, lw=0)
tw = np.linspace(TC - HALF, TC + HALF, 30)
ax.plot(tw, arc(tw, 2.4, 5.2), color=ACC2, lw=2.6)
ax.text(TC, 0.40, '5 frames\n0.167 s', ha='center', va='bottom',
        fontsize=BODY_PT - 1.1, color=ACC2, linespacing=1.25)
hi = arc(np.array([0.90]), 2.4, 5.2)[0]
ax.annotate('', xy=(0.90, hi + 14 * (0.90 - TC) ** 3), xytext=(0.90, hi - 14 * (0.90 - TC) ** 3),
            arrowprops=dict(arrowstyle='<->', color=MUT, lw=0.8, shrinkA=0, shrinkB=0))
ax.text(0.845, 4.5, 'metres\napart', ha='right', va='center',
        fontsize=BODY_PT - 1.1, color=MUT, linespacing=1.25)
ax.text(0.045, 5.6, 'identical inside the window', ha='left', va='top',
        fontsize=BODY_PT - 1.1, color=INK)
ax.set_xlim(0, 0.97); ax.set_ylim(0, 6.1)
yb = cap(ya - 0.030 - PH - 0.014,
         ['Three flights, one image track. Over 0.167 s the arc',
          'is straight and the height is unobservable; a whole',
          'stroke is long enough for drag and gravity to bend it.'])
lead(ya - 0.030 - PH / 2, c_stroke)

# ================================ c. pixels make it worse  ->  height head
head(yb - 0.024, 'c.  Adding pixels makes it worse')
PH2 = 0.088
ax = panel(yb - 0.030, PH2)
for i, (lab, v, col) in enumerate([('track only', 0.321, ACC),
                                   ('+ trunk pixel features', 0.510, ACC2)]):
    ax.barh(i, v, height=0.40, color=col, alpha=0.88)
    ax.text(v + 0.018, i, '%.3f m' % v, va='center', fontsize=BODY_PT - 0.9, color=INK)
    ax.text(0.012, i - 0.30, lab, va='bottom', fontsize=BODY_PT - 0.9, color=MUT)
ax.set_xlim(0, 0.74); ax.set_ylim(-0.85, 1.72); ax.invert_yaxis()
yc = cap(yb - 0.030 - PH2 - 0.014,
         ['The 64-d trunk feature at the detected point drowns',
          'the 14-d track it was meant to help. This is the',
          'strongest form of the claim in the title.'])
lead(yb - 0.030 - PH2 / 2, c_head)

# ============================== d. who can gate the answer  ->  refinement
head(yc - 0.024, 'd.  Who knows when the answer is wrong')
PH3 = 0.108
ax = panel(yc - 0.030, PH3)
for i, (lab, r) in enumerate([('detector confidence', 0.018), ('trajectory jerk', 0.099),
                              ('closed-form scale', 0.025), ("solver's own residual", 0.62)]):
    usable = r > 0.3
    ax.barh(i, max(r, 0.006), height=0.42, color=ACC if usable else LINE)
    ax.text(0.012, i - 0.30, lab, va='bottom', fontsize=BODY_PT - 0.9,
            color=INK if usable else MUT)
ax.text(0.645, 3.0, 'the only usable one', ha='left', va='center',
        fontsize=BODY_PT - 0.9, color=ACC)
ax.set_xlim(0, 1.02); ax.set_ylim(-0.85, 3.75); ax.invert_yaxis()
yd = cap(yc - 0.030 - PH3 - 0.014,
         ['Rank correlation with the true 3D error. Nothing the',
          'model can report predicts it; the solver measures its',
          'own fit after the fact, and the gate reads that.'])
lead(yc - 0.030 - PH3 / 2, c_gate)

# ---------------------------------------------------------------- footer
AX.text(BX + BW / 2, min(yd, BOT - 0.075) - 0.010,
        'Height error reaches 3D error through the closed form with a measured',
        ha='center', va='top', fontsize=BODY_PT, color=INK, style='italic')
AX.text(BX + BW / 2, min(yd, BOT - 0.075) - 0.010 - LS,
        'gain of 4.1x: one centimetre of height is four centimetres of position.',
        ha='center', va='top', fontsize=BODY_PT, color=INK, style='italic')

for ext in ('svg', 'pdf'):
    fig.savefig('fig/arch.' + ext, bbox_inches='tight')
fig.savefig('fig/arch.png', dpi=400, bbox_inches='tight', facecolor='white')
print('fig/arch.svg / .pdf / .png')
