"""Figure 5 - what the synthetic arena actually is.

The arena's whole claim is that its 3D is true and its 2D is corrupted the way
a real detector corrupts it. Both are things a reader should be able to SEE,
so this figure shows the flights, the noise that was injected, and what each
method returns on one of them -- not just a table of medians.

FIGURE CONTRACT
  - every trajectory drawn is read from the generated index, not re-simulated
  - panel (b) plots the clean projection and the corrupted detections that were
    actually written to the index, so the noise shown IS the noise used
  - panel (c) is one held-out stroke chosen at the MEDIAN of our per-stroke
    error, not the best case
"""
import sys, io, contextlib
sys.path.insert(0, '/home/gino/paper3/src'); sys.path.insert(0, '/home/gino/paper3/pylibs')
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geom, courtcam as cc, monotrack_baseline as mt, feasible_gpu as fg

P3 = '/home/gino/paper3/'
# EXP85: fitted to real detected tracks, the velocity-dependent coefficient
# collapses to zero (median k2 = 0.0000), so the data does not support it.
# The arena uses the constant model -- which is also what both optimisers
# fit, so the arena is biased toward the baselines, not against them.
DRAG = 'const'
INK, MUT, LINE = '#141b1e', '#5c6b70', '#dfe6e5'
OURS, THEM, MONO, TRUE = '#0e7c7b', '#e8552f', '#7a5ea8', '#141b1e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': LINE, 'xtick.color': MUT, 'ytick.color': MUT,
                     'text.color': INK, 'axes.labelcolor': INK})
W, L = cc.COURT_W, cc.COURT_L

D = pd.read_csv(P3 + 'out/p2_synth_%s_index.csv' % DRAG)
CAM = pd.read_csv(P3 + 'out/p2_cameras.csv').set_index('venue')


def proj(v):
    r = CAM.loc[v]
    corn = np.array([[r['c%d%s' % (k, a)] for a in 'uv'] for k in range(4)])
    Gh = geom.homography_from_corners(torch.tensor(corn, dtype=torch.float64)[None])[0].numpy()
    return cc.projection_from_focal(Gh, float(r['f']), float(r['cx']), float(r['cy']))


fig = plt.figure(figsize=(13.6, 3.9), dpi=220)
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(1, 4, width_ratios=[1.30, 1.0, 1.0, 1.0], wspace=0.42)

# ------------------------------------------------- (a) the court, in 3D
# The arena is a real court in metres seen by a real measured camera, so draw
# it that way: court lines on the floor, the net at its true height, and the
# generated flights as they actually are in space.
A = fig.add_subplot(gs[0], projection='3d')
A.set_box_aspect((W, L * 0.62, 5.2))
# court outline and the singles side lines
for xa, xb, ya, yb in ((0, W, 0, 0), (0, W, L, L), (0, 0, 0, L), (W, W, 0, L),
                       (0, W, L / 2, L / 2)):
    A.plot([xa, xb], [ya, yb], [0, 0], color=MUT, lw=0.9)
for xg in (0.46, W - 0.46):
    A.plot([xg, xg], [0, L], [0, 0], color=LINE, lw=0.7)
for yg in (L / 2 - 1.98, L / 2 + 1.98, 0.76, L - 0.76):
    A.plot([0, W], [yg, yg], [0, 0], color=LINE, lw=0.7)
# the net: posts at 1.55 m, centre sags to 1.524 m
xs = np.linspace(0, W, 40)
u_ = 2 * xs / W - 1
A.plot(xs, np.full_like(xs, L / 2), 1.55 - (1.55 - 1.524) * (1 - u_ ** 2),
       color=INK, lw=1.6)
for xp in (0, W):
    A.plot([xp, xp], [L / 2, L / 2], [0, 1.55], color=INK, lw=1.6)
for k in D.stroke_key.drop_duplicates().sample(45, random_state=3):
    g = D[D.stroke_key == k]
    A.plot(g.x, g.y, g.z, color=OURS, lw=0.7, alpha=0.55)
A.set_xlim(0, W); A.set_ylim(0, L); A.set_zlim(0, 7)
A.view_init(elev=14, azim=-60)
A.set_xticks([]); A.set_yticks([]); A.set_zticks([])   # ticks bled into panel (b)
A.tick_params(labelsize=7, colors=MUT, pad=-2)
A.set_title('(a) generated flights, true 3D', fontsize=9, loc='left', pad=-2, x=0.02)
A.text2D(0.02, 0.56, '0 to 6 m\nhigh', transform=A.transAxes, fontsize=7.5, color=MUT)
try:
    A.set_position(A.get_position().expanded(1.16, 1.20))
except Exception:
    pass
A.grid(False)
try:
    A.xaxis.pane.fill = A.yaxis.pane.fill = A.zaxis.pane.fill = False
    for pane in (A.xaxis.pane, A.yaxis.pane, A.zaxis.pane):
        pane.set_edgecolor('none')
except Exception:
    pass

# ------------------------- (b) does the injected noise match the measured one?
B = fig.add_subplot(gs[1])
ERR = np.load(P3 + 'out/det_err_vecs.npy')
real = np.linalg.norm(ERR, axis=1)
inj = []
for k in D.stroke_key.drop_duplicates().sample(400, random_state=1):
    g = D[D.stroke_key == k]
    Pm = proj(g.venue.iloc[0])
    Xg = g[['x', 'y', 'z']].to_numpy()
    q = Pm @ np.column_stack([Xg, np.ones(len(Xg))]).T
    inj.append(np.linalg.norm((q[:2] / q[2]).T - g[['u', 'v']].to_numpy(), axis=1))
inj = np.concatenate(inj)
bins = np.logspace(-1, 3, 50)
B.hist(real, bins=bins, density=True, color=MUT, alpha=0.45,
       label='measured, real detector')
B.hist(inj, bins=bins, density=True, histtype='step', color=THEM, lw=1.6,
       label='injected into the arena')
B.set_xscale('log')
B.set_xlabel('pixel error (px, log)'); B.set_ylabel('density')
B.set_title('(b) real noise, resampled', fontsize=9, loc='left', pad=7)
B.legend(frameon=False, fontsize=7.3)
for q_, c in ((np.median(real), MUT), (np.quantile(real, .99), MUT)):
    B.axvline(q_, color=c, lw=0.8, ls=(0, (3, 3)))
B.text(np.median(real), B.get_ylim()[1] * 0.44, ' median\n %.1f px' % np.median(real),
       fontsize=7, color=MUT, va='top')
B.text(np.quantile(real, .99), B.get_ylim()[1] * 0.44,
       ' p99\n %.0f px' % np.quantile(real, .99), fontsize=7, color=MUT, va='top')
B.grid(True, color=LINE, lw=0.6); B.set_axisbelow(True)
for s_ in ('top', 'right'):
    B.spines[s_].set_visible(False)

# --------------------------------- (c,d) one held-out stroke, all three methods
_MY = None
sys.argv = ['x', '--mirror', '0', '--index', P3 + 'out/p2_synth_%s_index.csv' % DRAG]
import os
os.chdir(P3)
_src = open('experiments/exp64_stroke_pixels.py').read()
_OLD = "net = Net(d=DIM, nl=NL, fd=FD, arch=ARCH).to(DEV)"
head = _src.split('\nbest = 1e9')[0].replace(_OLD, "net = Net(d=192, nl=6, fd=0, arch=ARCH).to(DEV)")
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
VN = {i: v for v, i in VI.items()}
net.load_state_dict(torch.load('out/synth_%s.pt' % DRAG, map_location='cpu')); net.eval()

res = pd.read_csv('out/exp84_synth_%s.csv' % DRAG)
ours = res[(res.method == 'ours (trained on synth)') & res.ok].sort_values('err')
pick = ours.iloc[len(ours) // 2].sid                      # the median stroke
st = [s for s in un if s['sid'] == pick][0]
n = st['n']
with torch.no_grad():
    uv, xyz, t, m, w_, rms, vi, ft, ld, lx, sc = [x.to(DEV) for x in pack([st])]
    Z = net(uv, t, m, CAMFEAT[vi])
    XY, _ = geom.xy_from_uvz(G_DEV[vi], COL_DEV[vi], uv.double(), Z.double())
    OU = torch.cat([XY, Z.double()[..., None]], -1)[0, :n].cpu().numpy()
v = VN[int(st['vi'])]; Pm = proj(v)
C = -np.linalg.inv(Pm[:, :3]) @ Pm[:, 3]
uvj = st['uv'][:n].astype(float); tt = st['t'][:n].astype(float); gt = st['xyz'][:n]
h = np.column_stack([uvj, np.ones(n)]) @ np.linalg.inv(Pm[:, :3]).T
h /= np.linalg.norm(h, axis=1, keepdims=True)
if np.dot(h[0], np.array([W / 2, L / 2, 1.5]) - C) < 0:
    h = -h
d0 = float(np.linalg.norm(C - np.array([W / 2, L / 2, 1.5])))
SO = fg.reconstruct_batch([dict(uv=uvj, t=tt.astype(np.float64), P=Pm, C=C, rays=h,
                                must_cross=False, X_prior=C[None, :] + d0 * h)])[0].get('X')
SO = np.asarray(SO, float) if SO is not None else None
MO = np.asarray(mt.reconstruct(uvj, tt, Pm, C,
                               start_half='near' if gt[0, 1] <= L / 2 else 'far'), float)

Cc = fig.add_subplot(gs[2])
Cc.plot(tt, gt[:, 2], '-', color=TRUE, lw=2.0, label='true (generated)')
Cc.plot(tt, OU[:, 2], '-o', color=OURS, lw=1.4, ms=2.6, label='ours')
if SO is not None:
    Cc.plot(tt, SO[:, 2], '-', color=THEM, lw=1.3, label='paper 1 solver')
Cc.plot(tt, MO[:, 2], '-', color=MONO, lw=1.3, label='MonoTrack')
Cc.set_xlabel('time (s)'); Cc.set_ylabel('height (m)')
Cc.set_title('(c) a held-out stroke, at our median', fontsize=9, loc='left', pad=7)
Cc.legend(frameon=False, fontsize=7.3)
Cc.grid(True, color=LINE, lw=0.6); Cc.set_axisbelow(True)
for s in ('top', 'right'):
    Cc.spines[s].set_visible(False)

Dd = fig.add_subplot(gs[3])
for tag, col, lab in (('ours (trained on synth)', OURS, 'ours'),
                      ('paper 1 solver, no learned prior', THEM, 'paper 1 solver'),
                      ('MonoTrack, given the true half', MONO, 'MonoTrack')):
    e = res[(res.method == tag) & res.ok].err.dropna().to_numpy()
    e = np.sort(e)
    Dd.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, lw=1.8, label=lab)
Dd.set_xscale('log'); Dd.set_xlim(0.03, 20)
Dd.set_xlabel('3D error against the true position (m)')
Dd.set_ylabel('fraction of strokes')
Dd.set_title('(d) all 200 held out', fontsize=9, loc='left', pad=7)
Dd.legend(frameon=False, fontsize=7.5, loc='lower right')
Dd.grid(True, color=LINE, lw=0.6); Dd.set_axisbelow(True)
for s in ('top', 'right'):
    Dd.spines[s].set_visible(False)

for ext in ('png', 'pdf', 'svg'):
    fig.savefig('/home/gino/project/project8/fig/arena.' + ext,
                facecolor='white', bbox_inches='tight')
print('fig/arena.{png,pdf,svg}  (stroke %s)' % pick)
