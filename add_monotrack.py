"""Add the MonoTrack comparison to 1.1, where MonoTrack is already discussed."""
import pandas as pd

D = pd.read_csv('/home/gino/paper3/out/exp81_monotrack.csv')
def row(m):
    d = D[(D.method == m) & D.ok]
    return d.err.median(), d.err.quantile(.9), 100 * (d.net.dropna() < 0).mean()

o = row('ours'); mi = row('MonoTrack, half inferred'); mg = row('MonoTrack, given the true half')

SEC = '''
<h4>MonoTrack, run on our strokes</h4>

<p>Discussing a baseline without running it is not a comparison, so we ran it.
<code>monotrack_baseline.py</code> is a reimplementation of MonoTrack's per-shot solver from the
paper and its released source, given exactly what our model gets: our detector's 2D track and the
venue camera, on 150 held-out strokes.</p>

<div class="tw"><table>
<tr><th>method</th><th>3D median (m)</th><th>p90 (m)</th><th>net violation</th><th>ms / stroke</th></tr>
<tr><td>ours</td><td class="good">@O0@</td><td>@O1@</td><td>@O2@%</td><td>2.8</td></tr>
<tr><td>paper 1 pipeline (&sect;4.2)</td><td>0.709</td><td>&mdash;</td><td>0.0%</td><td>400</td></tr>
<tr><td>MonoTrack, court half inferred</td><td>@MI0@</td><td>@MI1@</td><td>@MI2@%</td><td>5840</td></tr>
<tr><td>MonoTrack, given the true court half</td><td>@MG0@</td><td>@MG1@</td><td>@MG2@%</td><td>5840</td></tr>
</table></div>

<p><b>This is not a reproduction of MonoTrack's published system and should not be read as one.</b>
Their pipeline supplies its own hit segmentation, its own DLT calibration and its own tracker;
what is being run here is their solver on our inputs and our stroke segmentation. Two specific
handicaps are worth naming. Their optimiser constrains the launch to one court half and the
velocity toward the other, which their pipeline knows because it detects who hit the shuttle; our
track-only interface has to infer it and gets it wrong on 33% of strokes, producing time-reversed
fits that still reproject at 5&nbsp;px &mdash; a clean demonstration of &sect;4.1's point that
reprojection barely constrains monocular depth. Choosing the half by reprojection therefore does
not fix it, and the last row simply tells the solver the true half, an advantage we grant it.</p>

<p>The comparison that does mean something is between the two optimisers. Paper 1's pipeline is,
structurally, this solver plus a learned prior to initialise it, and it reaches 0.709&nbsp;m where
MonoTrack's heuristic initialisation reaches @MG0@&nbsp;m. Nothing about the physics differs. On
this data a per-stroke optimiser's accuracy is dominated by what initialises it, which is the
amortisation argument of &sect;4.2 arriving from a third direction: the learned component is
carrying the estimate, and the optimiser is refining whatever it is handed.</p>
'''
for k, v in (('@O0@', '%.3f' % o[0]), ('@O1@', '%.3f' % o[1]), ('@O2@', '%.1f' % o[2]),
             ('@MI0@', '%.3f' % mi[0]), ('@MI1@', '%.2f' % mi[1]), ('@MI2@', '%.1f' % mi[2]),
             ('@MG0@', '%.3f' % mg[0]), ('@MG1@', '%.2f' % mg[1]), ('@MG2@', '%.1f' % mg[2])):
    SEC = SEC.replace(k, v)

s = open('index.html', encoding='utf-8').read()
a = '<h2 id="method">2. Method</h2>'
assert a in s and 'MonoTrack, run on our strokes' not in s
s = s.replace(a, SEC + '\n' + a, 1)
open('index.html', 'w', encoding='utf-8').write(s)
print('added; %d bytes' % len(s))
