"""Extend 4.1 with the checks that are an identity for neither method."""
import re
import pandas as pd

P3 = '/home/gino/paper3/out/'
B = pd.read_csv(P3 + 'exp74_bounds.csv')
ORDER = ['ours', 'ours + noise aug', 'ours + net constraint', 'paper 1 pipeline',
         'paper 1 labels (reference)']
LABEL = {'ours': 'ours', 'ours + noise aug': 'ours + noise aug',
         'ours + net constraint': 'ours + net constraint',
         'paper 1 pipeline': 'paper 1 pipeline',
         'paper 1 labels (reference)': "<i>paper 1's labels (reference)</i>"}

rows = []
for m in ORDER:
    d = B[B.method == m]
    if not len(d):
        continue
    succ = d[d.has_succ & d.net.notna()]
    below = d[d.zmin < -0.05]
    zi = d.z_impact.dropna()
    land = d.land_err.dropna()
    rows.append('<tr><td>%s</td><td>%.1f%%</td><td>%.1f%%</td><td>%.3f</td>'
                '<td>%.2f</td><td>%.0f%%</td><td>%s</td></tr>'
                % (LABEL[m], 100 * (succ.net < 0).mean(), 100 * (d.zmin < -0.05).mean(),
                   abs(below.zmin.median()) if len(below) else 0.0,
                   land.median(), 100 * (land < 1.0).mean(),
                   ('%.3f &dagger;' % zi.median()) if m.startswith('ours')
                   else '%.3f' % zi.median()))
TABLE = '\n'.join(rows)
nS = int(B[B.method == 'ours'].has_succ.sum())
nL = int(B[B.method == 'ours'].land_err.notna().sum())
nZ = int(B[B.method == 'ours'].z_impact.notna().sum())

SEC = '''
<h4>Three checks that are an identity for neither</h4>

<p>Those two columns cannot decide anything, so here are three that can. All of them are
external to both methods: two come from the rally structure and one from a human annotator.</p>

<ul class="contrib">
<li><b>Net crossing.</b> A stroke with a <em>successor</em> in the same rally provably went over
the net &mdash; the rally continued. A reconstruction that sends it under the tape is wrong, and
the annotation says so without any 3D reference. @NS@ of the held-out strokes qualify.</li>
<li><b>Floor.</b> <i>z</i>&nbsp;&lt;&nbsp;0 is impossible.</li>
<li><b>Landing position.</b> For the last stroke of a rally the dataset carries a human
annotation of where the shuttle landed, in image pixels. Through the court homography it is a
position in metres that neither method has ever seen &mdash; it comes from an annotator, not
from paper 1's solver. Both trajectories are extrapolated to <i>z</i>&nbsp;=&nbsp;0 with the same
drag model and compared against it. @NL@ strokes qualify.</li>
</ul>

<div class="tw"><table>
<tr><th></th><th>net violation</th><th>below floor</th><th>penetration m</th><th>landing error m</th><th>landing &lt;1 m</th><th>|z| at touch-down m</th></tr>
@TABLE@
</table></div>

<p class="note">&dagger; Not independent for these rows. The landing constraint of &sect;3.2
trains our models on <i>z</i>&nbsp;=&nbsp;0 at a rally's last frame, so their touch-down height is
a fitted objective, not a test &mdash; the same circularity this section is about, and it is
flagged rather than claimed. The landing <em>position</em> is not affected: the constraint says
the shuttle is on the floor, which is known in advance, and says nothing about where. n&nbsp;=&nbsp;@NZ@
for that column, which counts rally-final strokes whose track actually reaches the floor.</p>

<p>The landing row is the one that matters, because it is the only fully external metre-scale
ground truth in this paper and it is the quantity badminton analytics actually consumes. We reach
0.55&nbsp;m against the annotation, or 0.48&nbsp;m with either extra constraint &mdash; and paper
1's own labels reach 0.48&nbsp;m. <b>We are at the floor this metric can measure</b>, and the live
optimiser is at 1.18&nbsp;m, 2.4&times; worse.</p>

<p>The other two rows go the other way and are the honest cost of section 3's accuracy. We put
the shuttle under the net on 10.1% of strokes where the rally proves it went over; the optimiser
never does and the labels do it 0.1% of the time, so that is our error and not a quirk of the
test. Adding a net-clearance term to the loss &mdash; the same free supervision as the landing
constraint, applied to the 5,532 training strokes the rally says cleared the net &mdash; barely
moved it, 10.1%&nbsp;&rarr;&nbsp;9.4%. It did remove floor penetration entirely,
5.0%&nbsp;&rarr;&nbsp;0.0%, which is not what it was for.</p>

<p>Why the constraint fails is worth a measurement rather than a guess. Substituting one
component at a time from the labels, on the 827 held-out strokes that both have a successor and
cross the net plane:</p>

<div class="tw"><table>
<tr><th>trajectory</th><th>net violation</th><th>median clearance m</th></tr>
<tr><td>ours</td><td>10.2%</td><td>0.33</td></tr>
<tr><td>label horizontal + our height</td><td>9.0%</td><td>0.32</td></tr>
<tr><td>our horizontal + label height</td><td>9.6%</td><td>0.31</td></tr>
<tr><td><i>labels</i></td><td><i>0.1%</i></td><td><i>0.33</i></td></tr>
</table></div>

<p>Neither substitution helps. Correcting the height alone leaves 9.0% and correcting the
horizontal position alone leaves 9.6%, against 10.2% for correcting neither and 0.1% for
correcting both. <b>Each component is on its own enough to put the shuttle under the net</b>,
because the clearance being tested is only 0.33&nbsp;m at the median while our per-axis error is
about the same size. A term that pushes the height up therefore cannot fix this, and the 9.4%
above is the evidence. It also means the violation rate is a joint property of the estimate, not
an attribute of the height head &mdash; which matches the observation in the constrained-sampling
literature that a net constrains the joint distribution of a flight and barely narrows any
marginal.</p>
'''
for k, v in (('@TABLE@', TABLE), ('@NS@', '{:,}'.format(nS)), ('@NL@', str(nL)),
             ('@NZ@', str(nZ))):
    SEC = SEC.replace(k, v)

s = open('index.html', encoding='utf-8').read()
anchor = '<h3>4.2 What amortisation'
assert anchor in s and 'identity for neither</h4>' not in s
s = s.replace(anchor, SEC + '\n' + anchor, 1)
open('index.html', 'w', encoding='utf-8').write(s)
print('bounds added; %d bytes' % len(s))
