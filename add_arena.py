"""Add the synthetic-arena section. One arena, the constant-drag one, because
EXP85 found real tracks do not support the extra parameter."""
import pandas as pd, numpy as np

P3 = '/home/gino/paper3/out/'
B = pd.read_csv(P3 + 'exp84_synth_const.csv')
ORDER = [('ours (trained on synth)', 'ours'),
         ('paper 1 solver, no learned prior', 'paper 1 solver'),
         ('MonoTrack, given the true half', 'MonoTrack')]
rows = []
best = min(B[(B.method == m) & B.ok].err.median() for m, _ in ORDER)
for m, lab in ORDER:
    d = B[(B.method == m)]
    g = d[d.ok]
    cls = ' class="good"' if abs(g.err.median() - best) < 1e-9 else ''
    rows.append('<tr><td>%s</td><td>%.0f%%</td><td%s>%.3f</td><td>%.3f</td>'
                '<td>%.3f</td><td>%.2f</td></tr>'
                % (lab, 100 * d.ok.mean(), cls, g.err.median(),
                   g.err.quantile(.9), g.ez.median(), g.rep.median()))
TABLE = '\n'.join(rows)
o = B[(B.method == ORDER[0][0]) & B.ok].err.median()
s1 = B[(B.method == ORDER[1][0]) & B.ok].err.median()
s2 = B[(B.method == ORDER[2][0]) & B.ok].err.median()

SEC = '''
<h3>4.6 An arena where the 3D is true</h3>

<p>Every number so far is scored against paper 1's coordinates or against the one human
annotation of &sect;4.1. A reviewer can still ask what the error would be against 3D that is
simply correct. So we built flights whose 3D is true by construction and looked at them through
the cameras and the detector noise this project measured.</p>

<ul class="contrib">
<li><b>The flights are generated, so their 3D is exact.</b> Shot mix, duration and distance are
drawn from the 13,141 real strokes of the dataset's own annotation &mdash; smash 0.50&nbsp;s over
9.6&nbsp;m, clear 1.23 over 11.4, and so on &mdash; and the launch is solved so the flight
actually reaches its target under the drag model used.</li>
<li><b>The cameras are the eight real ones</b>, from their measured corners and focal lengths, not
invented viewpoints.</li>
<li><b>The pixel noise is resampled, not modelled.</b> Each track is corrupted with a contiguous
run drawn from 201,515 measured error vectors of our detector against the labels, so the heavy
tail (median 3.7&nbsp;px, p99 404&nbsp;px) and the temporal correlation both survive. A Gaussian
would erase exactly the failure mode &sect;4.2 showed matters.</li>
<li><b>The drag model is the one the baselines fit.</b> Fitted to real detected tracks, a
velocity-dependent drag coefficient collapses to zero (median k&#8322;&nbsp;=&nbsp;0.0000, better on
only 21% of strokes), so the data does not support it and the arena uses constant drag. That is
also what paper 1's solver and MonoTrack assume, so <b>the arena hands both baselines the exact
functional form they fit</b>. The bias runs toward them, not toward us.</li>
</ul>

<div class="tw"><table>
<tr><th>on 200 held-out synthetic strokes, true 3D</th><th>returned</th><th>3D median (m)</th><th>p90 (m)</th><th>|&Delta;z| (m)</th><th>reprojection (px)</th></tr>
@TABLE@
</table></div>

<p>The ordering is the same as on real data and the gaps are wider: @O@&nbsp;m against
@S1@&nbsp;m for paper 1's solver run without a learned prior, and @S2@&nbsp;m for MonoTrack given
the court half it cannot infer. Nothing here is fitted to a pseudo-label, and the model was
trained on this arena's own training split, so this is the learned map competing with the
optimisers on ground that favours them.</p>

<p>The last column is the one this paper's framing demands, and it is only measurable here.
Section&nbsp;4.1's reprojection is taken against the <em>detected</em> pixels and is therefore
zero by construction for our method. In the arena the reference is the clean projection of the
true 3D, while our estimate was solved from the corrupted pixels, so the number is a real
measurement for all three. <b>On 2D the three methods are indistinguishable &mdash; @R0@,
@R1@ and @R2@&nbsp;px &mdash; while in 3D they differ by a factor of @RATIO@.</b> Our
@R0@&nbsp;px is essentially the injected detector error itself (median 3.74&nbsp;px), which is what
a closed-form horizontal solve must give: the pixel error passes through unchanged, neither
amplified nor filtered. This is the whole argument in one row. A method that starts from 2D and
is judged on 2D looks fine; the disagreement is entirely in the direction the image cannot
see.</p>

<p class="note">What this does <em>not</em> establish is that the numbers transfer to broadcast.
The arena reproduces the geometry, the shot statistics and the detector's error distribution, but
its flights come from the same integrator the labels' solver uses, so a systematic error shared
by that integrator and the real shuttle would be invisible here. It is evidence that the
comparison in section 3 is not an artefact of scoring against an optimiser's output; it is not a
second, independent measurement of accuracy. The honest reading is that two very different
checks &mdash; a human landing annotation in &sect;4.1 and true 3D here &mdash; both put the
learned map ahead of the optimisers, and neither is the pseudo-label.</p>

<figure>
<img src="fig/arena.png" alt="The synthetic arena">
<figcaption><b>Figure 5. The arena.</b> <b>(a)</b> generated flights in true 3D on a real court.
<b>(b)</b> the pixel error injected into them is resampled from our detector's measured errors,
not drawn from a Gaussian; the two distributions are the same by construction. <b>(c)</b> one
held-out stroke at the median of our per-stroke error, with all three methods against the truth.
<b>(d)</b> the whole held-out set as cumulative distributions.</figcaption>
</figure>
'''
r = {lab: B[(B.method == m) & B.ok].rep.median() for m, lab in ORDER}
for k, v in (('@TABLE@', TABLE), ('@O@', '%.3f' % o), ('@S1@', '%.3f' % s1),
             ('@S2@', '%.3f' % s2),
             ('@R0@', '%.2f' % r['ours']), ('@R1@', '%.2f' % r['paper 1 solver']),
             ('@R2@', '%.2f' % r['MonoTrack']), ('@RATIO@', '%.1f' % (s2 / o))):
    SEC = SEC.replace(k, v)

s = open('index.html', encoding='utf-8').read()
a = '<h2 id="ceiling">5. What the labels can support</h2>'
assert a in s and 'id="arena"' not in s and '4.6 An arena' not in s
s = s.replace(a, SEC + '\n' + a, 1)
open('index.html', 'w', encoding='utf-8').write(s)
print('4.6 added; %d bytes' % len(s))
