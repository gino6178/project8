"""Add 3.3's learned-uncertainty result and 4.5's cross-sport CRLB study."""
import re
import pandas as pd

CS = pd.read_csv('/home/gino/paper3/out/exp82_crosssport.csv')
rows = []
for sp in CS.sport.unique():
    d = CS[CS.sport == sp]
    rows.append('<tr><td>%s</td><td>%.1f</td><td>%.0f</td><td>%.1f</td><td>%.3f</td>'
                '<td>%.0f&times;</td></tr>'
                % (sp, d.v_T.iloc[0], d.frames.median(), d.crlb_5f.median(),
                   d.crlb_full.median(), d.ratio.median()))
CSROWS = '\n'.join(rows)
rho = CS.groupby('sport').ratio.median().corr(CS.groupby('sport').v_T.first(),
                                              method='spearman')

UNC = '''
<h4>A learned confidence signal, and what it replaces</h4>

<p>"The model cannot tell when it is right" is a claim about the signals we could read off a
model that was never asked to produce one, which is a weaker claim than it sounds. The missing
control is a head <em>trained</em> to predict the error. We added two: a heteroscedastic
log-&sigma; on the height, and a direct regression on the per-frame 3D error. Both are trained
against detached targets, and in the row marked <i>read-out</i> the trunk is detached as well, so
the uncertainty outputs cannot cost the height head any capacity.</p>

<div class="tw"><table>
<tr><th>confidence signal</th><th>&rho; with the 3D error</th><th>3D median (m)</th><th>keep the best 80%</th></tr>
<tr><td>detector confidence</td><td>&minus;0.018</td><td>&mdash;</td><td>&mdash;</td></tr>
<tr><td>trajectory jerk</td><td>+0.099</td><td>&mdash;</td><td>worse than not selecting</td></tr>
<tr><td>closed-form scale</td><td>+0.025</td><td>&mdash;</td><td>&mdash;</td></tr>
<tr><td>learned log-&sigma;, joint</td><td>+0.393</td><td>0.374</td><td>&mdash;</td></tr>
<tr><td>learned error head, joint</td><td class="good">+0.430</td><td>0.374</td><td>0.340 vs 0.403</td></tr>
<tr><td>learned error head, read-out</td><td class="good">+0.406</td><td>0.350</td><td>0.335 vs 0.388</td></tr>
<tr><td><i>no uncertainty head</i></td><td><i>&mdash;</i></td><td><i>0.337</i></td><td><i>&mdash;</i></td></tr>
</table></div>

<p>Trained for it, the model can tell &mdash; four times better than any signal already present,
and enough to make selective prediction work where it previously backfired: discarding the least
confident 20% of frames improves the median by 14%, against the jerk-based version making it
worse. Attached to the trunk this costs 11% of the main metric; detached it costs 4%, which is
about two seed standard deviations (&sect;3.5).</p>

<p>This matters beyond confidence reporting, because the gate above is the one place the paper
still needs the optimiser at inference. A learned signal at &rho;&nbsp;=&nbsp;0.41 is not yet the
solver's <code>rms_px</code>, which is a post-hoc measurement of a fit rather than a prediction,
and we have not shown the two are interchangeable for gating. But it moves the question from
"no learned signal works" to "how good does a learned signal have to be", and it is the obvious
route to dropping the 400&nbsp;ms refinement pass entirely. We flag it as the most promising
open direction rather than a solved one.</p>
'''

CROSS = '''
<h3>4.5 Is any of this specific to badminton?</h3>

<p>The paper's central claim &mdash; five frames cannot observe a ball's height and a whole arc
can &mdash; is a statement about the conditioning of an inverse problem, not about our network,
so it can be tested for other sports without a frame of their video. The Cram&eacute;r&ndash;Rao
bound on 3D position follows from the Fisher information of the launch state and the projection
Jacobian; it is the best <em>any</em> estimator could do. A sport enters through four numbers:
terminal velocity, court size, camera placement, and launch speeds.</p>

<div class="tw"><table>
<tr><th>sport</th><th>v<sub>T</sub> (m/s)</th><th>frames in the arc</th><th>CRLB, 5 frames (m)</th><th>CRLB, whole arc (m)</th><th>ratio</th></tr>
@CSROWS@
</table></div>

<p>A five-frame window is worse conditioned than the whole arc in every sport simulated, by one
to three orders of magnitude, and <b>the ratio does not track terminal velocity</b> (Spearman
@RHO@ across the five sports). So the effect is not badminton's extreme drag: it is that five
frames at 30&nbsp;fps span too little of any ballistic arc to separate depth from height, and
drag only changes how much of the arc a given number of frames buys you. Table tennis shows the
smallest ratio because its arcs are the shortest &mdash; 17 frames against roughly 52 elsewhere
&mdash; which is the same mechanism seen from the other side.</p>

<p class="note">This is a simulation, and the only sport whose four numbers were measured rather
than taken from a reference is badminton. It says the <em>conditioning</em> argument generalises;
it does not say a model trained here would work on tennis video, which we have not tested and
have no data for. The badminton whole-arc bound of 0.061&nbsp;m is consistent with the 0.160&nbsp;m
we measured for real long shots in earlier work, which is the sanity check that the machinery is
producing sensible magnitudes. The five-frame column should be read as "the bound diverges",
not as a length: with six launch parameters and ten observations the Fisher matrix is nearly
singular, which is precisely the claim.</p>
'''
CROSS = CROSS.replace('@CSROWS@', CSROWS).replace('@RHO@', '%.2f' % rho)

s = open('index.html', encoding='utf-8').read()
a = '<h3>3.4 What the extra output costs the detector</h3>'
assert a in s and 'A learned confidence signal' not in s
s = s.replace(a, UNC + '\n' + a, 1)
b = '<h2 id="ceiling">5. What the labels can support</h2>'
assert b in s
s = s.replace(b, CROSS + '\n' + b, 1)
open('index.html', 'w', encoding='utf-8').write(s)
print('added; %d bytes' % len(s))
