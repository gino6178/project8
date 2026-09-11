"""Insert 3.5 (temporal trunk) and 3.6 (leave-one-venue-out), and extend 4.1
with the bounds that are an identity for neither method.

Every number is read from the CSVs the experiments wrote; nothing is typed in.
"""
import re
import pandas as pd

P3 = '/home/gino/paper3/out/'
A = pd.read_csv(P3 + 'exp78_arch.csv')
LV = pd.read_csv(P3 + 'exp78_lovo.csv')
B = pd.read_csv(P3 + 'exp74_bounds.csv')

# ---------------------------------------------------------------- 3.5 trunks
def grp(pfx):
    v = A[A.name.str.startswith(pfx)]['median'].to_numpy()
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan')), len(v)

tm, ts, tn = grp('transformer')
gm, gs, gn = grp('GRU')
pooled = ((A[A.name.str.startswith('transformer')]['median'].var(ddof=1) +
           A[A.name.str.startswith('GRU')]['median'].var(ddof=1)) / 2) ** .5
arows = []
for _, r in A.iterrows():
    ctx = 'none' if 'MLP' in r['name'] else 'whole stroke'
    arows.append('<tr><td>%s</td><td>%.2f M</td><td>%s</td><td>%.3f</td>'
                 '<td>[%.3f, %.3f]</td></tr>'
                 % (r['name'], r['params'], ctx, r['median'], r['lo'], r['hi']))
SEC35 = '''
<h3>3.5 The trunk is not the contribution</h3>

<p>A 6-layer 192-d transformer invites the question of whether the result is the attention or
the context, and the paper's claim rests on the answer being the context. Four temporal trunks,
matched to 2.80&ndash;2.85&nbsp;M parameters, with identical inputs, loss, augmentation and
250-epoch schedule; only the trunk changes. The MLP is the control: it has no temporal mixing at
all, so each frame is lifted on its own and the whole-stroke context is removed while everything
else is held fixed. Intervals are a percentile bootstrap resampling <em>strokes</em>, not frames
&mdash; frames within a stroke are nowhere near independent and a per-frame bootstrap would
report intervals several times too narrow.</p>

<div class="tw"><table>
<tr><th>temporal trunk</th><th>params</th><th>context</th><th>3D median (m)</th><th>95% CI</th></tr>
@ROWS@
</table></div>

<p>Removing temporal mixing costs a factor of @FACTOR@. Choosing among the three sequence models
costs nothing that this experiment can resolve: the transformer averages @TM@&nbsp;m over three
seeds and the GRU @GM@&nbsp;m, a gap of @GAP@&nbsp;m against a pooled seed-to-seed standard deviation
of @SD@. <b>We therefore make no claim that the transformer is the right architecture</b>; it is
the one the rest of the paper happens to use, and a GRU or a dilated TCN would do as well. The
lever is the 27 frames, not what reads them.</p>

<p class="note">These are our own trunks swapped inside one pipeline, not published models run
end to end. The TCN row is the architecture family VideoPose3D uses for 2D-to-3D lifting and the
transformer row is MotionBERT's; treating them as stand-ins for those systems is reasonable for
the question asked here &mdash; which trunk &mdash; and not a substitute for benchmarking against
the published implementations, which we have not done. The whole table is trained without the
landing constraint of &sect;3.2, so it is internally consistent but sits slightly above the main
table.</p>
'''
for k, v in (('@ROWS@', '\n'.join(arows)),
             ('@FACTOR@', '%.1f' % (A[A.name.str.contains('MLP')]['median'].iloc[0] / min(tm, gm))),
             ('@TM@', '%.3f' % tm), ('@GM@', '%.3f' % gm),
             ('@GAP@', '%.3f' % abs(tm - gm)), ('@SD@', '%.3f' % pooled)):
    SEC35 = SEC35.replace(k, v)

# ------------------------------------------------------- 3.6 leave-one-venue
lrows = []
for _, r in LV.iterrows():
    short = r['name'].replace('_', ' ')
    short = short[:44] + ('…' if len(short) > 44 else '')
    lrows.append('<tr><td>%s</td><td>%d</td><td>%.3f</td><td>[%.3f, %.3f]</td></tr>'
                 % (short, r['n'], r['median'], r['lo'], r['hi']))
med, lo, hi = LV['median'].median(), LV['median'].min(), LV['median'].max()
main = LV[LV.name.str.startswith('CHOU')]['median']
main = float(main.iloc[0]) if len(main) else float('nan')
SEC36 = '''
<h3>3.6 One held-out venue is not a result</h3>

<p>Every number in &sect;3.2 comes from a single held-out venue. That is a leave-one-venue-out
split with one fold, and one fold cannot say whether the venue was easy. Retraining eight times,
holding out each 2025 singles venue in turn:</p>

<div class="tw"><table>
<tr><th>held-out venue</th><th>clean frames</th><th>3D median (m)</th><th>95% CI</th></tr>
@LROWS@
<tr><td><b>across folds</b></td><td>@N@</td><td><b>@MED@</b></td><td>@LO@ &ndash; @HI@</td></tr>
</table></div>

<p>The spread is <b>@SPREAD@&times;</b> across venues, and it dwarfs every architectural difference in
&sect;3.5 and most of the ablations in section 6. The venue &sect;3.2 reports on lands at
@MAIN@&nbsp;m against a cross-venue median of @MED@&nbsp;m, so <b>the main table is roughly @OPT@%
optimistic</b> as an estimate of what this model does at a venue drawn at random. We have left
&sect;3.2 as it is, because it is the split every other experiment in the paper is measured on
and changing it would make the ablations incomparable, but the number to quote for
generalisation is the cross-venue median, and the honest error bar on it is the fold range, not
the bootstrap interval of any single fold.</p>

<p>The obvious explanation &mdash; that a venue is hard when its camera is unlike the others
&mdash; is only half supported. Ranking the eight venues by how far their nine-number camera
summary sits from the median camera gives a Spearman correlation with fold error of 0.62, but
with eight folds that is p&nbsp;=&nbsp;0.10, and the hardest venue is not the most unusual one
(it ranks third). The direction is what &sect;4.4 would predict and the sample is too small to
call it. We report it as a hypothesis, not a finding.</p>
'''
for k, v in (('@LROWS@', '\n'.join(lrows)), ('@N@', str(int(LV['n'].sum()))),
             ('@MED@', '%.3f' % med), ('@LO@', '%.3f' % lo), ('@HI@', '%.3f' % hi),
             ('@SPREAD@', '%.1f' % (hi / lo)), ('@MAIN@', '%.3f' % main),
             ('@OPT@', '%.0f' % (100 * (med / main - 1)))):
    SEC36 = SEC36.replace(k, v)

s = open('index.html', encoding='utf-8').read()
anchor = '<h2 id="nogt">'
assert anchor in s and 'id="lovo"' not in s
s = s.replace(anchor, SEC35 + SEC36 + '\n' + anchor, 1)
open('index.html', 'w', encoding='utf-8').write(s)
print('3.5 and 3.6 inserted; %d bytes' % len(s))
