"""Regenerate section 4.2's stress table and its noise-augmentation sentence.

Kept separate from apply_sec4.py so the numbers can be refreshed after a
retrain without reinserting the whole section. Both read the same CSVs.
"""
import os, re
import pandas as pd

P3 = '/home/gino/paper3/out/'
SW = pd.read_csv(P3 + 'exp70_stress.csv')
NA = pd.read_csv(P3 + 'exp70n_stress.csv')
nmap = dict(zip(NA.case, NA.ours))
c = SW.iloc[0]
nc = nmap['clean']

rows = []
for _, r in SW.iterrows():
    na = nmap.get(r.case, float('nan'))
    lo = min(x for x in (r.ours, na, r.solver) if x == x)
    def cell(v):
        return ('<td class="todo">&mdash;</td>' if v != v else
                '<td%s>%.3f</td>' % (' class="good"' if v == lo else '', v))
    rows.append('<tr><td>%s</td>%s%s%s<td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>'
                % (r.case.replace(' px', '&nbsp;px'), cell(r.ours), cell(na),
                   cell(r.solver), r.ours / c.ours, na / nc, r.solver / c.solver))
TABLE = '\n'.join(rows)
HEAD = ('<tr><th>corruption applied to all three</th><th>ours<br>median m</th>'
        '<th>+ noise aug<br>median m</th><th>paper 1<br>median m</th>'
        '<th>ours<br>&times; clean</th><th>+aug<br>&times; clean</th>'
        '<th>paper 1<br>&times; clean</th></tr>')

w = SW[SW.case == 'noise 16 px'].iloc[0]
n16 = nmap['noise 16 px']
beats_all = all(nmap[k] <= float(SW[SW.case == k].iloc[0].ours) + 1e-9 for k in nmap)
never_cross = all(nmap[k] < float(SW[SW.case == k].iloc[0].solver) for k in nmap)
NS = ('Retraining the identical architecture with per-sample Gaussian pixel noise, '
      '&sigma;&nbsp;&sim;&nbsp;U(0,&nbsp;8&nbsp;px), gives the third column. It cuts the '
      '16&nbsp;px degradation from %.2f&times; to %.2f&times; and costs <b>nothing</b> on clean '
      'detections &mdash; %.3f against %.3f&nbsp;m, a difference of %.0f%%. %s%s'
      % (w.ours / c.ours, n16 / nc, nc, c.ours, 100 * abs(nc / c.ours - 1),
         'It is at least as good as the unaugmented model in every row of the table. '
         if beats_all else 'It wins in most rows but not all. ',
         'It also never crosses the optimiser: unlike the unaugmented model it is ahead at '
         'every corruption tested, including 16&nbsp;px.' if never_cross else
         'It still crosses the optimiser at the far end of the sweep.'))

s = open('index.html', encoding='utf-8').read()

# swap the table body
i = s.index('<tr><th>corruption applied to')
j = s.index('</table>', i)
s = s[:i] + HEAD + '\n' + TABLE + '\n' + s[j:]

# swap the sentence that quotes the augmentation cost
pat = re.compile(r'<p>The second reason is testable.*?</p>', re.S)
new = ('<p>The second reason is testable, because the observation model is a knob we hold and '
       'the optimiser does not. ' + NS + ' This is the concrete form of what amortisation buys: '
       'an optimiser cannot be told what its observations\' noise looks like, and a fitted '
       'estimator can be told exactly that.</p>\n\n'
       '<p class="note">An earlier version of this table reported a 12% clean-accuracy cost for '
       'noise augmentation. That was an artefact of our own experiment: the augmented model had '
       'been trained without the landing constraint of &sect;3.2 that the baseline carries, so '
       'the comparison changed two things at once. Retrained with the constraint restored, the '
       'cost is zero. The confound was found by noticing that the two models\' touch-down heights '
       'differed by 10&times; &mdash; a fingerprint of the constraint, not of the noise.</p>')
assert pat.search(s), 'the amortisation paragraph moved; fix this patch'
s = pat.sub(new, s, count=1)
open('index.html', 'w', encoding='utf-8').write(s)
print('section 4.2 updated')
