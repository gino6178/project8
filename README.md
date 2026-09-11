# TrackNet3D

A single forward pass that lifts a whole stroke of 2D shuttlecock detections to metric
3D, in place of a per-stroke constrained optimiser. The court calibration is an *input*,
not something the model estimates — the title says "lifting", not "calibration-free",
because the height head reads a nine-number camera summary and the horizontal solve reads
the homography.

**The paper is [`index.html`](index.html)** — open it in a browser, or read it at the
GitHub Pages URL once Pages is enabled for this repository.

This is **paper 2** and a separate contribution from paper 1 (SimTrack3D). It uses paper 1's
3D coordinates as the comparison target and does not restate paper 1's results.

## The one-line result

A model that never sees a pixel — given the whole stroke's 2D track and the venue's
calibration — reaches a median 3D error of **0.417 m** against paper 1's coordinates on a
held-out venue, where the end-to-end model reading five frames of pixels reaches **0.985 m**.
Refining with paper 1's own solver, accepted only where the solver's fit residual is small,
gives **0.351 m** and moves the fraction of frames within 5 cm from 0.1% to **6.9%**.

Temporal context at stroke scale is the lever. It saturates exactly there: extending past one
stroke is worse.

## Where the numbers come from

Experiments live in `~/paper3/experiments` and write logs to `~/paper3/out`. Nothing in this
repository re-runs them; this repository is the write-up.

| section | claim | script / log |
|---|---|---|
| §3.2 | main table | `exp51_train_e2e_geom.py --eval`, `exp64_stroke_pixels.py` |
| §3.3 | solver gate | `exp68_hybrid_solve.py` → `out/exp68_hybrid.csv` |
| §3.4 | detector cost, TrackNetV5 protocol | `exp57_tnv5_metric.py` → `out/exp57_tnv5_metric.csv` |
| §4.1 | reprojection and physics, and which of them is an identity | `exp70_nogt_suite.py` → `out/exp70_ours.csv`, `_solver.csv` |
| §4.2 | noise and held-frame stress sweep | `exp70_nogt_suite.py` → `out/exp70_stress.csv` |
| §4.2 | the same net trained with a noise model | `exp64 --noise-aug 8`, then `exp70 --ckpt out/p1corpus_naug.pt --no-solver` → `out/exp70n_stress.csv` |
| §4.3 | latency, throughput, frontier | `out/exp70_cost.csv` |
| §4.4 | 25 unseen venues, one plausibility test on both | `exp71_zeroshot.py` → `out/exp71_strokes.csv` |
| §5.1 | label ceiling, per corpus | `out/p2_label_selfres.npy` |
| §5.2 | label-2D vs detected-2D | `exp63_cache_feats.py`, `exp64 --det-uv` |
| §6.1 | physics, three ways | `exp59_physics_fit.py`, `exp61`, `exp62` |
| §6.2 | venue depth | `exp58_venue_depth.py`, `exp51 --depth`, `--depth-aux` |
| §6.3 | remaining ablations | `exp60`, `exp64 --win`, `--corpus`, `--prior` |
| figure | reconstruction GIF | `exp69_gif.py` → `out/exp69_recon.gif` |
| figures | stage strip and architecture | `make_stages.py`, `make_fig.py` (SVG + PDF + PNG) |
| figure | stress and frontier | `make_stress.py` (SVG + PDF + PNG) |

Two baselines are easy to measure wrongly and both were, once. Paper 1's solver
must be run from **its own** lifter prior — seeding it with our estimate measures
the hybrid of §3.3, not the baseline, and flatters it by nearly 2×. And the
plausibility rate of §4.4 must use the same test on both outputs; the solver's own
feasibility flag answers a different question.

Established facts and the traps we hit are recorded in `~/paper3/CLAUDE.md`.

## On the 5 cm figure people will compare against

MonoTrack's widely quoted ~5 cm is measured on **synthetic** trajectories and is the
saturation value for the longest flights; its synthetic average is 8.0 cm, and on real
footage it reports reprojection error only (up to 37.1 px) because real 3D ground truth
was not available. That is a different measurement from agreement with an independent
reconstruction of real broadcast video, which is what this paper reports.

## Two methodology notes worth carrying to other work

1. **Measure label quality per corpus before quoting a ceiling.** Ours differed by 8.7× between
   paper 1's own 3D and our regeneration of its pipeline, and measuring them together made a
   reachable target look arithmetically impossible.
2. **Evaluate with detected 2D, not label 2D.** Feeding the label track costs 3.2× less error
   than feeding the model's own detections. Reporting the first is overstating.
