# Run Check (RC) of the A3 headline

Verdict: **A3 APPROVE WITH CAVEATS.** No target leakage found. The result is "comparable to USTC", not "beats USTC".

## 1. Code audit (run_attempt.py, train_litcone_gt_round1.py)
- Features hold no row-own target. `control_feature` blanks control rows (no variant_C self-leak). `ctrl_*` and the divisor come from the same-paper neat control, a legitimate input. Controls are never in the pool or test.
- Imputer and scaler sit inside the per-fold pipeline. Log-ratio clip bounds come from training rows only. The median baseline uses training rows only.
- Training drops every row whose first-FR identity is in the test fold. The asserts hold (no shared identity, no shared rows).
- Bootstrap resamples FR groups on the first seed's OOF. Correct, but the CI is single-seed.
- Synonym and skeleton merging uses names and SMILES only, with no targets. The merge is conservative (more merging, stricter split).
- **Paper fingerprint.** The `ctrl_*` vectors are effectively a paper ID: 644 distinct vectors for 647 training papers. 101 of 183 test rows (seed 0) have a same-paper row in training under a different FR. An FR-grouped protocol allows this, and USTC's neat-matrix inputs likewise fingerprint the matrix. It matters as a caveat, not a leak. A4 (test papers also held out) gives 141.7, so about 14 MAE of A3 comes from paper-mates.

## 2. Non-first-FR leak (A3 seed-0 folds)
- 97 of 183 test rows share at least one FR/synergist identity with a training row. Without the alias map the count is 84. The overlap is mostly APP (57 training rows), AlPi (23), DOPO and ATH as second components.
- That is material, so **RC1** (A3 plus drop any shared identity, `exclude_any_fr`, default off) was run: **MAE 134.1 ± 4.3, CI [120.5, 161.5], R2 0.747**. Cost: +6.3 against A3. Training shrinks by about 15%.

## 3. Fresh seeds
| Run | MAE | CI | R2 |
|---|---|---|---|
| A3 seeds 0-9 | 127.8 ± 3.9 | [113.5, 149.9] | 0.774 |
| A3 seeds 10-19 | 128.0 ± 3.7 | [115.4, 148.2] | 0.774 |

There is no sign of selection overfit.

## 4. Other attempts
- A0s (Arnav with strict groups): 170.3, against 171.8 with his groups. The synonym leak was only about 1.5 MAE, so his result stands as a reproduction.
- A5 (no `ctrlbase`): 133.1.
- A6 (v2 population, n=168): 155.2 [129.9, 170.8]. This is the nearest population to USTC and does not match 125.5.
- A1 129.2 and A2 126.2 use Arnav's 45 first-FR groups, which have a synonym leak. They are not headline candidates.

## Footnotes required
1. Split by first-FR identity with 37 strict groups (Arnav's has 45). Same-paper rows with other FRs stay in training.
2. Paper-held-out A4 gives 141.7 and any-FR-held-out RC1 gives 134.1. Report these beside 127.8.
3. Training pool is all matrices' non-control rows that have a control, not the B' population alone.
4. The CI is a first-seed group bootstrap and overlaps USTC's 125.5. The USTC figure is unconfirmed and uses a different dataset, so do not claim a win.
5. The 195 control-times-median-ratio baseline is the floor.
