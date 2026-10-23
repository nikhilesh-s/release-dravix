# Critic: LOI round 7 (delta-from-control, supplementary)

VERDICT: SOUND WITH CAVEATS

Scope: config experiments/configs/loi_round7.json, code data/v2/train_loi_round7.py, log experiments/runs/train_loi_round7.log, results exp100/exp101. Independent recompute of the control matching from loi_rpuf_literature.csv + rows_w*.csv.

## (1) Implementation vs config
- Copy check: lines 1-609 differ from train_loi_round6.py only in docstring, a GroupKFold import, and the removed round-6 fit/summary block. exp092 repro passes (B 0.3738, CI [0.2007, 0.4701]). Faithful.
- Control matching follows the config: rows with total_fr_wtpct == 0 (NaN coded -1, so a missing loading never counts as a control) are controls. Each row gets the mean of its paper's controls with the identical polyol string, else the paper mean. I recomputed it independently and got the same 313 rows / 63 papers, with 291 matched on polyol.
- Scored set = the non-control rows of control papers. Control rows are never scored and never used for training (confirmed). Features are c3 (47), as in round 6.
- GroupKFold(5, shuffle, seed r) by ref_id x10, with an assert that train and test papers are disjoint. CI = ep.bootstrap_ci on repeat-0 OOF, paper-level resampling, 2000 draws. Matches protocol B.
- Delta R2 for d1/d2 is scored as r2(ds, oof - cs), which is correct.

## (2) Leakage
- control_LOI is the declared input: a user who knows their own neat-foam LOI. It is legitimate for this task. A held-out paper's controls enter only through its own test rows' cs. Training rows' cs comes from training papers only. Features carry no paper-level target statistic. I found no path for held-out information to leak into training.

## (3) Baselines
- d0 = -1.83 is arithmetically correct: 1 - (sd_d^2 + mean_d^2)/sd_y^2 = 1 - (13.3 + 22.9)/12.8 = -1.83. But it is a straw man, since the control is biased by +4.8. A fairer naive baseline is control + mean delta. With an in-sample shift it gives R2 = -0.04, and corr(control, LOI) = 0.10. So the control on its own explains almost none of the row-to-row variance, and all the skill comes from the FR features.
- d1 vs ref is fair: same rows, same splits, same RF and features, 10/10 paired wins, +0.08 mean. The CIs overlap heavily ([0.10, 0.54] vs [0.05, 0.48]). Note that ref is handicapped: it is trained on 313 rows, not all available non-held-out rows.

## (4) Rows with total_fr_wtpct == 0 that are not clean neat controls (sampled all 91)
- N016 "industrial polyol Poliuretan Spray S-303HF" (LOI 23.8, UL-94 V-0, "no added FR stated") is a commercial system that almost certainly contains FR. It is coded as a control. N016's two FR rows ("biopolyol 100 pbw") fail the exact string match against "biopolyol (liquefied pine sawdust...) 100 pbw", so they fall back to the paper mean 21.85 instead of the true control 19.9. That affects 2 rows. Effect is small but it is a real data error.
- M141's control is EMF (a melamine-formaldehyde reactive polyol, LOI 24.8). It is an inherently FR polyol but common to all of that paper's rows, so it is acceptable as that paper's baseline.
- Reactive-FR-in-polyol papers (M001 PTOP/PTOSi, M055 P-polyol, M070, N022 ESBO-DYM, M049, M058) have FR rows whose polyol strings differ from the control's, so they correctly fall back to the paper control. Their FR rows are non-zero-coded and get scored. That is fine.
- M071 has 4 controls at different conditions (20.1-21.3), each paired with a 7 wt% row. Averaging them loses the pairing. Minor.
- M082 is all controls (10 rows) and contributes nothing. I found no case of a missing loading coded as 0.

## (5) Reporting
Reporting d1 as a supplementary "given-control" LOI number with d0 and ref beside it is sound, provided it is never put next to Sun & Fu 0.84 and never called a record. The result JSON's paired_vs is d0 (10/10 there is trivial). Quote the ref comparison instead.

## Top 3 caveats
1. d0 = -1.83 is a straw man (control + mean shift is about -0.04). Report d1 against ref (0.376 vs 0.296, 10/10, overlapping CIs), not against d0. The honest claim is that knowing the control adds about +0.08 R2.
2. N016's S-303HF row is a commercial (likely FR-containing, V-0) system coded as a neat control. Exact polyol-string matching also misses its biopolyol pairing, so 2 rows get a contaminated control (21.85 vs 19.9). Fix in a later data pass. It does not change the verdict.
3. The control is a coarse paper/polyol mean: within-paper pairing (M071), density and index differences are ignored, and the CI is wide ([0.10, 0.54], 63 papers). Report the CI, and the delta R2 0.40, alongside the point estimate.
