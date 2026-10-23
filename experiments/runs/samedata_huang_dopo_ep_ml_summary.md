# samedata_huang_dopo_ep_ml (LOI) + _ul94 (UL-94 V-0): results

Written 2026-10-08 09:09 PDT by data/v2/train_samedata_dopo.py report. Configs followed literally; no tuning. Source manuscript unpublished, not peer-reviewed. Built with PriorLabs-TabPFN.

## samedata_huang_dopo_ep_ml (LOI R2)

Reproduction gate (r0_full on P1_pub, authors' env): per-fold mean 0.7012 ± 0.0393 (gate [0.699, 0.719]) -> PASS; folds equal the published per-row folds: False; selected models ['XGB', 'GBDT', 'ExtraTrees', 'XGB', 'XGB']; published 0.709 ± 0.084. Diagnostic (same routine, outer folds forced to the published per-row folds): 0.7050 ± 0.0816 -> PASS, models ['XGB', 'GBDT', 'GBDT', 'GBDT', 'GBDT'] (see samedata_flags_dopo.md FLAG 1).

| candidate | P1_pub | P1 | P1_pub_mol | P1_mol | P2 | P3 |
|---|---|---|---|---|---|---|
| r0 | 0.711 ± 0.083 | 0.705 ± 0.017 (rep0: 0.728) | 0.722 ± 0.042 | 0.706 ± 0.015 (rep0: 0.691) | 0.712 ± 0.020 (rep0: 0.733) CI [0.670, 0.782] | R2 -0.208, MAE 2.470 (n=20) |
| c1 | 0.682 ± 0.073 | 0.669 ± 0.023 (rep0: 0.703); 1/10 > r0 | 0.715 ± 0.045 | 0.677 ± 0.022 (rep0: 0.682); 1/10 > r0 | 0.677 ± 0.021 (rep0: 0.711) CI [0.655, 0.759]; 1/10 > r0 | R2 -0.531, MAE 2.693 (n=20) |
| c2 | 0.662 ± 0.115 | 0.662 ± 0.019 (rep0: 0.675); 0/10 > r0 | 0.697 ± 0.050 | 0.675 ± 0.025 (rep0: 0.669); 0/10 > r0 | 0.670 ± 0.023 (rep0: 0.685) CI [0.610, 0.745]; 0/10 > r0 | R2 -0.304, MAE 2.248 (n=20) |
| c3 | 0.693 ± 0.091 | **0.685 ± 0.015 (rep0: 0.709); 1/10 > r0** | 0.726 ± 0.041 | **0.699 ± 0.013 (rep0: 0.697); 2/10 > r0** | **0.692 ± 0.016 (rep0: 0.718) CI [0.653, 0.768]; 1/10 > r0** | R2 -0.339, MAE 2.310 (n=20) |

P1_pub / P1_pub_mol: mean ± SD (ddof=1) over the 5 published folds, information only. P1 / P1_mol: mean of per-fold metric per repeat, mean ± SD (ddof=0) over 10 repeats (rep0 = repeat-0 value). P2: pooled OOF per repeat; CI = repeat-0 paper bootstrap (2000, seed 0). 'k/10 > r0' = paired repeats in which the candidate's value is higher than r0's on the same splits. Bold = c* (highest 10-repeat mean, within protocol).

Modified-only rows (Loading > 0), same folds, mean of per-fold values (reported, not selected on): r0 P1_pub 0.432, P1 0.414; c1 P1_pub 0.351, P1 0.313; c2 P1_pub 0.300, P1 0.304; c3 P1_pub 0.373, P1 0.353. Their modified-formulation subset: 0.419 ± 0.210.
Context (always printed): the overall 0.709 is lifted by the 131 neat-epoxy rows; Dravix's RPUF LOI benchmark has no such rows. P1_mol bar (molecule-grouped headline) 0.716 ± 0.036; P3 their -0.21 / 2.47.

Allowed sentences (templates from samedata_common.json, filled verbatim):
- P1: "On Huang, Zheng and Shi's data and their protocol (10 random splits instead of their one), Dravix's c3 reaches mean R2 0.685 (repeat 0: 0.709) vs their published 0.709 (their single split); their own model re-run on the same 10 splits gives 0.705." Paired: c3 > r0 in 1/10 repeats. All three: c1 0.669, c2 0.662, c3 0.685.
- P2: "Under Dravix's paper-grouped 5x10 protocol on the same rows: c3 0.692 [0.653, 0.768], their model re-run 0.712." Paired: c3 > r0 in 1/10 repeats. All three: c1 0.677, c2 0.670, c3 0.692.
- Exit trigger (c* P1 mean > bar, or P2 ci_lo > bar): no.

## samedata_huang_dopo_ep_ml_ul94 (UL-94 V-0 macro-F1)

Reproduction gate (r0_full on P1_pub, authors' env): per-fold mean 0.8119 ± 0.0458 (gate [0.807, 0.827]) -> PASS; folds equal the published per-row folds: True; selected models ['ExtraTrees', 'ExtraTrees', 'ExtraTrees', 'ExtraTrees', 'ExtraTrees']; published 0.817 ± 0.051.

Blocked: c2u: TabPFNValidationError: Number of features `667` reaching the model is greater than the maximum number of features `500` officially supported by the TabPFN model. Set `ignore_pretraining_limits=True` t; c3u: needs both members

| candidate | P1_pub | P1 | P2 |
|---|---|---|---|
| r0 | 0.809 ± 0.052 | 0.805 ± 0.007 (rep0: 0.797) | 0.804 ± 0.005 (rep0: 0.805) CI [0.776, 0.834] |
| c1u | 0.790 ± 0.043 | **0.783 ± 0.006 (rep0: 0.783); 0/10 > r0** | **0.782 ± 0.009 (rep0: 0.801) CI [0.769, 0.832]; 0/10 > r0** |
| c2u | — | — | — |
| c3u | — | — | — |

P1_pub / P1_pub_mol: mean ± SD (ddof=1) over the 5 published folds, information only. P1 / P1_mol: mean of per-fold metric per repeat, mean ± SD (ddof=0) over 10 repeats (rep0 = repeat-0 value). P2: pooled OOF per repeat; CI = repeat-0 paper bootstrap (2000, seed 0). 'k/10 > r0' = paired repeats in which the candidate's value is higher than r0's on the same splits. Bold = c* (highest 10-repeat mean, within protocol).

Modified-only rows (Loading > 0), same folds, mean of per-fold values (reported, not selected on): r0 P1_pub 0.745, P1 0.742; c1u P1_pub 0.733, P1 0.725. Their modified-only subset: 0.755 ± 0.067; molecule-grouped headline 0.779 ± 0.031.
Their tuned threshold (r0) is an advantage on F1 that c1u-c3u (argmax) do not get; the gap is printed, not adjusted.

Allowed sentences (templates from samedata_common.json, filled verbatim):
- P1: "On Huang, Zheng and Shi's data and their protocol (10 random splits instead of their one), Dravix's c1u reaches mean macro-F1 0.783 (repeat 0: 0.783) vs their published 0.817 (their single split); their own model re-run on the same 10 splits gives 0.805." Paired: c1u > r0 in 0/10 repeats. All three: c1u 0.783, c2u BLOCKED, c3u BLOCKED.
- P2: "Under Dravix's paper-grouped 5x10 protocol on the same rows: c1u 0.782 [0.769, 0.832], their model re-run 0.804." Paired: c1u > r0 in 0/10 repeats. All three: c1u 0.782, c2u BLOCKED, c3u BLOCKED.
- Exit trigger (c* P1 mean > bar, or P2 ci_lo > bar): no.

Multiplicity: c* is the best of 3 by mean (winner's curse, order of the between-candidate spread); all three and r0 are printed. No p-values, no significance language.
