# samedata ma_jmi2025: summary

Config: experiments/configs/samedata_ma_jmi2025.json (+ samedata_common.json). Code: data/v2/train_samedata.py. Bar: R2 0.919 (test set of one random 4:1 split, iteration 1, Ridge(alpha=0.1)). Built with PriorLabs-TabPFN (c2, c3).

Reproduction gate: r0 P1 mean R2 0.9407, interval [0.889, 0.949]: **PASS** (no exact split exists). r0 LOOCV R2 on the training parts: mean 0.9317 (their 0.945, context only).


## P1

| cand | R2 mean ± SD (ddof 0) | R2 repeat 0 | RMSE mean | MAE mean | wins vs r0 |
|---|---|---|---|---|---|
| r0 | 0.9407 ± 0.0233 | 0.9524 | 1.879 | 1.409 | - |
| c1 | 0.9548 ± 0.0192 | 0.9585 | 1.628 | 1.183 | 9/10 |
| c2 (BLOCKED, not selectable) | 0.9499 ± 0.0234 | 0.9630 | 1.709 | 1.145 | 8/10 |
| c3 (BLOCKED, not selectable) | 0.9552 ± 0.0191 | 0.9624 | 1.623 | 1.138 | 9/10 |

Selected c* (P1, highest mean R2 among c1): **c1**. c* R2 > r0 R2 in 9/10 paired repeats (same splits); mean paired difference +0.0141.

Sensitivity, all 227 rows (no exclusion; never selected on, never the claim): r0 0.9187 ± 0.0431; c1 0.9542 ± 0.0163; c2 blocked; c3 blocked

Allowed sentence (P1, template verbatim):

> On Ma and co-authors' data and their protocol (10 random splits instead of their one), Dravix's c1 reaches mean R2 0.955 (repeat 0: 0.959) vs their published 0.919 (their single split); their own model re-run on the same 10 splits gives 0.941.

All three candidates and r0 are printed (multiplicity: c* is the best of 3 by mean; winner's curse applies). No p-values. Wording beyond the template is the owner's decision.

Runtime: 67s wall.
