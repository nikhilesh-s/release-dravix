# samedata chen_cej2023: summary

Config: experiments/configs/samedata_chen_cej2023.json (+ samedata_common.json). Code: data/v2/train_samedata.py. Bar: R2 0.642 (single random hold-out test set, 263 rows (train 790)). Built with PriorLabs-TabPFN (c2, c3).

Reproduction gate: r0 on P1_pub (train 789 / test 264) R2 0.6420 vs published 0.642, interval [0.637, 0.647]: **PASS**.


## P1_pub

| cand | R2 mean ± SD (ddof 0) | R2 repeat 0 | RMSE mean | MAE mean | wins vs r0 |
|---|---|---|---|---|---|
| r0 | 0.6420 ± 0.0000 | 0.6420 | 2.459 | 1.930 | - |
| c1 | 0.5947 ± 0.0000 | 0.5947 | 2.616 | 2.015 | - |
| c2 | 0.7447 ± 0.0000 | 0.7447 | 2.077 | 1.492 | - |
| c3 | 0.7057 ± 0.0000 | 0.7057 | 2.230 | 1.668 | - |

P1_pub is one split (= P1 seed 1); c1-c3 here are information only.

## P1

| cand | R2 mean ± SD (ddof 0) | R2 repeat 0 | RMSE mean | MAE mean | wins vs r0 |
|---|---|---|---|---|---|
| r0 | 0.5671 ± 0.0473 | 0.5809 | 2.728 | 1.998 | - |
| c1 | 0.5681 ± 0.0404 | 0.5757 | 2.727 | 2.000 | 2/10 |
| c2 | 0.6897 ± 0.0422 | 0.6865 | 2.309 | 1.515 | 10/10 |
| c3 | 0.6554 ± 0.0378 | 0.6537 | 2.435 | 1.685 | 10/10 |

Selected c* (P1, highest mean R2 among c1, c2, c3): **c2**. c* R2 > r0 R2 in 10/10 paired repeats (same splits); mean paired difference +0.1226.

Allowed sentence (P1, template verbatim):

> On Chen and co-authors' data and their protocol (10 random splits instead of their one), Dravix's c2 reaches mean R2 0.690 (repeat 0: 0.687) vs their published 0.642 (their single split); their own model re-run on the same 10 splits gives 0.567.

## P2

| cand | R2 mean ± SD (ddof 0) | R2 repeat 0 | RMSE mean | MAE mean | wins vs r0 | CI rep-0 |
|---|---|---|---|---|---|---|
| r0 | 0.1332 ± 0.0466 | 0.1083 | 3.806 | 2.920 | - | [-0.018, 0.219] |
| c1 | 0.1016 ± 0.0367 | 0.1234 | 3.876 | 2.985 | 3/10 | [0.002, 0.224] |
| c2 | 0.0766 ± 0.0251 | 0.0627 | 3.929 | 3.012 | 1/10 | [-0.084, 0.183] |
| c3 | 0.1433 ± 0.0213 | 0.1437 | 3.785 | 2.908 | 6/10 | [0.019, 0.244] |

Selected c* (P2, highest mean R2 among c1, c2, c3): **c3**. c* R2 > r0 R2 in 6/10 paired repeats (same splits); mean paired difference +0.0101.

Allowed sentence (P2, template verbatim):

> Under Dravix's paper-grouped 5x10 protocol on the same rows: c3 0.143 [0.019, 0.244], their model re-run 0.133.

All three candidates and r0 are printed (multiplicity: c* is the best of 3 by mean; winner's curse applies). No p-values. Wording beyond the template is the owner's decision.

Runtime: 24279s wall.
