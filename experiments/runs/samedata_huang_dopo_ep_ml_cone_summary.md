# samedata_huang_dopo_ep_ml_cone (DOPO-EP-ML pHRR + THR): results

Written 2026-10-22 by the DOPO Benchmark Engineer (branch `dopo-cone`). Config `experiments/configs/samedata_huang_dopo_ep_ml_cone.json` (commit 5a25f56) and splits (7116a96) were committed before any Dravix fit. Followed literally, no tuning. Source manuscript unpublished, not peer-reviewed. Built with PriorLabs-TabPFN.

## Bottom line

| Target | Their published (molecule-grouped nested 5x5) | Their re-run, published folds | Their re-run, 10 molecule-grouped repeats (P1_mol) | Dravix c*, P1_mol | Paired wins vs re-run | Dravix c*, paper-grouped P2 [repeat-0 CI] | Their re-run, P2 |
|---|---|---|---|---|---|---|---|
| pHRR R2 | 0.604 +- 0.042 | 0.607 +- 0.038 | 0.617 +- 0.017 | **c2 0.661 +- 0.012** | 10/10 | **c2 0.659 [0.555, 0.705]**, 10/10 | 0.612 [0.499, 0.669] |
| THR R2 | 0.483 +- 0.133 | 0.480 +- 0.142 | 0.391 +- 0.036 | **c4 0.440 +- 0.022** | 9/10 | **c2 0.444 [0.347, 0.551]**, 9/10 | 0.414 [0.283, 0.499] |

- pHRR: on the authors' data and molecule-grouped protocol, Dravix c2 (TabPFN v2) reaches mean R2 0.661 (repeat 0: 0.679) vs their published 0.604 (their single fold set); their own model re-run on the same 10 fold sets gives 0.617. MAE 154.9 vs re-run 161.9 kW/m2 (published 161.6). On their exact published folds c2 gives 0.659 +- 0.043.
- THR: Dravix c4 (ratio model) reaches mean R2 0.440 (repeat 0: 0.437) vs their published 0.483; their re-run on the same 10 fold sets gives 0.391. Their 0.483 sits on a favourable fold set: the same routine averages 0.391 over 10 other molecule-grouped fold sets. On their exact folds c4 gives 0.494 +- 0.103. MAE 12.55 vs re-run 13.28 MJ/m2 (published 12.43).
- Every model here, theirs included, gets the same paper's neat-epoxy value (EP_matrix_PHRR / EP_matrix_THR) as an input for modified rows. Their SelectKBest keeps it in every fold. So this is the same information as theirs, unlike the USTC amendment-13 comparison.

## FLAGS for the owner

1. **Reproduction gate failed as declared.** When run in their pinned env, `evaluate_task_nested` builds outer folds that differ from their published per-row folds. Those folds give pHRR 0.624 +- 0.101 and THR 0.369 +- 0.183, both outside the gate. Molecule groups match exactly (127 / 123 groups, none split across published folds), so the cause is GroupKFold tie order or row order. Diagnostic (the LOI FLAG 1 precedent): the same routine with the outer folds forced to the published ones gives **pHRR 0.607 and THR 0.480**, both inside the +-0.010 gate. The routine reproduces once the folds match. Owner decides whether the diagnostic satisfies the gate; until then, the sentences above are not cleared for use.
2. **Exit trigger hit for pHRR.** c* P1_mol mean 0.661 is above the published bar 0.604; P2 ci_lo 0.555 is not. THR: no trigger (0.440 < 0.483). Wording is the owner's call. Never "beats SOTA".

## All candidates (R2; MAE in kW/m2 or MJ/m2; modR2 = modified rows only)

pHRR (n = 406: 114 neat + 292 modified; 116 papers; 127 molecule groups; K = 330 to about 245 after their front end)

| model | P1_pub_mol | P1_mol mean +- sd (MAE, modR2) | wins vs re-run | P2 mean [CI] | wins |
|---|---|---|---|---|---|
| their re-run (r0_full) | 0.607 +- 0.038 (pub folds) | 0.617 +- 0.017 (161.9, 0.614) | - | 0.612 [0.499, 0.669] | - |
| r0 (their GBDT, no selection) | 0.627 | 0.636 +- 0.015 (158.7, 0.630) | 9/10 | 0.621 [0.526, 0.685] | 7/10 |
| c1 RF+ET+Ridge | 0.594 | 0.542 +- 0.125 (165.0, 0.495) | 1/10 | 0.586 [0.516, 0.668] | 3/10 |
| **c2 TabPFN v2** | 0.659 | **0.661 +- 0.012** (154.9, 0.661) | 10/10 | **0.659 [0.555, 0.705]** | 10/10 |
| c3 (c1+c2)/2 | 0.651 | 0.639 +- 0.031 (155.5, 0.638) | 9/10 | 0.646 [0.559, 0.700] | 10/10 |
| c4 ratio model | 0.605 | 0.610 +- 0.031 (156.9, 0.607) | 5/10 | 0.616 [0.424, 0.704] | 4/10 |
| b0 ctrl x median ratio | 0.553 | 0.563 +- 0.013 (178.6, 0.525) | 0/10 | 0.576 [0.504, 0.632] | 0/10 |

THR (n = 397: 111 neat + 286 modified; 113 papers; 123 molecule groups; K = 30)

| model | P1_pub_mol | P1_mol mean +- sd (MAE, modR2) | wins vs re-run | P2 mean [CI] | wins |
|---|---|---|---|---|---|
| their re-run (r0_full) | 0.480 +- 0.142 (pub folds) | 0.391 +- 0.036 (13.28, 0.442) | - | 0.414 [0.283, 0.499] | - |
| r0 (their GBDT) | 0.479 | 0.419 +- 0.037 (12.91, 0.495) | 8/10 | 0.427 [0.320, 0.545] | 7/10 |
| c1 | 0.491 | 0.419 +- 0.038 (12.85, 0.508) | 8/10 | 0.425 [0.343, 0.548] | 7/10 |
| c2 | 0.489 | 0.438 +- 0.031 (12.61, 0.531) | 10/10 | **0.444 [0.347, 0.551]** | 9/10 |
| c3 | 0.494 | 0.435 +- 0.030 (12.67, 0.532) | 10/10 | 0.440 [0.348, 0.554] | 10/10 |
| **c4** | 0.494 | **0.440 +- 0.022** (12.55, 0.556) | 9/10 | 0.442 [0.365, 0.548] | 9/10 |
| b0 | 0.438 | 0.384 +- 0.022 (13.67, 0.423) | 3/10 | 0.396 [0.288, 0.518] | 2/10 |

P1_pub_mol / P1_mol: mean of 5 per-fold R2 (their aggregation); P1_pub_mol SD over folds (ddof=1), P1_mol SD over 10 repeats (GroupKFold(5, shuffle, rs 0..9) on their molecule groups). P2: GroupKFold by Reference, pooled OOF R2, repeat-0 paper bootstrap CI (2000, seed 0). Wins = paired repeats with higher R2 than their re-run on identical folds. Their re-run = full nested model choice among 8 models in their env, with only the outer folds swapped.

## Caveats

- Winner's curse: c* is the best of 4 by mean. All four are printed above. THR c2/c3/c4 are within 0.005 of each other.
- 114/406 (pHRR) and 111/397 (THR) rows are neat epoxy, which raises R2. The modified-only R2 is printed and gives the same ordering.
- c1's Ridge member extrapolates on 4 PHRR folds (predictions from -1139 to 5605 kW/m2; P1_mol_r7 R2 0.175). Fixed a priori, so it was not patched.
- The P1_mol repeats are Dravix's 10 fold sets, not theirs. Their single published fold set is P1_pub_mol.
- THR inputs are only 30 features. Per-fold R2 is very noisy (their SD 0.13).

## Files

- Config: `experiments/configs/samedata_huang_dopo_ep_ml_cone.json`
- Code: `data/v2/export_dopo_cone.py` (authors' env), `data/v2/train_samedata_dopo_cone.py` (Dravix env)
- Splits: `experiments/runs/samedata_huang_dopo_ep_ml_cone_splits.json`
- Results JSON: `experiments/results/samedata/samedata_huang_dopo_ep_ml_cone.json` (subfolder, so `update_trackers.py ingest` skips it)
- Logs: `experiments/runs/samedata_huang_dopo_ep_ml_cone_*.log`
- Fold matrices, gates and re-runs: `experiments/runs/samedata_dopo_cone_folds/`; predictions in `experiments/runs/samedata_dopo_cone_preds/`
- Download (git-ignored): `data/v2/raw/dopo_ep_ml/` (code @ e550e4a, MIT LICENSE; Mendeley licence record CC BY 4.0; SOURCE.md)
