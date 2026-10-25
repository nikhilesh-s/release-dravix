# Wave-2 exam, round 1: results (run once, 2026-10-22)

The current engine (`engine/dravix_engine.py`) was trained on the pre-wave-2 snapshot (`cone_lit_benchmark_v1.csv`, 4,529 rows, sha256 `0b2af8e1…`) and scored once on wave 2. Wave 2 is held out: these rows were collected after the model was built.

- Config: `experiments/configs/wave2_exam_round1.json`. Script: `data/v2/wave2_exam_round1.py`. Both were committed in c3fa907 before scoring.
- Results: `experiments/results/wave2_exam_round1.json`. Log: `experiments/runs/wave2_exam_round1.log`.

## Population
773 wave-2 rows from 133 papers after dedupe. Dropping textile and coated rows leaves 655 rows from 116 papers. No paper also appears in the training data, by DOI, PubMed ID or (DOI, formulation), so 0 rows were dropped for overlap.

## Results (all exam rows; R² in raw units; 95% CI from a 2,000-resample paper bootstrap)

| Target | Rows / papers | Engine R² [CI] | Engine MAE | Spearman | Matrix-median R² | Engine minus median, R² [CI] | Earlier unseen-paper R² |
|---|---|---|---|---|---|---|---|
| TTI (sealed) | 329 / 49 | **0.186** [0.113, 0.636] | 24.0 s | 0.739 | 0.055 | +0.131 [0.083, 0.306] | -0.247 |
| THR (sealed) | 528 / 90 | **0.486** [0.342, 0.603] | 23.2 MJ/m² | 0.743 | 0.125 | +0.360 [0.095, 0.773] | 0.244 |
| avg HRR (sealed) | 209 / 25 | **0.642** [0.426, 0.716] | 56.7 kW/m² | 0.827 | 0.423 | +0.219 [-0.006, 0.387] | 0.234 |
| pHRR (engine unseen, not sealed) | 514 / 89 | **0.509** [0.417, 0.595] | 170.4 kW/m² | 0.794 | 0.273 | +0.236 [0.130, 0.340] | 0.286 |

"Earlier unseen-paper R²" is the P3 paper-grouped 5×10 mean on the training snapshot (`litcone_gt_round1_FINAL.log`, primary).

## Neat-control comparison (rows that are not controls and whose paper has a matching neat control)

| Target | Rows / papers | Engine R² | Same-paper neat control R² | Engine minus control, R² [CI] |
|---|---|---|---|---|
| TTI | 271 / 49 | 0.170 | 0.436 | -0.266 [-0.301, 0.093] |
| THR | 420 / 88 | 0.481 | 0.593 | -0.113 [-0.266, 0.098] |
| avg HRR | 179 / 25 | 0.632 | 0.184 | +0.448 [0.161, 1.445] |
| pHRR | 408 / 87 | 0.483 | -2.369 | +2.852 [0.754, 6.879] |

The neat-control baseline uses each paper's own measured control, which the engine never gets as input. For TTI and THR it is a "copy the paper's control" reference, and it beats the engine on point R². The CI on that gap includes 0.

## Caveats
- **Sample size.** avg HRR rests on 25 papers and TTI on 49, so their CIs are wide. The TTI point R² (0.186) sits near the bottom of its bootstrap CI because a few heavy-tailed papers dominate the raw-unit R².
- **Matrices not in training.** These rows fall into the engine's "other" matrix bucket: TTI 26, pHRR 65, THR 84, avg HRR 33. They were not removed (the config scores all exam rows).
- **Why wave 2 beats the earlier P3 numbers.** The two populations differ. Wave 2 is one open-access extraction wave with its own matrix mix, while P3 used cross-validation over 806 mixed-source papers. The wave-2 numbers are one draw, not proof that the engine generalises better than P3 suggested.
- **pHRR.** Wave 2 is no longer sealed for pHRR, because amendment-14 fits used it. The pHRR score is still valid held-out evidence for this engine only, since the engine never saw wave 2.
- **Training and data paths.** The engine was re-trained from the sha-checked snapshot into a temporary path, and per-target training counts matched FINAL (3368/4017/3355/759). The repo bundle was not used. Population processing used the snapshot's own `load_rows()` path.
- **Run once.** No re-runs under this config. Nothing goes on the site; wording is Nik's call. Never "beats SOTA"; `comparable` stays a human decision.
