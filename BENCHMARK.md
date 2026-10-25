# Dravix cone-calorimeter literature benchmark: protocol (v1)

This file defines how to score a model on `cone_lit_benchmark_v1.csv`, which numbers count as evidence, and what "comparable" means. The same file ships in the software and dataset records. Numbers below are from the run logs shipped with software release 2.0.0 (`experiments/runs/litcone_gt_round1_FINAL.log`, `..._GTPROTO2.log`, `litcone_ustc_style.log`, `samedata_*`, `experiments/challenger/`, `wave2_exam_round1.*`).

Paths are relative to the software record. In the dataset record the main logs are in `results/`, the config and critic are in `protocol/`, and the other cited files are in the software record only. `splits_v1.csv` and `make_splits.py` are in the dataset record only.

## 1 - Tasks

Predict four cone-calorimeter outputs from a formulation and its test condition:

| target | column | rows | papers |
|---|---|---|---|
| TTI | `tti_s` | 3,368 | 598 |
| pHRR | `phrr_kWm2` | 4,017 | 725 |
| THR | `thr_MJm2` | 3,355 | 619 |
| average HRR | `avg_hrr_kWm2` | 759 | 130 |

**Row rule:** for each target, keep rows with target > 0 and `heat_flux_kWm2` present, **in file order** (the folds depend on order). Allowed inputs: anything in the row except the four targets and the paper's other outputs for that row (e.g. do not feed pHRR when predicting THR). `group`, `doi`, `pmcid`, `row_id`, `evidence` and `notes` are **not** inputs. If your model uses a paper identifier or the paper's measured control value, say so; such results are reported separately (section 4).

## 2 - Protocols

| id | name | splits | what it measures |
|---|---|---|---|
| **P3** | **paper-grouped CV (headline)** | `GroupKFold(n_splits=5, shuffle=True, random_state=r)`, r = 0..9, groups = `group` | prediction for papers (labs, resins, formulations) the model has never seen |
| P2 | random CV | `KFold(5, shuffle=True, random_state=r)`, r = 0..9 | interpolation; rows of one paper sit on both sides, so it is optimistic |
| P1 | training-data fit | 5-fold ensemble scored on the training rows, plus in-sample fit | **not held-out evidence.** Only used to match studies that published this protocol |

The exact folds are in `splits_v1.csv` (columns `target, protocol, repeat, row_id, group, fold`), written by `make_splits.py` with **scikit-learn 1.7.1**. Use the file instead of re-deriving the folds, because other scikit-learn versions shuffle differently.

## 3 - Metrics

- **R²**, in raw target units (a model may fit log1p internally but must back-transform), **pooled over all out-of-fold predictions of one repeat**. Report the mean ± SD over the 10 repeats (SD is NumPy's default, ddof = 0).
- **95% CI**: 2,000 bootstrap resamples (NumPy `default_rng(0)`) of the repeat-0 out-of-fold predictions. P3 resamples **papers** (all rows of a drawn paper); P2 resamples rows. The CI is for repeat 0 only, so it need not contain the 10-repeat mean (e.g. primary TTI P3).
- Also report **MAE** (repeat 0) and the number of rows and papers scored.
- A model "improves" on another only if it wins on P3 in at least 8 of 10 **paired** repeats (same folds) and its CI lower bound is not lower than the other model's. That is the project's adoption gate.

## 4 - Reference results (v1)

**Shipped engine model** (`primary`: histogram gradient boosting (HGB) + ExtraTrees + Ridge blend, log1p target, v3 features (see `engine/dravix_engine.py`), clipped to the training range):

| target | P3 paper-grouped, mean ± SD [CI rep 0] | P2 random | P1 train-fit (5-fold ens.) |
|---|---|---|---|
| TTI | **-0.247 ± 0.117** [-0.153, 0.250] | 0.376 ± 0.042 | 0.555 |
| pHRR | **0.286 ± 0.058** [0.188, 0.442] | 0.604 ± 0.003 | 0.697 |
| THR | **0.244 ± 0.055** [-0.090, 0.334] | 0.683 ± 0.010 | 0.780 |
| avg HRR | **0.234 ± 0.042** [0.051, 0.324] | 0.677 ± 0.017 | 0.828 |

**`gt_protocol_et2`** (ExtraTrees, min_samples_leaf 1, **plus a paper-identity code and the paper's measured control value as inputs**, identical-input rows averaged first, so TTI 2,750 / pHRR 3,312 / THR 2,772 / avg HRR 626 rows). It was built only to match the Tran et al. protocol:

| target | P1 train-fit (5-fold ens.) | in-sample | P2 random | P3 paper-grouped* |
|---|---|---|---|---|
| TTI | 0.983 | 1.000 | 0.575 [0.452, 0.688] | 0.261 [0.159, 0.406] |
| pHRR | 0.985 | 1.000 | 0.641 [0.580, 0.690] | 0.464 [0.397, 0.544] |
| THR | 0.990 | 1.000 | 0.758 [0.717, 0.798] | 0.599 [0.513, 0.663] |
| avg HRR | 0.987 | 1.000 | 0.674 [0.552, 0.782] | 0.357 [0.184, 0.541] |

\* With a paper code as an input, P3 is "not meaningful" (amendment 9 in `experiments/configs/litcone_gt_round1.json`): the code is unseen at test time. It is listed for completeness. Do **not** compare it with the primary P3.

Withdrawn: the `variant_C` "paper control" numbers in scoreboard entries 2-3 (2026-10-17) leaked the control rows (critic: `experiments/critic/litcone_gt_round1_critic.md`). Do not quote them.

## 5 - What "comparable" means

A Dravix number is **comparable** with a published number only if all of these hold. Comparability is a human decision, recorded per result; none is recorded as comparable today.
1. same target and units;
2. same population (same rows, or a population the authors agree is equivalent);
3. same or harder protocol (held-out difficulty no easier; no leakage);
4. the competitor's own model reproduced on that protocol first (a "gate");
5. repeated splits with a CI, not one split, and the CI lower bound clears the bar.

Results that fail any item are reported as **"reaches X under protocol Y vs their published Z"**. They are never reported as a win.

## 6 - Status of published comparisons (none is "comparable")

Every row below fails at least one item of section 5 (usually the population). Read each as "reaches X under protocol Y vs their published Z", never as a win.

| study | their number | Dravix | reading | files |
|---|---|---|---|---|
| Tran et al. 2025 (Georgia Tech / Matmerize), doi 10.1039/D4PY01417K | R² 0.95 / 0.94 / 0.97 / 0.96 (TTI/pHRR/THR/avg) **on their training data**; rows not released | `gt_protocol_et2` P1 0.983 / 0.985 / 0.990 / 0.987 | training-fit match only, on a different and larger population. In-sample 1.000 means memorisation; held-out numbers (P2/P3, section 4) are much lower. **No claim of better prediction.** | `experiments/runs/litcone_gt_round1_GTPROTO2.log` |
| Li et al. 2025 (USTC), Polym. Degrad. Stab. 234, 111209: pHRR of organophosphorus-FR epoxy | MAE 125.5 kW/m² (abstract), test split grouped by FR molecule | challenger A3 on the epoxy + P-FR rows of this dataset, GroupKFold by FR identity, with the paper's neat-epoxy pHRR as an input: seeds 0-9 MAE **127.8 ± 3.9 [113.5, 149.9]**, R² 0.774. All 26 A3 seeds: mean 128.0, best 121.2, 8 of 26 below 125.5 | reaches MAE 128.0 (mean of 26 seeds) vs their published 125.5; 8 of 26 seeds fall below 125.5 and the best is 121.2 (a best-of-26 selection, not a typical result). Different dataset; an input (the neat control) that USTC did not use; stricter diagnostics are worse (A4 141.7 with test papers also held out; A6 155.2 on the closest-scope population) | `experiments/challenger/` (`RESULTS_BEST_OF_N.md`, `LOG.md`, `runs/`) |
| Pomázi, Magyar & Toldy 2026 (BME Budapest), Polym. Degrad. Stab. 245, 111857 | held-out (random 80/20 + 4 validation): pHRR 0.30, TTI 0.58, THR 0.67, char 0.82; full-data fits 0.88-0.96 | **not run on their data** (not released). On a reconstruction from the paper (72 rows for pHRR, 60 for the other targets; reconstruction not shipped), mean of 50 random 80/20 splits: pHRR 0.604 [0.557, 0.650], THR 0.828 [0.791, 0.861], TTI 0.615, char residue 0.807 | ahead of the published pHRR and THR numbers on the reconstruction (CI excludes them); TTI and char residue within noise. Same random-split protocol, different population (a reconstruction, not their data): not comparable | `experiments/results/bme_*`, `experiments/runs/bme_*` |
| Chen et al. 2023, Chem. Eng. J. 455, 140547 (LOI, same data) | R² 0.642 (one random split) | best of 3 declared candidates (TabPFN v2) 0.690 ± 0.042 on 10 random splits; their model re-run 0.567 | random split only. Under paper-grouped CV every model is near 0.14 (Dravix 0.143, theirs 0.133) | `experiments/configs/samedata_chen_cej2023.json`, `experiments/runs/samedata_chen_*` |
| Ma et al. 2025, J. Mater. Inform. 5 (LOI, same data) | R² 0.919 (one random split) | 0.955 ± 0.019 on 10 splits; their model re-run 0.941 (9/10 paired wins) | single-lab data; most of the gap to the published number is split luck | `experiments/configs/samedata_ma_jmi2025.json`, `experiments/runs/samedata_ma_*` |
| DOPO-EP-ML (Huang et al., open dataset; paper unpublished) | LOI R² 0.709; UL-94 macro-F1 0.817; pHRR R² 0.604; THR R² 0.483 | LOI 0.685 (re-run 0.705); UL-94 0.783 (re-run 0.805); pHRR 0.661 (10/10 paired wins vs their re-run); THR 0.440 (re-run 0.391, 9/10) | **behind** on LOI and UL-94; pHRR 0.661 vs their published 0.604 (molecule-grouped protocol); THR 0.440 is below their published 0.483 but above their re-run on 10 fold sets (0.391). Their data, not Dravix's | `experiments/configs/samedata_huang_dopo_*`, `experiments/runs/samedata_huang_dopo_*` |

"Dravix" in the Chen, Ma and DOPO-EP-ML rows is the best of the pre-declared candidate models (c1-c3) chosen by mean score, on the other group's data; it is not always the shipped engine, and choosing the best of several inflates the result slightly.

## 7 - Wave-2 exam (one-shot test on later data)

After the engine was built, a second extraction wave collected 773 rows from 133 papers; after dropping textile and coated-substrate rows, 655 rows from 116 papers were scored (none in this dataset, checked by DOI, PubMed ID and formulation). The engine was scored once, with the script and config committed before scoring. R² in raw units, 95% CI from a 2,000-resample paper bootstrap:

| target | rows / papers | engine R² [CI] | matrix-median baseline R² | Spearman |
|---|---|---|---|---|
| TTI | 329 / 49 | 0.186 [0.113, 0.636] | 0.055 | 0.739 |
| pHRR | 514 / 89 | 0.509 [0.417, 0.595] | 0.273 | 0.794 |
| THR | 528 / 90 | 0.486 [0.342, 0.603] | 0.125 | 0.743 |
| avg HRR | 209 / 25 | 0.642 [0.426, 0.716] | 0.423 | 0.827 |

One draw from one extraction wave with small paper counts and wide CIs; it is a different population from P3, so it is not proof of better generalisation. The paper's own neat-polymer value predicts TTI (0.436) and THR (0.593) better than the engine does. Wave-2 rows are not in this release. Files: `experiments/configs/wave2_exam_round1.json`, `experiments/runs/wave2_exam_round1.md`, `experiments/results/wave2_exam_round1.json`.

## 8 - What we do not claim

- We do not claim any result is better than a published model. No result is marked comparable.
- We do not claim that the engine predicts new materials well. Paper-grouped R² (P3) is 0.23-0.29 for pHRR, THR and average HRR, and **negative for TTI**, i.e. worse than predicting the mean.
- We do not claim training-fit R² near 1 shows skill.
- We do not claim the dataset is error-free. 0.22% is measured on a 20% sample; the rest is unchecked.
- We do not claim results transfer across dataset versions. New rows mean new folds.
- The project's "48/48 reproduced" check covers the FSRI and LOI records, not this benchmark.

## 9 - How to report a result

Report the dataset version DOI, the software version DOI, the scikit-learn version, the protocol id, the 10 per-repeat R² values, the mean ± SD, the CI (with resampling unit), the MAE, and the rows/papers scored. Also report whether any paper-identity or control-value input was used.
