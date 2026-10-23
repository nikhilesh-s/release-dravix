> DRAFT, NOT REFRESHED. Written against an older commit; update the numbers from the current numbers of record before upload.

# Dravix cone benchmark: protocol (v0.1)

This file defines how to score a model on `cone_lit_benchmark_v1.csv`, which numbers count as evidence, and what "comparable" means. The same file ships in the software and dataset records. Numbers below are from the run logs at `origin/master` @ b702117 (`experiments/runs/litcone_gt_round1_FINAL.log`, `..._GTPROTO2.log`, `litcone_ustc_style.log`, `samedata_*_summary.md`).

## 1 - Tasks

Predict four cone-calorimeter outputs from a formulation and its test condition:

| target | column | rows | papers |
|---|---|---|---|
| TTI | `tti_s` | 3,368 | 598 |
| pHRR | `phrr_kWm2` | 4,017 | 725 |
| THR | `thr_MJm2` | 3,355 | 619 |
| average HRR | `avg_hrr_kWm2` | 759 | 130 |

**Row rule:** for each target, keep rows with target > 0 and `heat_flux_kWm2` present, **in file order** (the folds depend on order). Allowed inputs: anything in the row except the four targets and the paper's other outputs for that row (e.g. do not feed pHRR when predicting THR). `group`, `doi`, `pmcid`, `row_id`, `evidence` and `notes` are **not** inputs. If your model uses a paper identifier or the paper's measured control value, say so; such results are reported separately (section 5).

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
- A model "improves" on another only if it wins on P3 in at least 8 of 10 **paired** repeats (same folds) and its CI lower bound is not lower. That is the repo's adoption gate.

## 4 - Reference results (v0.1)

**Shipped engine model** (`primary`: HGB + ExtraTrees + Ridge blend, log1p target, v3 features, clipped to the training range):

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

\* With a paper code as an input, P3 is "not meaningful" (amendment 9): the code is unseen at test time. It is listed for completeness. Do **not** compare it with the primary P3.

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

| study | their number | Dravix | reading |
|---|---|---|---|
| Tran et al. 2025 (Georgia Tech / Matmerize), doi 10.1039/D4PY01417K | R² 0.95 / 0.94 / 0.97 / 0.96 (TTI/pHRR/THR/avg) **on their training data**; rows not released | `gt_protocol_et2` P1 0.983 / 0.985 / 0.990 / 0.987 | only a training-fit match, on a different and larger population. In-sample 1.000 means memorisation. The held-out numbers (P2/P3 above) are much lower. **No claim of better prediction.** |
| Li et al. 2025 (USTC), OP-FR epoxy | pHRR MAE 125.5 kW/m² (UNCONFIRMED; full text not read) | USTC-style grouped-by-FR test on 643 epoxy rows: MAE 218.0 ± 9.7, R² 0.416 | **behind** (about 1.7x the error); bar unconfirmed |
| Pomázi et al. (BME) | held-out pHRR 0.30 etc.; full-data fits 0.95/0.96/0.88 | different data (75 samples) | not comparable; label any fit numbers as fits |
| Chen et al. 2023 CEJ (LOI, same data) | 0.642 (one split) | their model re-run 0.567 ± 0.047 on 10 splits; TabPFN (off-the-shelf) 0.690 | random split only, best of three models; all models collapse under paper-grouped splits (about 0.08-0.14). Wording is pending the owner |
| Ma et al. 2025 JMI (LOI, same data) | 0.919 (one split) | 0.955 (9/10 paired wins, +0.014 vs their re-run) | easy single-lab data; most of the gap is split luck |
| DOPO-EP-ML (Huang et al., unpublished) | LOI 0.709; UL-94 F1 0.817 | 0.685; 0.783 | **behind** on both |

## 7 - What we do not claim

- We do not claim to beat any state of the art. No result is marked comparable.
- We do not claim that the engine predicts new materials well. Paper-held-out R² is 0.23-0.29 for pHRR, THR and average HRR, and **negative for TTI**, i.e. worse than predicting the mean.
- We do not claim training-fit R² near 1 shows skill.
- We do not claim the dataset is error-free. 0.22% is measured on a 20% sample; the rest is unchecked.
- We do not claim results transfer across dataset versions. New rows mean new folds.
- We do not claim the "48/48 reproduced" check covers this benchmark. It covers the FSRI/LOI records. TODO: add a litcone reproduction run before release.

## 8 - How to report a result

Report the dataset version DOI, the software version DOI, the scikit-learn version, the protocol id, the 10 per-repeat R² values, the mean ± SD, the CI (with resampling unit), the MAE, and the rows/papers scored. Also report whether any paper-identity or control-value input was used.
