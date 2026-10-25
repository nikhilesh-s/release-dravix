# CFRS weights: pre-declared analyses

Declared 2026-10-22, committed before any weight was scored. Changing a data set, metric or rule
after results are seen requires a new dated amendment below, not an edit.

Score under test: `engine/score/cfrps.py`, 100 x sum w_k * clip(norm_k, 0, 1), renormalised over
present inputs. Six inputs, fixed order: TTI, pHRR, THR, avg_HRR, char_residue, LOI.
Incumbent `v0` (placeholders, `plans/FABLE_PLAN_2026-10-01.md:209`): TTI .20, pHRR .30, THR .10,
avg_HRR .20, char .10, LOI .10. Baseline `eq`: 1/6 each.
Shared code: `common.py` (loaders, UL-94 parser, vectorised scorer identical to `cfrps()`).

## Data

- **D1 lit-UL94.** `data/v2/cone_lit_benchmark_v1.csv` (snapshot used for the reference ranges),
  rows with a parseable UL-94 class (`common.ul94_class`: V-0/5V = 3, V-1 = 2, V-2 = 1,
  HB/NR/fail = 0; first-listed thickness; worst per-specimen result; VTM/HF and other tests
  excluded). Pairs: two rows of the **same paper** (`group`) **and same heat flux**, different
  UL-94 class, at least one shared input present in both. Correct = the better-rated row gets
  the lower CFRS. Within-paper pairing removes lab, thickness and flux offsets.
- **D2 lit-LOI.** Same snapshot, same pairing rule, label = LOI, pairs with |dLOI| >= 3 vol%.
  LOI is **removed from the inputs** for D2 (otherwise circular). Secondary to D1.
- **D3 FSRI.** `data/v2/cone_replicates_fsri.csv`, 50 kW/m2, per-material replicate means
  (TTI, pHRR, THR, residue; no avg_HRR, no LOI). No outcome labels; robustness only.
- **D4 external checks (descriptive, no fitting).** FAA tables in `data/v2/loi_faa_*.csv`
  (UL-94 + LOI + Lyon heat release capacity; OSU pHRR/THR for the 9 AR-01/53 epoxies), and
  published indices computed per paper and flux against the paper's neat control:
  Vahabi FRI = [THR x pHRR/TTI]_neat / [THR x pHRR/TTI]_composite; Petrella FPI = TTI/pHRR.
  FRI and FPI are functions of the same inputs, so agreement is a consistency check, not
  independent validation. MIL-STD-2031 pass/fail needs 75-100 kW/m2 and smoke data the repo
  does not have; reported as unavailable unless an open source with per-material values is found.

## Metric

- **C (within-paper concordance):** share of D1 (or D2) pairs ordered correctly by CFRS; ties = 0.5.
  Reported pooled over pairs, plus per-paper-averaged.
- **Kendall tau-b** between two rankings of the same rows; **top-k overlap** (k = 10 for D3,
  top 10% for lit) = |top_k(a) & top_k(b)| / k.

## A1 Outcome calibration (D1 primary, D2 secondary)

- Fit w on the simplex (softmax parametrisation, scipy) minimising mean pairwise logistic loss
  log(1 + exp(-beta * (CFRS_worse - CFRS_better) / 100)) with free beta > 0, plus
  lambda * ||w - w_prior||^2 with w_prior = eq and lambda = 0.1 (fixed; the AHP weights replace
  w_prior when they arrive, same lambda).
- CV: GroupKFold(5, shuffle=True, random_state=r) by paper, r = 0..9; fit on training papers'
  pairs, score test papers' pairs; pooled out-of-fold C per repeat. Compare fitted vs `v0` vs `eq`
  on the same folds.
- **Adoption rule:** fitted weights (fit on all of D1) become v1 only if their OOF C beats `v0` in
  >= 8/10 repeats **and** the paper-bootstrap (2000 resamples of papers) 95% CI of
  C(fitted) - C(v0) on repeat-0 OOF scores has lower bound > 0. Otherwise v1 = whichever of `v0`
  / `eq` has the higher mean OOF C, and the fit only sets the uncertainty band.
- **How well the data constrains each weight:** 500 paper-bootstrap refits -> 5th-95th percentile
  per weight. "Constrained" = 90% interval width < 0.15; "weak" = 0.15-0.30; "unconstrained" > 0.30.
  Plus drop-one: OOF C with that weight forced to 0 (others renormalised).

## A2 Robustness (D1 rows with >= 4 inputs, at 50 kW/m2 and at 35 kW/m2 separately; D3)

- R1 one-at-a-time: each weight x 0.5 and x 1.5, renormalised to sum 1; Kendall tau vs base and
  top-k overlap; also change in D1 C.
- R2 Dirichlet: 2000 draws w ~ Dir(50 * w_base); median and 5th percentile tau and top-k overlap.
- R3 normalisation: fixed 5-95 percentile ranges (current) vs batch-relative min-max within the
  ranked set vs physical reference ranges (documented with sources in the robustness script);
  tau between rankings and D1 C under each.
- R4 missing inputs: on rows with all six inputs, mask (char, LOI), then (avg_HRR), and compare
  coverage renormalisation (current) vs imputing subscore 0.5 (old DFRS rule) vs training median;
  tau with the full-input ranking and D1 C.
- **Stability threshold (tested in `test_weights.py`):** for the recommended v1 weights, R2 median
  tau >= 0.90 and 5th-percentile tau >= 0.80 on the lit 50 kW/m2 set and on D3.
  A weight "matters" if its R1 x0.5 or x1.5 move drops tau below 0.95; otherwise it "barely moves"
  rankings.

## A3 Expert elicitation (AHP; no data, design only)

- Criteria: TTI, pHRR, THR, avg HRR, LOI, char residue, smoke (smoke is a gap: its weight is
  recorded, and the six others are renormalised until the engine predicts smoke).
- Saaty 1-9 scale, 21 pairwise questions. Priority vector = principal eigenvector.
  CR = CI / RI, CI = (lambda_max - n) / (n - 1), Saaty RI (n = 7: 1.32; n = 6: 1.24).
  CR < 0.10 accepted; otherwise the expert is asked to revisit the most inconsistent triads.
- Two experts aggregated by element-wise geometric mean of their judgment matrices (AIJ), then
  eigenvector; individual vectors and CRs also reported.
- Use: AHP weights become `w_prior` in A1 (amendment-free; declared here).

## Amendments

**A1-1 (2026-10-22, written after the first A1 run; results in `results/calibration.json`).**
The unfloored D1 fit set char_residue to 0.000 (bootstrap 5th-95th: 0.00-0.19). A zero weight
makes the CFRS ignore a descriptor the patent assigns to the thermal-stability index (spec
[00271]) and breaks the pre-existing strict-monotonicity test in `test_cfrps.py`. Change: every
weight >= 0.05 (w = 0.05 + 0.70 * softmax(t)). The floored fit gets the same CV, paper-bootstrap
CI and adoption rule against `v0`. If it passes, it is v1. If it fails, v1 falls back to the
declared rule (better of `v0` / `eq` by mean OOF C); the unfloored fit is reported, not adopted.
Output: `results/calibration_floor05.json`.
