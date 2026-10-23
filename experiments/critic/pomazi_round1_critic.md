# Critic: Pomázi reconstruction round 1 (exp060-063)
Fresh-context review 2026-10-01; scratch code `$SCRATCH/critic_pomazi/`. I re-ran protocol B for all 16 target x candidate cells. All logged numbers reproduce exactly. Comparability is a human decision, and this file makes no claim about it.
## Verdicts
| target | declared result (gbr_theirs) | verdict |
|---|---|---|
| TTI (bar 0.58) | B 0.683±0.172, ci_lo 0.652 (repeat 0); A 0.654 | **SOUND WITH CAVEATS**: the clearance is fragile (see 2, 3, 4) |
| THR (bar 0.67) | B 0.782±0.044, ci_lo 0.696; A 0.818 | **SOUND WITH CAVEATS**: robust to CI construction but mostly fibre/heat-flux separation |
| pHRR (bar 0.30) | B 0.621, ci_lo 0.198 (no B trigger); A 0.526 | **SOUND WITH CAVEATS**: whether B triggers depends on the CI construction and the grouping fix |
| CR (bar 0.82) | B 0.787, ci_lo 0.609 (no B trigger); A 0.850 | **SOUND WITH CAVEATS**: no B construction clears 0.82; A only |
## 1. Leakage
- (a) The formulation grouping has a bug: it does not do what the config says. Groups are built from the string of the 8 COMP inputs rounded to 0.1. Inputs derived by the mixture rule differ from the tabulated matrix rows by ≤0.01. Rounding at a .x5 boundary, or `-0.0` vs `0.0` (the APP vector carries -0.27 cycloaliphatic and -0.02 aromatic), then puts them in different groups. This splits 6 matrix-composite pairs: PZ0037|PZ0050, PZ0003|PZ0095/97/98/100, PZ0015|PZ0065, PZ0038|PZ0051, PZ0033|PZ0046 and PZ0010|PZ0059. With tolerance grouping (max abs diff <0.1, connected components) there are 40/32 groups, not 46/38.
  Re-running B with corrected groups (rep-0 ci_lo / pooled ci_lo): TTI gbr 0.773, 0.660/0.666; THR gbr 0.833, 0.732/0.753; CR gbr 0.808, 0.520/0.698. **pHRR: rf becomes the selected candidate (B 0.594) with ci_lo 0.306 > 0.30**, and ridge has rep-0 ci_lo 0.411. So the fix does not inflate TTI or THR, but it flips pHRR to a B trigger.
- (a) Other repeated physical samples. The 6 pds2020 matrix re-reports are correctly dropped as `dup_of`. These remain, each inside one group:
  PER/T58 neat matrix in 3 series (identical X; pHRR 944/962/706); PZ0097/98 (one panel, both faces); hemp trio PZ0145-147 (identical X).
  No material appears at both 25 and 50 kW/m². Heat flux 25 ⇔ SPE/GFTE exactly (23 of 63 rows, 30 of 75 for pHRR), so heat flux is a system identifier.
- (b) No target information is in the inputs. Composition comes from FR loading by mixture rule (leave-system-out error ≤0.02). Ratios come from N balance. LOI/UL94 are in the CSV but are not features. Hemp rows are encoded as PER base + 30 % fibre, so the hemp-fabric P treatment is invisible. That adds noise, not leakage. The notes tag them "probably exclude", yet they were included.
- (c) In-fold fitting is clean. The imputer and scaler sit inside pipelines that are cloned per fold. RF/HGB handle NaN natively. Groups come from inputs only. NaN ratio inputs: 11 rows (pHRR) and 5 rows (TTI/THR/CR).
- (d) fibre_wtpct_input is a near-identifier: 30 = hemp only, 40 = PZ0067 only, 42 = jtac11770 only, 67 = infused panels only. A GBR on **heat flux + fibre only** scores:
| | TTI | THR | CR | pHRR |
|---|---|---|---|---|
| B | 0.567 | **0.677** (point above bar) | 0.665 | 0.167 |
| A | 0.527 | 0.688 | 0.772 | 0.418 |
## 2. CI robustness (formulation bootstrap, 2000 resamples, seeds 0-4)
| target / cand | per-repeat R2 range | per-repeat ci_lo > bar (of 10, seeds 0-4) | median per-rep ci_lo | pooled-10-rep ci_lo (s0-4) | Bonferroni-4 ci_lo |
|---|---|---|---|---|---|
| TTI gbr | 0.294-0.827 | 6,7,7,6,6 | 0.639 (min -0.089) | 0.577-0.586 (straddles 0.58) | 0.551 |
| THR gbr | 0.701-0.857 | 5,6,6,6,6 | 0.687 (min 0.477) | 0.691-0.701 | 0.651 |
| pHRR gbr | 0.395-0.716 | 5 each | 0.275 | 0.292-0.313 | 0.201 |
| pHRR rf | 0.488-0.659 | 8 each | 0.358 | 0.367-0.388 | 0.292 |
| CR gbr | 0.664-0.848 | 0 each | 0.680 | 0.635-0.646 | 0.600 |
- TTI repeat 0 is not the luckiest repeat (R² rank 6/10, ci_lo 0.651-0.659 across seeds; median ci_lo is 0.639). However, TTI's per-repeat distribution is bimodal: repeats 1 and 8 give 0.29 and 0.43. Pooled over repeats, ci_lo sits on the bar.
- THR clears under every construction except Bonferroni.
- pHRR's "no B trigger" holds only under the declared repeat-0 rule. Under pooled CIs, rf and hgb clear 0.30.
## 3. Selection (4 candidates per target)
| target | rf B / ci_lo | gbr_theirs | hgb | ridge |
|---|---|---|---|---|
| pHRR | 0.593 / 0.214 | 0.621 / 0.198 | 0.580 / 0.218 | 0.460 / 0.285 |
| TTI | 0.623 / 0.577 | **0.683 / 0.652** | 0.612 / 0.566 | 0.410 / 0.109 |
| THR | 0.742 / 0.666 | **0.782 / 0.696** | 0.705 / 0.694 | 0.658 / 0.524 |
| CR | 0.704 / 0.379 | 0.787 / 0.609 | 0.702 / 0.423 | 0.763 / 0.602 |
- Selection inflation on the B mean is small:
  - E[max-of-4] minus E[gbr] over bootstrap resamples: TTI 0.008, THR 0.001.
  - gbr wins in 78 % (TTI) and 94 % (THR) of resamples.
  - Choosing on even repeats and scoring on odd repeats (or the reverse) picks gbr both times.
- On the CI, it matters. Only 1 of 4 TTI candidates clears (rf misses by 0.003). THR clears with 2 of 4 (gbr, hgb). Bonferroni-adjusted ci_lo is below the bar for both TTI (0.551) and THR (0.651).
- The trigger rule had 16 cells x 2 protocols of chances to fire.
- Their bars are also best-of-3 algorithms. Their XGBoost-only R² values were pHRR 0.25, TTI 0.39, THR 0.67 and CR 0.81. The TTI bar 0.58 is their ANN result.
## 4. Influence
- Leave-one-formulation-out (metric only, pooled): max |ΔR²| is 0.023 for TTI and 0.017 for THR, so no single formulation drives the result. pHRR is different: dropping the PER-base group (11 rows, including hemp) lowers it by 0.125.
- Leave-one-source-paper-out, with refit and tolerance groups (B mean, rep-0 ci_lo / pooled ci_lo):
| source dropped (n) | TTI | THR |
|---|---|---|
| thesis (18) | 0.809, 0.082 / 0.217 | not recorded |
| pz2023 (27) | 0.824, 0.724 / 0.747 | **0.494, 0.119 / 0.208** |
| jtac11770 (5) | pooled 0.475 | not recorded |
| polym2018 (8) | pooled 0.522 | not recorded |
| any one of the others | not recorded | pooled 0.625-0.729 |
- Within heat-flux strata (gbr OOF, pooled):
| target | 25 kW/m² | 50 kW/m² |
|---|---|---|
| TTI | 0.280 | 0.514 |
| THR | **-0.030** | 0.758 |
| pHRR | 0.593 | 0.563 |
| CR | 0.903 | 0.655 |
  Within matrix / composite, TTI is 0.498 / 0.762 and THR 0.649 / 0.667. Much of the TTI and THR R² comes from separating the 25 vs 50 kW/m² and matrix vs composite means.
- Excluding the 3 hemp rows raises TTI (0.789, ci_lo 0.741), THR (0.846 / 0.733) and CR (0.809 / 0.736). It lowers pHRR (0.466 / 0.200).
- Plan §6 defined protocol B as **GroupKFold by resin system**. The config redefined B as formulation grouping and demoted resin system to diagnostic C (leave-one-system-out). Under C, gbr scores TTI **0.405**, THR 0.688, pHRR 0.468 and CR 0.667.
## 5. Population facts
- All 75/63 usable rows come from 2017-2023 sources: pz2023 Table 1 (39), thesis (18), polym2018 (8), pds2020 (5) and jtac11770 (5). No row comes from 2024-2026 papers, because the jtac2026 and polycomp2026 rows lack inputs.
- All 39 pz2023 Table 1 rows (their pHRR resin database) are in our pHRR set. Only 27 of them have TTI/THR/CR, because 12 have no OA source. Their 75 presumably carried all four outputs.
- Their fibre input takes 4 values (~0/40/60/67, pz2026 p.9). Our 3 hemp rows at 30 % fall outside that set, so at least 3 of our 36 composites are probably not in their 75. Their 4 PER/RDP validation composites are not in ours.
- Their per-sample targets are not published. The SD implied by Table 9, sqrt(MSE/(1-R²)), for their held-out set is roughly pHRR 142 kW/m², TTI 19-23 s, THR 26-32 MJ/m² and CR 20 %. Ours over all rows: pHRR 267, TTI 21.5, THR 28.2, CR 22.1. This assumes the printed "14.079" means 14079.
## 6. Protocol-A comparability (facts)
- Theirs: one random 80/20 split (15 test rows if exactly 20 %; repo notes say 19 held out). The R² is pooled with 4 external validation composites, a new formulation series that they underpredicted for THR (13-26 predicted vs 19.5-31 measured). A single draw.
- Ours: mean of 50 splits with 13 test rows (15 for pHRR) and no external rows. On average 52-55 % of test rows have their formulation in train.
- Split-level spread for TTI is [-0.146, 0.879]. Share of our 50 splits above their bar: pHRR 0.80, TTI 0.76, THR 0.88, CR 0.70.
## Caveats a human must see
1. The grouping bug changes the pHRR B trigger: under corrected groups, rf gives ci_lo 0.306 > 0.30. TTI and THR stay above their bars.
2. The TTI pooled-over-repeats ci_lo (0.577-0.586) sits on the bar. 4/10 repeats fail, Bonferroni gives 0.551, only 1/4 candidates clears, and the plan's resin-system protocol gives 0.405.
3. THR is met largely by the fibre and heat-flux split: a 2-feature GBR reaches B 0.677, within-25 kW/m² R² is -0.03, and without pz2023 B is 0.494.
4. The 3 hemp rows tagged "probably exclude" are in the data, and probably not in their population.
5. Process: plan §6 protocol B (resin system) was replaced by formulation grouping in the config. The plan's go/no-go gate was Oct 2 13:00, and the run happened Oct 1 ~22:03. The config was committed at 22:01:50, before the run. The training script was first committed with the results (6c22c47).
6. The bars are best-of-3 algorithms on one split plus 4 external samples. Our A is a 50-split mean on a partly different population.
