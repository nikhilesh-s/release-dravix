# TTI narrative corrections: "good (~0.8) at other fluxes"

Plan item A3 (`plans/LEVER_SWEEP_2026-10-06.md`). Drafted 2026-10-06 by a lever-sweep workflow subagent. **This file edits nothing.** It lists every place in the repo that states or implies TTI is good (~0.8) away from 25 kW/m², quotes the exact current text, and gives the exact replacement text. The orchestrator decides what to apply and commits it. Companion config: `experiments/configs/tti_secondary_metrics.json` (report-only metrics for the whole TTI shelf).

The numbers of record do not change. exp046 stays 0.3133 [0.071, 0.496].

## 1. Verified facts the replacements use

All from the skeptic verdicts on tti-1 / tti-3 in `plans/LEVER_SWEEP_2026-10-06_findings.json`, except exp046's overall log R² 0.391 with its CI and its repeat-0 row Spearman (0.49 / 0.44 / 0.39), which come from the tti-2 finding and verdict (exp046 values only; no b2 / ET-alone / C5 number is used). Each was reproduced in scratch, and exp046 was reproduced exactly (0.3133 [0.0715, 0.4958]). Means are over the 10 repeats unless marked "repeat 0".

1. **What "other fluxes" was.** The legacy "other fluxes" figure is seconds R² on every row with heat flux ≠ 25 kW/m² (the `~f25` mask in `train_tti_v5_physics.run()`, `train_tti_round1.py` and `train_tti_round2.py`). On the 681-row table that mask covers 458 rows: 3 at 15 kW/m² (all Masonite_Board, 398–488.5 s), 16 at 35, 221 at 50, 217 at 75 and 1 at 100. The 3 PMMA 15 kW/m² rows are non-ignitions and are dropped. On the 672-row `tti_fsri.csv` ignited set behind `reports/sprint_0929/tti.md` (exp024 / exp039), the same mask covers 3 Masonite rows at 15, 12 at 35, 220 at 50, 214 at 75 and 3 at 100. The drafter checked these row counts; no score was involved.
2. **exp046 on that pool:** 0.798 (repeat 0), 0.768 ± 0.070 (10-repeat). The 3 Masonite rows hold **74.5%** of the pool's target sum of squares. Without them the pool scores 0.294 ± 0.205.
3. **exp046 within a single flux.** Seconds R² is **0.042 / 0.274 / 0.283** at 25 / 50 / 75 kW/m². Pooled over 50+75 it is **0.368** (repeat-0 CI [−0.42, 0.59]). Log-space R² is **0.025 / 0.018 / −0.006**, so the model barely separates materials at any single flux. Rank skill is moderate: material-median Spearman 0.51 / 0.50 / 0.44 (repeat 0: 0.50 / 0.46 / 0.40). Overall log R² is 0.391 (CI [0.09, 0.57]).
4. **Why 25 kW/m² dominates.** The 25 kW/m² rows hold **85.2%** of the target sum of squares and 97.8% of exp046's repeat-0 OOF SSE. So the pooled-seconds headline mostly measures 25 kW/m². The model is not "good elsewhere and broken at 25". (All-row R² without the 15 kW/m² rows is 0.285 ± 0.040. That is a fact about the metric, not a proposal to drop rows.)
5. **25 kW/m² ceilings** (label-only, 681 rows). At 25 kW/m² (223 rows / 69 materials), leave-one-replicate-out (LORO) gives **0.558** and the in-sample cell mean gives 0.776. LORO is 0.958 at 50 and 0.955 at 75. Overall the ceiling is 0.843 in-sample and 0.690 LORO.
6. **Own-50/75 reference model.** A model given each material's own measured 50 and 75 kW/m² ignition times scores **0.29–0.49** at 25 kW/m², depending on how replicates are aggregated. Combined with the LORO cell means at the other fluxes, it scores **0.50–0.62** overall. This is a reference point, not a strict bound. Scaling t50 by 4 gives R² 0.329 (Spearman 0.87).
7. **HRR-curve ignition times.** Correcting tti-3's own evidence: these times do shrink the replicate scatter at 25 kW/m². On matched cells the 90th-percentile within-cell CV falls from 0.48 to 0.36–0.37, but the scatter stays large.
8. **Other shelf models** report "other fluxes" on the same mask: t1 0.802, t3 0.705, t4 0.499, e1 0.801, e2 0.819, e3 0.321, e4 0.798 and exp047 0.397 (repeat 0). Their per-flux values have **not** been computed yet. `tti_secondary_metrics.json` declares that run.
9. **Hold these until the declared run and its critic.** The post hoc secondary numbers for b2, ET alone and C5 Ridge (tti-2) must not be quoted in any doc until `tti_secondary_metrics.json` has run and its critic has passed. They must never be set against 0.58.

## 2. Wording rules applied below

- Call the legacy mask "non-25 rows pooled (15/35/50/75/100 kW/m²)". Never call it "other fluxes" or "at 50/75".
- Report 50 and 75 kW/m² separately, plus pooled 50+75.
- Label exp046's per-flux numbers as a post hoc diagnostic (lever sweep tti-1).
- Never compare any per-flux or log number with the 0.58 bar.
- For dated historical files (sprint reports, the acquisition log, plans, the archive), prefer an erratum over a rewrite. Both options are given where relevant.

## 3. Index

| # | Place | States / implies | Who may edit | Recommended action |
|---|---|---|---|---|
| A1 | `STATUS.md` (all 49 lines) | none found | orchestrator | optional one-line addition (row 9) |
| A2 | `reports/model_cards/TTI.md` line 27, plus a new block after line 25 | implies | orchestrator | apply |
| A3 | `reports/V2_FINAL_SUMMARY.md` line 44 | implies | orchestrator | apply |
| A4 | `reports/sprint_0929/tti.md` lines 26–27, 141, 153, 203, 213–214, 223–224, 332 (+ erratum after line 1); line 304 (fenced leaderboard-row copy) no change | states | orchestrator | erratum block + line edits |
| A5 | `experiments/numbers_of_record.json` TTI `note` | none found | orchestrator only | optional note sentence |
| B1 | `reports/sprint_0929/tti_replicates.md` lines 23, 49 | states | orchestrator | apply |
| B2 | `blockers/BLOCKER-20261004-1156-tti-round1-no-gain.md` lines 5, 11, 16 | states / implies | orchestrator | apply |
| B3 | `blockers/BLOCKER-20261005-tti-round2-no-gain.md` line 8 | implies | orchestrator | apply |
| B4 | `reports/V2_DATA_ACQUISITION_LOG.md` line 240 | states | orchestrator | erratum (or line edit) |
| C1 | `plans/V3_PLAN_2026-10-04.md` line 17 | states | orchestrator | erratum under the table |
| C2 | `plans/FABLE_PLAN_2026-10-01.md` line 101 | states | orchestrator | erratum |
| C3 | `docs/archive/SPRINT_HANDOFF_2026-09-29.md` line 202 | states | orchestrator | leave (archive), optional erratum |
| D1 | `plans/LEVER_SWEEP_2026-10-06_findings.json` tti-3 `evidence` | wrong HRR-curve claim (not "other fluxes") | orchestrator | leave (verdict already corrects it), optional fix |
| E | leaderboard / xlsx notes, result JSON notes, committed configs, run logs, record-script print labels, critic files | states (label) | nobody by hand | do not edit (section 8) |

## 4. Files named in the task

### A1. `STATUS.md`: nothing to correct

No line states or implies it. TTI appears on lines 1, 9, 24, 28 and 31, always as "parked" or "no gain". Optional, because STATUS is rewritten every round: line 9 can carry the corrected story.

Current (line 9):
```text
| TTI (FSRI) | exp046 | 0.313 [0.071, 0.496] | 0.58 | parked (round 1 and round 2 / emissivity: no gain) |
```
Replacement:
```text
| TTI (FSRI) | exp046 | 0.313 [0.071, 0.496] | 0.58 | parked (round 1 and round 2 / emissivity: no gain). Weak at every flux, not only 25 kW/m2: pooled 50+75 R² 0.368, within-flux log R² ~0; the old "0.80 at other fluxes" was mostly 3 Masonite rows at 15 kW/m2. 25 kW/m2 replicate ceiling (LORO) 0.558 |
```

### A2. `reports/model_cards/TTI.md`

**(a) Line 27 (implies a 25-only failure).** Current:
```text
**V3 round 1 (Oct 4; record unchanged).** Four pre-declared candidates (`experiments/configs/tti_round1.json`, `data/v2/train_tti_round1.py`): GLM + log(q″ − q_cr) 0.298; HGB Poisson + critical-flux terms 0.212; HGB log1p 0.267; Ridge on TTI^-1/2 −38 (its declared back-transform blows up). 0/10 paired wins for each; exp046 reproduced exactly. The critical-flux term is ~95% explained by log q″ and log ΔT, which exp046 already has. R² at 25 kW/m² stays ≤ 0.06 for three model families. Critic: `experiments/critic/tti_round1_critic.md`.
```
Replacement:
```text
**V3 round 1 (Oct 4; record unchanged).** Four pre-declared candidates (`experiments/configs/tti_round1.json`, `data/v2/train_tti_round1.py`): GLM + log(q″ − q_cr) 0.298; HGB Poisson + critical-flux terms 0.212; HGB log1p 0.267; Ridge on TTI^-1/2 −38 (its declared back-transform blows up). 0/10 paired wins for each; exp046 reproduced exactly. The critical-flux term is ~95% explained by log q″ and log ΔT, which exp046 already has. R² at 25 kW/m² stays ≤ 0.06 for three model families (repeat 0). The weakness is not confined to 25 kW/m² (see "Per-flux skill" above). Critic: `experiments/critic/tti_round1_critic.md`.
```

**(b) New block: insert after line 25** (the last "Known limitations" bullet), with a blank line before and after. It adds the per-flux correction and the tti-3 ceilings the plan asks for, including the corrected HRR-curve statement.
```text
**Per-flux skill (post hoc diagnostic, report-only; lever sweep 2026-10-06, `plans/LEVER_SWEEP_2026-10-06_findings.json` tti-1).** exp046 is not "good away from 25 kW/m²". The often-quoted "R² ≈ 0.80 at other fluxes" (repeat 0: 0.798; 10-repeat mean 0.768 ± 0.070) pools every non-25 row (15, 35, 50, 75 and 100 kW/m²), and 3 Masonite_Board rows at 15 kW/m² (398–489 s) hold 74.5% of that pool's target sum of squares; without them it is 0.294 ± 0.205. Within one flux (10-repeat means) seconds R² is 0.042 at 25, 0.274 at 50 and 0.283 at 75 kW/m², and 0.368 pooled over 50+75 (repeat-0 CI [−0.42, 0.59]). Log-space R² is 0.025 / 0.018 / −0.006, so the model barely separates materials at any single flux; rank skill is moderate (material-median Spearman 0.51 / 0.50 / 0.44). The pooled-seconds headline mostly measures the 25 kW/m² rows, which hold 85.2% of the target sum of squares and 97.8% of exp046's repeat-0 out-of-fold SSE. These numbers are descriptive, were found after the fact, and are never compared with the 0.58 bar. The same metrics for every TTI model on the shelf are declared in `experiments/configs/tti_secondary_metrics.json` (report-only; never promotes).

**The 25 kW/m² ceiling (label-only; tti-3).** On the 223 rows / 69 materials at 25 kW/m², predicting each replicate from its siblings in the same material × flux cell (leave-one-replicate-out) gives R² 0.558 (in-sample cell mean 0.776); at 50 and 75 kW/m² the same ceiling is 0.958 and 0.955. A reference model that already knows each material's own measured 50 and 75 kW/m² ignition times scores 0.29–0.49 at 25 kW/m², depending on how replicates are aggregated, or 0.50–0.62 overall when combined with the leave-one-replicate-out cell means at the other fluxes. That is a reference point, not a strict upper bound, but the pooled-seconds bar (0.58) sits at roughly the level of that oracle. Ignition times read from the HRR curve shrink the 25 kW/m² replicate scatter (90th-percentile within-cell CV 0.48 → about 0.36) but leave it large.
```
Note: line 19 quotes the numbers_of_record note verbatim. If A5 is applied, update line 19 to match.

### A3. `reports/V2_FINAL_SUMMARY.md`

**Line 44 (implies a 25-only failure).** Current:
```text
- **TTI round 1** (`experiments/critic/tti_round1_critic.md`): a critical-heat-flux term, a 1/√t target, and HGB variants. None beat exp046 (best 0.298, 0/10 paired wins). The 25 kW/m² failure (R² ≤ 0.06 across three model families) is not fixed by bench-derived physics. TTI parked (`blockers/BLOCKER-20261004-1156-tti-round1-no-gain.md`).
```
Replacement:
```text
- **TTI round 1** (`experiments/critic/tti_round1_critic.md`): a critical-heat-flux term, a 1/√t target, and HGB variants. None beat exp046 (best 0.298, 0/10 paired wins). R² on the 25 kW/m² rows stays ≤ 0.06 across three model families (repeat 0) and bench-derived physics does not fix it. The weakness is not confined to 25 kW/m²: within 50 or 75 kW/m², exp046's log R² is about 0 and its pooled 50+75 R² is 0.368; the 25 kW/m² rows dominate the headline because they hold 85% of the seconds variance (post hoc diagnostic, lever sweep tti-1). TTI parked (`blockers/BLOCKER-20261004-1156-tti-round1-no-gain.md`).
```
Line 56 ("TTI 0.332 (7/10, not adopted; R2 at 25 kW/m2 0.06 -> 0.13)") makes no claim about other fluxes. No change.

### A4. `reports/sprint_0929/tti.md` (Sep 29 sprint report: exp024 era, 672 rows)

This is a dated report, so the main fix is an erratum after the title. Per-line edits follow for anyone who wants the body consistent too. Every "0.84 at 50/75" or "0.84 at the other fluxes" here (not the 0.84 / 0.842 in-sample ceiling on lines 32, 157 and 220) is C6's **repeat-0** "other fluxes" value on the 672-row set (`experiments/runs/train_tti_v5_physics_onset_bc.log`: 0.1075 / 0.8356). It was not recomputed per flux. The corrected per-flux numbers come from the same model form on the canonical 681-row table (exp046), and the text says so.

**Erratum block: insert after line 1** (blank line before and after):
```text
> **Erratum (2026-10-06, lever sweep tti-1; `experiments/critic/tti_narrative_corrections.md`).** Every "0.84 at 50/75 kW/m2" or "0.84 at the other fluxes" below is C6's repeat-0 R2 on *all non-25 rows pooled* (on these 672 rows: 3 Masonite_Board rows at 15 kW/m2, 12 at 35, 220 at 50, 214 at 75, 3 at 100). It is not a 50/75 kW/m2 score. On the canonical 681-row table the same model form (exp046) gets 0.798 on that pool (repeat 0), but the 3 Masonite rows at 15 kW/m2 hold 74.5% of the pool's target sum of squares (0.294 +- 0.205 without them); its pooled 50+75 R2 is 0.368 and its log R2 within any single flux is about 0 (0.025 / 0.018 / -0.006 at 25 / 50 / 75). The 672-row per-flux values were not recomputed. At 25 kW/m2 alone the replicate ceiling (leave-one-replicate-out, 681 rows) is 0.558.
```

**(a) Lines 26–27 (states).** Current:
```text
- **Where the physics works, and where it can't.** With the corrected onset, C6's out-of-fold R2 in seconds is **0.84 at 50/75
  kW/m2 and 0.11 at 25 kW/m2**. Before any fitting, (pi/4) k rho c (T_onset-T0)^2 / q''^2 has Spearman +0.784 with measured TTI.
```
Replacement:
```text
- **Where the physics works, and where it can't.** With the corrected onset, C6's repeat-0 out-of-fold R2 in seconds is **0.11 on
  the 25 kW/m2 rows**, and 0.84 on all other rows pooled (15/35/50/75/100 kW/m2; not a 50/75 score, see the erratum above).
  Before any fitting, (pi/4) k rho c (T_onset-T0)^2 / q''^2 has Spearman +0.784 with measured TTI.
```

**(b) Line 141 (column definition).** Current (line 140 ends with "R2 on 25 kW/m2 rows / on all"):
```text
other fluxes.
```
Replacement:
```text
other fluxes pooled (15/35/50/75/100 kW/m2; not a 50/75 score, see the erratum above).
```

**(c) Line 153 (table cell emphasises the pooled figure).** In this exact current line, change only the last cell from `0.108 / **0.836** |` to `0.108 / 0.836 |`:
```text
| **C6 Poisson GLM physics (pick)** | **0.425 +- 0.171 (0.144, 0.633)** | **0.367 +- 0.039** | **[0.201, 0.520]** | **0.372 +- 0.025 [-0.032, 0.594]** | 0.58 | 0.402 | 0.108 / **0.836** |
```
Replacement:
```text
| **C6 Poisson GLM physics (pick)** | **0.425 +- 0.171 (0.144, 0.633)** | **0.367 +- 0.039** | **[0.201, 0.520]** | **0.372 +- 0.025 [-0.032, 0.594]** | 0.58 | 0.402 | 0.108 / 0.836 |
```

**(d) Line 203 (states).** Current:
```text
GKF stds. The gap is also consistent with the physics diagnostics: 0.84 R2 at 50/75 kW/m2, and the corrected onset raises the
```
Replacement:
```text
GKF stds. The gap is also consistent with the physics diagnostics: 0.84 R2 on the pooled non-25 rows (mostly between-flux spread and 3 Masonite rows at 15 kW/m2; see the erratum above), and the corrected onset raises the
```

**(e) Lines 213–214 (states, and credits the flux law for it).** Current:
```text
   median -2.08 (IQR -2.52 to -1.73, 67 materials), which is the thermally thick q''^-2 law. That's why the physics GLM reaches 0.84
   at 50/75 kW/m2. But TTI(25)/TTI(50) exceeds 6 for 21 of 67 materials (max 13.9, plastic laminate), where the law says 4. Examples
```
Replacement:
```text
   median -2.08 (IQR -2.52 to -1.73, 67 materials), which is the thermally thick q''^-2 law. The physics GLM captures that law (how
   TTI moves with flux), which is most of what its 0.84 on the pooled non-25 rows measures; it does not mean the GLM separates
   materials at 50 or 75 kW/m2 (erratum above). But TTI(25)/TTI(50) exceeds 6 for 21 of 67 materials (max 13.9, plastic laminate), where the law says 4. Examples
```

**(f) Lines 223–224 (states).** Current:
```text
4. **The 25 kW/m2 rows are unpredictable for every model tried.** OOF R2 on them runs from -0.14 to +0.23 across all 32
   model-by-row-set runs, and it's 0.11 for the pick. At the other fluxes the pick reaches 0.84.
```
Replacement:
```text
4. **The 25 kW/m2 rows are unpredictable for every model tried.** OOF R2 on them runs from -0.14 to +0.23 across all 32
   model-by-row-set runs, and it's 0.11 for the pick. The pick's 0.84 on the other rows is a pooled non-25 figure, not skill at
   50/75 kW/m2: on the 681-row table the same model (exp046) scores 0.274 / 0.283 in seconds at 50 / 75 kW/m2 and about 0 in log
   space (10-repeat means, lever sweep tti-1).
```

**(g) Optional, line 222 (adds the 25-only ceiling to point 3).** Current:
```text
   on a target where the ceiling itself is set by 25 kW/m2 behaviour.
```
Replacement:
```text
   on a target where the ceiling itself is set by 25 kW/m2 behaviour. At 25 kW/m2 alone the leave-one-replicate-out ceiling is 0.558
   (681-row table; lever sweep tti-3, 2026-10-06).
```

**(h) Line 332 (inside the fenced "Proposed V2_DATA_ACQUISITION_LOG.md entry"; states).** Current:
```text
**Why it stops there:** the GLM explains 84% of the variance in seconds at 50/75 kW/m2 and 11% at 25 kW/m2. About 20 materials sit
```
Replacement:
```text
**Why it stops there:** the GLM explains 11% of the variance in seconds at 25 kW/m2 and is weak at the other fluxes too (its 84% on all non-25 rows pooled is mostly between-flux spread and 3 Masonite rows at 15 kW/m2, not skill at 50/75). About 20 materials sit
```

**No change:** line 304, inside the fenced section 9 "Proposed `experiments/leaderboard.csv` rows", carries "OOF R2 0.84 at 50/75 kW/m2 vs 0.11 at 25 kW/m2". It is a verbatim copy of the ingested exp024 leaderboard note (section 8: not hand-editable), so the copy stays identical to the row. The erratum block covers it.

**No change:** line 244, item (d), "Report TTI in log space or per flux as a secondary metric, because raw-seconds R2 on FSRI mostly measures 25 kW/m2 behaviour". This is correct, and it is the Sep 29 precedent that `tti_secondary_metrics.json` cites.

### A5. `experiments/numbers_of_record.json`: nothing to correct

The TTI `note` makes no per-flux claim: `"Parked per plan B5. Physics Poisson GLM on the canonical per-replicate table."`. The `pomazi_recon.TTI` block is a different population and is not affected. Optional, orchestrator only. If applied, update the verbatim quote on line 19 of `reports/model_cards/TTI.md`, then run `scripts/update_trackers.py regen`, because the By Target sheet shows the note.

Current:
```text
      "note": "Parked per plan B5. Physics Poisson GLM on the canonical per-replicate table."
```
Replacement:
```text
      "note": "Parked per plan B5. Physics Poisson GLM on the canonical per-replicate table. Not good away from 25 kW/m2 either: pooled 50+75 R2 0.368, within-flux log R2 ~0 (0.025 / 0.018 / -0.006 at 25 / 50 / 75); the old 'other fluxes 0.80' pooled every non-25 row and was mostly 3 Masonite rows at 15 kW/m2 (lever sweep tti-1, post hoc, report-only). 25 kW/m2 replicate ceiling (LORO) 0.558. Shelf-wide secondary metrics: experiments/configs/tti_secondary_metrics.json."
```

## 5. Other live documents found by grep

### B1. `reports/sprint_0929/tti_replicates.md` (the T1 report that made exp046 the record)

**Line 23 (states).** Current:
```text
- **Where the error sits (GKF repeat 0, unchanged picture from exp039):** C6 OOF R2 is 0.798 on rows at fluxes other than 25 kW/m2 and 0.059 on 25 kW/m2 rows. The 4 slowest igniters hold 52% of its SSE. The replicate-noise ceiling on these rows is 0.843 in-sample and 0.690 leave-one-replicate-out.
```
Replacement:
```text
- **Where the error sits (GKF repeat 0, unchanged picture from exp039):** C6 OOF R2 is 0.059 on 25 kW/m2 rows and 0.798 on all other rows pooled (15/35/50/75/100 kW/m2). *Erratum (2026-10-06, lever sweep tti-1):* the 0.798 is not skill at 50/75 kW/m2; 3 Masonite_Board rows at 15 kW/m2 hold 74.5% of that pool's target sum of squares (0.294 +- 0.205 without them, 10-repeat). Within one flux C6 scores 0.042 / 0.274 / 0.283 in seconds and 0.025 / 0.018 / -0.006 in log space at 25 / 50 / 75 kW/m2 (10-repeat means); pooled 50+75 it is 0.368. The 4 slowest igniters hold 52% of its SSE. The replicate-noise ceiling on these rows is 0.843 in-sample and 0.690 leave-one-replicate-out (at 25 kW/m2 alone: 0.776 / 0.558).
```

**Line 49 (table header).** Current:
```text
| cand. | legacy | GKF (headline) | CI | family GKF [CI] | top-4 SSE share | R2 excl. top 4 | R2 @25 / other | OOF R2 log1p |
```
Replacement:
```text
| cand. | legacy | GKF (headline) | CI | family GKF [CI] | top-4 SSE share | R2 excl. top 4 | R2 @25 / non-25 pooled | OOF R2 log1p |
```
The cell values on lines 51–52 do not change.

### B2. `blockers/BLOCKER-20261004-1156-tti-round1-no-gain.md` (open FLAG listed in STATUS)

**Line 5 (table header).** Current:
```text
| Candidate | GKF mean | CI | Paired wins vs exp046 | R² at 25 kW/m² | Other fluxes |
```
Replacement:
```text
| Candidate | GKF mean | CI | Paired wins vs exp046 | R² at 25 kW/m² | Non-25 rows pooled* |
```
**Insert after line 11** (the last table row, so before the blank line 12):
```text

\*Repeat 0, all rows at 15/35/50/75/100 kW/m² pooled; not skill at 50/75 kW/m². For exp046, 3 Masonite_Board rows at 15 kW/m² hold 74.5% of this pool's target sum of squares; its pooled 50+75 R² is 0.368 and its within-flux log R² is about 0 (lever sweep tti-1, 2026-10-06). The candidates' per-flux values are in the report-only run declared by `experiments/configs/tti_secondary_metrics.json`.
```
**Line 16 (implies a 25-only failure).** Current:
```text
- Three real model families (GLM, HGB Poisson, HGB log1p) all score R² ≤ 0.06 at 25 kW/m² (repeat 0). The gap looks like data rather than model choice, but this is not proven.
```
Replacement:
```text
- Three real model families (GLM, HGB Poisson, HGB log1p) all score R² ≤ 0.06 at 25 kW/m² (repeat 0). The gap looks like data rather than model choice, but this is not proven. The weakness is not confined to 25 kW/m²: exp046's log R² within 50 or 75 kW/m² is also about 0; 25 kW/m² dominates the headline because it holds 85% of the seconds variance.
```

### B3. `blockers/BLOCKER-20261005-tti-round2-no-gain.md`

**Line 8 (implies a 25-only failure).** Current:
```text
- meaning: neither bench physics (round 1) nor measured absorptance (round 2) fixes the 25 kW/m2 failure. Remaining levers are new data (melt/char behaviour, more materials), not features from the FSRI tables.
```
Replacement:
```text
- meaning: neither bench physics (round 1) nor measured absorptance (round 2) fixes TTI's weak separation of materials, which is worst in seconds at 25 kW/m2 (85% of the target variance) but is also near zero in log space within 50 and 75 kW/m2. Remaining levers are new data (melt/char behaviour, more materials), not features from the FSRI tables.
```

### B4. `reports/V2_DATA_ACQUISITION_LOG.md` (dated log; CLAUDE.md lists it as not-read-by-default)

Line 240 is the same text as tti.md line 332. Preferred fix: append an erratum after line 246, the last line of that dated entry. Line edit only if the log is not treated as append-only.

Erratum to insert after line 246:
```text
*Erratum (2026-10-06, lever sweep tti-1):* "84% of the variance in seconds at 50/75 kW/m2" is C6's repeat-0 R2 on all non-25 rows pooled (15/35/50/75/100), mostly between-flux spread and 3 Masonite rows at 15 kW/m2. On the canonical 681-row table exp046's pooled 50+75 R2 is 0.368 and its within-flux log R2 is about 0. See `experiments/critic/tti_narrative_corrections.md`.
```
Alternative line edit. Current (line 240):
```text
**Why it stops there:** the GLM explains 84% of the variance in seconds at 50/75 kW/m2 and 11% at 25 kW/m2. About 20 materials sit
```
Replacement:
```text
**Why it stops there:** the GLM explains 11% of the variance in seconds at 25 kW/m2 and is weak at the other fluxes too (its 84% on all non-25 rows pooled is mostly between-flux spread and 3 Masonite rows at 15 kW/m2, not skill at 50/75). About 20 materials sit
```

## 6. Historical plans and archive (decision records: prefer an erratum)

### C1. `plans/V3_PLAN_2026-10-04.md` line 17 (states)

Current:
```text
| TTI | 0.313 (exp046) | 0.58 | near-critical-flux failure (OOF R² 0.80 away from 25 kW/m², 0.06 at 25) |
```
Preferred: leave the row and insert this erratum after line 19 (the last table row, `| pHRR, CR, FIGRA | ...`; inserting after line 18 would split the table), with a blank line before it:
```text
> Erratum (2026-10-06, lever sweep tti-1): "OOF R² 0.80 away from 25 kW/m²" pooled every non-25 row (15/35/50/75/100); 3 Masonite_Board rows at 15 kW/m² hold 74.5% of that pool's target sum of squares. exp046's pooled 50+75 R² is 0.368 and its within-flux log R² is about 0. See `experiments/critic/tti_narrative_corrections.md`.
```
If rewritten instead:
```text
| TTI | 0.313 (exp046) | 0.58 | weak material separation at every flux, worst at 25 kW/m² (OOF R² 0.06 there; the "0.80 away from 25" pooled all non-25 rows and was mostly 3 Masonite rows at 15 kW/m²; pooled 50+75 R² 0.368) |
```

### C2. `plans/FABLE_PLAN_2026-10-01.md` line 101 (states)

Current:
```text
- **Report as is:** 0.313 [0.071, 0.496], with the existing finding that out-of-fold R² is 0.80 away from 25 kW/m² and 0.06 at 25 kW/m².
```
Preferred: leave the line and insert directly after it:
```text
  - *Erratum (2026-10-06, lever sweep tti-1):* the "0.80 away from 25 kW/m²" pooled every non-25 row and was mostly 3 Masonite_Board rows at 15 kW/m² (74.5% of that pool's target sum of squares). exp046's pooled 50+75 R² is 0.368 and its within-flux log R² is about 0.
```
If rewritten instead:
```text
- **Report as is:** 0.313 [0.071, 0.496], with the existing finding that out-of-fold R² is 0.06 at 25 kW/m² (repeat 0); exp046 is weak within 50 and 75 kW/m² too (pooled 50+75 R² 0.368, within-flux log R² about 0; the earlier "0.80 away from 25 kW/m²" pooled all non-25 rows and was mostly 3 Masonite rows at 15 kW/m²).
```

### C3. `docs/archive/SPRINT_HANDOFF_2026-09-29.md` line 202 (states)

This is an archive file, and CLAUDE.md says not to read `docs/archive/*` unless a named question requires it. Recommendation: leave it. If it is touched, current:
```text
2. **TTI:** decide whether `early_stop` rows should stay. Those 9 specimens did ignite before the test was stopped, so their TTI is valid. Also try a two-stage or censored model ("ignites at this flux?" then "how long?"). The error lives at 25 kW/m² (R² 0.06 there vs 0.80 elsewhere).
```
Replacement:
```text
2. **TTI:** decide whether `early_stop` rows should stay. Those 9 specimens did ignite before the test was stopped, so their TTI is valid. Also try a two-stage or censored model ("ignites at this flux?" then "how long?"). Most of the seconds error is at 25 kW/m² (R² 0.06 there). The "0.80 elsewhere" pooled all non-25 rows and was mostly 3 Masonite rows at 15 kW/m²; within 50 or 75 kW/m² log R² is about 0 (erratum 2026-10-06, lever sweep tti-1).
```

## 7. Plan artifact

### D1. `plans/LEVER_SWEEP_2026-10-06_findings.json`, finding tti-3, `evidence` (the HRR-curve claim the plan asks to correct)

The skeptic verdict in the same object already corrects this sentence, and plan line 71 records the correction. Recommendation: leave the findings file as the research record. If it is edited, current (a substring of the `evidence` string):
```text
Ignition times read from the HRR curve (first HRRPUA >= 25/50/100 kW/m2 held for 10 s) do not reduce the scatter (90th-pct CV at 25 kW/m2 0.41 vs 0.44), so the scatter is real specimen behaviour, not operator timing.
```
Replacement:
```text
Ignition times read from the HRR curve (first HRRPUA >= 25/50/100 kW/m2 held for 10 s) shrink the scatter but leave it large (skeptic, matched cells: 90th-pct CV at 25 kW/m2 0.48 -> 0.36-0.37, and 0.39 -> 0.21 at the 100 kW/m2 threshold), so part of the scatter is how ignition is timed and a large part remains (specimen behaviour is the leading hypothesis).
```
A related minor point, not an "other fluxes" claim: tti-1's `evidence` gives "Material-median Spearman is 0.50, 0.46 and 0.40" next to 10-repeat means. Those are repeat-0 values; the 10-repeat means are 0.51 / 0.50 / 0.44 (verdict).

## 8. Places that say "other fluxes" but must not be hand-edited

| Place | Text | Why not edited | What to do |
|---|---|---|---|
| `experiments/leaderboard.csv` row exp024 `notes` | "OOF R2 0.84 at 50/75 kW/m2 vs 0.11 at 25 kW/m2" | Hand-editing is banned; `scripts/update_trackers.py` is the only writer and has no note-amendment path (checked) | None now. The correction lives in the card, this file and the declared report. Fixing it needs a tracker-script change under its own config (Arnav's call). |
| `experiments/leaderboard.csv` rows exp096–exp099, exp104–exp107 `notes`; same text in `reports/leaderboard.xlsx` / `reports/daily_scoreboard.xlsx` (generated) | "R2 at 25 kW/m2 …, other fluxes 0.802" etc. | Same; the xlsx files are regenerated from the CSV | Same |
| `experiments/results/exp096…exp099, exp104…exp107*.json` `note` | "other fluxes …" | Ingested worker records; editing breaks provenance | Leave |
| `experiments/configs/tti_round1.json` line 21, `tti_round2.json` line 25 | "R2 at 25 kW/m2 vs other fluxes reported" | Committed pre-declarations; editing would rewrite what was declared. The label names what was computed. | Leave |
| `experiments/runs/train_tti_v5_physics*.log`, `train_tti_round1.log`, `train_tti_round2.log`, `repro_*.log`, `reports/sprint_0929/mcc_fix_scripts/*.log` | "seconds R2 on 25 kW/m2 rows …, other fluxes …" | Immutable run output | Leave |
| `data/v2/train_tti_v5_physics.py` lines 274–275; `train_tti_round1.py` line 105; `train_tti_round2.py` line 127 | print / note label "other fluxes" | Record-reproduction scripts (`run_numbers_of_record.sh` runs v5); a label change alters the reproduction logs | Leave. New scripts label the mask "non-25 rows pooled (15/35/50/75/100)" (rule in `tti_secondary_metrics.json`). |
| `experiments/critic/tti_round1_critic.md` lines 28, 39–43 | "the 25 kW/m2 failure", "predicts the 25 kW/m2 blow-ups" | Historical critic verdict. It refers to the real 25 kW/m² blow-ups and makes no claim about other fluxes. | Leave |
| `data/v2/train_tti_v5_physics.py` line 167 (comment) | "diagnosis put the error at 25 kW/m2" | True for seconds SSE (97.8%) | Leave |

## 9. Searched with no hit

`README.md`, `HANDOFF_TO_NEXT_CLAUDE.md`, `FABLE_BRIEF_2026-10-01.md`, `demo/predict.py`, `reports/figures/v2_final/README.md`, the other model cards (including `pomazi_recon.md`), `experiments/critic/tti_round2_critic.md`, `experiments/critic/blend_round1_critic.md`, and `plans/LEVER_SWEEP_2026-10-06.md` (already correct). Searches combined these patterns: `grep -rn -i -E "other flux|away from 25|elsewhere|0\.798|0\.80|0\.84 at|at 50/75|50/75 kW|R2 0\.8|R² 0\.8|25 kW/m² failure|25 kW/m2 failure|error lives at 25|error sits"` over `*.md *.json *.py *.log *.txt *.csv`, excluding `data/v2/raw/`, then reading each TTI hit in context.
