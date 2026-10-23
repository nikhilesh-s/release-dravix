# Critic: A3 TTI secondary metrics (report-only), blocked run + narrative corrections
Verdicts:
- **Metric run: the STOP is SOUND, and A3's metric part stays BLOCKED.** The script follows `experiments/configs/tti_secondary_metrics.json`. Step 1 reproduces all 8 records and the ET-alone anchor. Step 2 fails on 1 of 66 checks: exp046 repeat-0 row Spearman at 50 kW/m² (SPR50) is 0.4346 against an expected 0.44. The cause is the config's reference value, not the script. The source scripts printed 0.435, and that was rounded a second time to 0.44. Under the declared rule the failure is real, and the budget rule allows no rerun. **This file clears no secondary-metric number for quoting.** No table (2)–(7) was computed or seen by anyone, including this critic.
- **Narrative corrections (part 2): SOUND WITH CAVEATS.** All six edited files are byte-identical to HEAD plus the exact "Replacement" blocks of `experiments/critic/tti_narrative_corrections.md`, and nothing else in those files changed. The caveats are wording points in the committed correction text and four items (B1–B4) that were not applied.

No SOTA claim is made or implied. No number from this run is set against 0.58.

## Pre-declaration and order (Q1)
- The config was committed at 14:52:15 (dd9c9c6), already carrying the e1–e4 amendment.
- The script was last modified at 14:58:22.
- The executor's smoke test ran at 14:58:54 (`scratchpad/a3/smoke.py`). It used synthetic lognormal labels and fake OOF arrays, so no real label statistic or score was computed.
- The log was written at 15:00:09. The per-model fit times add up to about 51 s, so the run started around 14:59. There is no gap for a second real run after the last script edit, and the scratch directory holds only the smoke files. The evidence fits one real run.
- The blocker was written at 15:01:50.
- Environment: I confirmed python3 3.12.4, sklearn 1.7.1, numpy 2.2.6, pandas 2.3.3, scipy 1.16.1, as pinned.

## Script vs config, item by item (Q2)
| Config item | Script | OK |
|---|---|---|
| data.build / asserts | `v5.df.copy()` after the argv swap. Asserts 681 / 76 / TTI > 0 / spark == 1 / the exact flux counts, plus the mask row and material counts and the 458 / 455 / 3-Masonite counts | yes |
| row_rule | All 681 rows train and score. No exclusion anywhere | yes |
| extra_columns, round 1 | `q_kWm2`, `q_over_dT` and `inv_z` are identical to `train_tti_round1.py` lines 32–33 and 39–40 | yes |
| extra_columns, round 2 | **Not built** (SIGMA and the emissivity join). Disclosed in the docstring and the blocker | deviation, see below |
| models (10, fixed order) | exp046 GLM/PHYS/F; exp047 `v5.MODELS[0][1]`/F0/T; b1 `Avg[glm, log1p ET, log1p ridge]`; b2 `Avg[glm, log1p ET]`; et_alone `log1p_wrap(et())`/F; c5 `linear(Ridge 1.0)`/T; t1–t4 identical to `CANDS` lines 45–50 | yes |
| imports | No import of `train_tti_round1` or `train_tti_round2` | yes |
| protocol.cv | One `repeated_group_kfold(..., n_splits=5, repeats=10, log_target=flag)` per model. Every metric comes from that OOF | yes |
| S*, LOG*, SPR*, SPM* | `r2_score` on the mask; natural log with clip 1e-6; `spearmanr(...).statistic`; per-material medians, then `spearmanr` | yes |
| n_clipped | (repeat, row) pairs with OOF <= 1e-6, per model | yes |
| aggregation / NaN | `np.mean` / `np.std` (ddof 0). A NaN stays NaN, is stored as null and is listed | yes |
| CIs | `bootstrap_ci(y[m], oof[0][m], groups[m], 2000, 0)` for S25/S50/S75/S50_75. LOG_ALL uses all rows. No CI elsewhere | yes |
| paired_report | Strict `>`. A NaN on either side is not a win. Computed for the 14 listed metrics; the config says "16" but lists 14 (config typo) | yes |
| legacy / label checks | NOT25 rep0 + mean ± std; NOT25_NO15 mean ± std; Masonite share of NOT25 SS; F25 share of all-row SS. Labelled "legacy mask, NOT a secondary metric" and "non-25 rows pooled (15/35/50/75/100)" | yes |
| reference_ceilings | Same formulas as `v5.ceiling()`. F50/F75 LORO only. The oracle is not recomputed | yes |
| step 1 | ±0.002 on rkf_mean and CI bounds. exp047's CI is compared at 3 decimals. et_alone has an anchor and c5 has none. Exit 2 on any miss | yes |
| step 2 | All 66 config values are present (26 exp046 + 19 b2 + 8 et_alone + 5 c5 + 8 label/ceiling) with the right rule (±0.002 / equal after 2-dp rounding / win range) | yes |
| stop | Exit 3, no table, no JSON. `experiments/results/` and the trackers are untouched | yes |

- **Round-2 columns not built.** This departs from the wording of `data.extra_columns`. The amendment did not change that wording ("Everything else unchanged").
  - It has no effect on any number: each model uses its explicit feature list, and no model in the amended list reads these columns.
  - It is also what the amendment's own reasoning requires. Joining `emissivity_fsri.csv` with no round-2 model to feed would be a use outside the "TTI round 2 only" exemption.
  - Strictly, the executor resolved a conflict inside the config instead of stopping, but the result is the stricter reading. **Record it in the config amendment before any rerun.**
- **The `EPS = 1e-9` float guard on tolerances** changes nothing. No check lies within 1e-4 of its boundary.

## Leakage (Q3)
- `_fit_predict` clones the model per fold.
- Imputer and scaler sit inside every linear pipeline. `Avg.fit` clones each member, and the log1p wraps are inside the fold.
- GroupKFold is grouped by material (76 groups, no merges), and the disjointness assert runs in `eval_protocol`.
- The ceilings are label-only context and never features.
- No row or material is excluded, and nothing in the script depends on the outcome.

## Replication (scratch, not the repo)
- I copied the script to scratch and pinned `ROOT` to the repo. I redirected `OUT_JSON` to scratch; the diff against the repo script is those two lines.
- Run settings: `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`. Result: **exit 3. Output identical to `experiments/runs/tti_secondary_metrics.log` except the per-model timing column.** Same step-1 table to 4 decimals, same 66 check lines, same single FAIL.
- Step-1 reproduction against the records (leaderboard / result JSONs / `train_tti_v5_physics_replicate_table.log`):

| Model | Record | Reproduced |
|---|---|---|
| exp046 | 0.3133 [0.0715, 0.4958] | 0.3133 [0.0715, 0.4958] |
| exp047 | 0.2613 [0.130, 0.441] | 0.2613 [0.1300, 0.4409] |
| b1 (exp123) | 0.3265 [0.2068, 0.5350] | identical |
| b2 (exp124) | 0.3316 [0.2163, 0.5348] | identical |
| t1–t4 (exp096–099) | 0.2979 / −38.1081 / 0.2123 / 0.2671 | identical, CIs too |
| et_alone | anchor 0.255 | 0.2550 |

- **I did not run past step 2, and I computed no table (2)–(7).** The config publishes nothing from a blocked run. Seeing those tables before the orchestrator decides on an amendment would make that decision post hoc.

## The step-2 failure (Q4)
- **The check:** `exp046.SPR50 rep0`, expected 0.44 under the 2-decimal "equal after rounding" rule. The run got 0.4346, which rounds to 0.43. The FAIL is correct as declared. No single rounding convention turns 0.4346 into 0.44; only rounding twice does (0.4346 → 0.435 → 0.44).
- **Source check, independent of the executor's arrays:**
  - Two saved OOF arrays exist. The finder's `scratchpad/tti/exp046_oof.csv` was written at 13:35, before the config was drafted. The skeptic's `scratchpad/skeptic_tti/oof_exp046.npy` agrees with it to 6e-14.
  - On repeat 0 of those arrays, the row Spearman at 50 kW/m² is **0.43456**.
  - The finder's `tti/an2.py` line 9 prints `round(., 3)`, which gives **0.435**. The skeptic's `an2.py` line 37 prints `%.3f`, which also gives **0.435**.
  - The 25 and 75 kW/m² values print as 0.494 and 0.387, and both round correctly to 0.49 and 0.39.
  - So the "0.44" in the tti-2 evidence and verdict, and from there in the config, is 0.435 rounded a second time. **The executor's diagnosis is confirmed.**
- **No code bug, so no rerun.** The budget rule permits a rerun only for a crash or a code bug. The executor was right not to rerun and not to edit the check, since changing the check to match would be "changing a definition to make a number match".
- **Where the stale "0.44" still lives:**
  - `plans/LEVER_SWEEP_2026-10-06_findings.json` (tti-2 evidence and verdict).
  - The config's `prior_looks` (line 16) and `step2_definition_check.exp046` (line 219).
  - `tti_narrative_corrections.md` line 9, in the facts preamble only.
  - **It is in none of the applied doc text.**

## Narrative corrections (Q5)
- **Exactness.** I rebuilt each file from `git show HEAD:<file>` by applying the corrections file's "Current" → "Replacement" blocks at the cited lines.
  - Every "Current" block matched HEAD at its cited line.
  - All six results are **byte-identical** to the working tree.
  - `git diff` on these files shows only these changes (25 insertions, 12 deletions).
  - Two inserts reuse the existing blank line instead of adding a second one (TTI.md after line 25, tti.md after line 1). The rendered text is the same.

| Item | File | Applied |
|---|---|---|
| A1 | `STATUS.md` line 9 | exact |
| A2 (a), (b) | `reports/model_cards/TTI.md` line 27; two blocks after line 25 | exact |
| A3 | `reports/V2_FINAL_SUMMARY.md` line 44 | exact |
| A4 erratum, (a)–(f), (h) | `reports/sprint_0929/tti.md` | exact; (g) is optional and was not applied |
| C1 | `plans/V3_PLAN_2026-10-04.md`, erratum after line 19 | exact |
| C2 | `plans/FABLE_PLAN_2026-10-01.md`, erratum after line 101 | exact |
| A5, B1–B4, C3, D1 | not applied | A5 is orchestrator-only; C3 and D1 are "leave"; **B1–B4 are marked "apply" in the index** |

- **B1–B4 are still open.** The executor says they were outside its task list, which I cannot see. Commit or record a reason for each.
  - B2 matters most. `BLOCKER-20261004-1156` is an open FLAG, and its table header still reads "Other fluxes".
  - B1 (`tti_replicates.md` line 23) still says "0.798 on rows at fluxes other than 25 kW/m2".
- **Provenance of the numbers in the applied text.** None of them comes from this run. The executor's claim that step 2 reproduced all of them is slightly overstated:
  - **Covered by step 2:** 0.042 / 0.274 / 0.283, 0.368 [−0.42, 0.59], 0.025 / 0.018 / −0.006, SPM 0.51 / 0.50 / 0.44, 0.798, 0.768 ± 0.070, 0.294 ± 0.205, 74.5%, 85.2%, and the 0.843 / 0.690 / 0.776 / 0.558 / 0.958 / 0.955 ceilings.
  - **Not covered:** the 97.8% SSE share (I recomputed 0.9779 from the finder's saved repeat-0 OOF) and the Masonite range 398–488.5 s (checked).
  - **Neither covered nor recomputed:** the oracle range 0.29–0.49 / 0.50–0.62 and the HRR-curve CV 0.48 → 0.36, which are verified by the tti-3 skeptic only (the config forbids recomputing the oracle). Also the 672-row flux counts, which are the drafter's row count.
- **Wording.**
  - No "SOTA", "beats" or "best TTI model" is attached to a secondary metric, and no per-flux or log number is set against 0.58.
  - The model-card ceiling block says the 0.58 bar "sits at roughly the level of" the label-only oracle. That is a statement about how reachable the bar is, not a performance claim, and it is the committed correction text.
  - **The `STATUS.md` line 9 text carries no provenance label.** Every other applied text says "post hoc diagnostic" or "lever sweep tti-1", as the corrections file's own §2 rule asks. Add "(post hoc diagnostic, lever sweep tti-1)" at the next STATUS rewrite.
  - STATUS and V2_FINAL_SUMMARY give within-flux log R² ~0 without the plan's qualifier that rank skill is moderate (material-median Spearman 0.44–0.51). The model card keeps the qualifier. Adding it elsewhere is optional.

## Blocker file and executor files (Q6)
- **The blocker** (`BLOCKER-20261006-1501-tti-secondary-repro.md`) is accurate. Its name follows the config pattern, it is level flag, and its options are correct.
  - It quotes one number from this run, 0.4346. A useful blocker needs it, it carries the report-only label, and this critic now covers it.
  - Trivial imprecision: "every other exp046 check built on the same repeat-0 OOF" also lists the S50/LOG50 10-repeat means.
- **The log** holds only step 1 and the step-2 check lines. Its header carries the report-only / post-hoc / not-comparable-to-0.58 label. The check lines include ET-alone S50_75 0.5800. That is a prior look, and it must never be quoted against 0.58.
- **Executor files:** I changed nothing. There is no mechanical error to fix.
- **Not A3:** `data/v2/conformal*.py`, `experiments/runs/conformal*`, `data/v2/validate_macfp_wood_crosslab.py`, `reports/macfp_wood_crosslab.md`, and the modified `demo/README.md`, `demo/predict.py` and `models/v2/README.md` belong to other items. Keep them out of the A3 commit.

## Unblocking (recommendation for the orchestrator)
**Option 1 (recommended): an amendment before any rerun.** Add to `amendments`:
1. The exp046 SPR50 repeat-0 reference changes from "0.44 (2 dp)" to "0.435 (3 dp, ±0.002)".
   - Cause: the source value was rounded twice.
   - Evidence: the finder's `tti/an2.py` and the skeptic's `an2.py` both print 0.435, and this critic recomputed 0.43456 from their saved OOF arrays, which predate the config.
2. `data.extra_columns`: drop the round-2 emissivity columns. This follows from the e1–e4 amendment, and no model uses them.
3. `paired_report`: "16" becomes "14".
4. Allow exactly one rerun for this cause, and list it in the critic addendum.

This option is legitimate:
- No definition, model, mask or tolerance rule changes.
- Tables (2)–(7) have never been computed or seen, so the amendment cannot be steered by the secondary results.
- The executor's matching edit is mechanical: line 306 of the script, `("50", 0.44)` → a 3-decimal check at 0.435.

**Option 2: park A3's metric run.** It costs nothing. The narrative corrections do not depend on it.

## Caveats to carry
1. **The A3 metric run is BLOCKED.** No secondary metric exists yet, and nothing from tables (2)–(7) may be quoted. After any rerun, this file needs an addendum, or a fresh critic must run, before any number is used. That review covers:
   - the tables, `n_clipped` and the NaN cells;
   - whether t2 is degenerate (capped 1/max(z, 0.02)² back-transform, pooled seconds R² −38);
   - the two headline-template conditions: ≥8/10 LOG_ALL wins, and the allowed-interpretation sentence only if both et_alone and c5_ridge meet it;
   - the 0.58 rule.
2. **Never quote the step-1 diagnostics.** C5 Ridge's pooled seconds 0.2825 and ET alone's repeat-0 CI [0.158, 0.463] are diagnostics only and never candidates.
3. **The six doc edits are exact and use no number from this run.** STATUS line 9 needs a provenance label at the next rewrite, and that line must be carried into each STATUS rewrite.
4. **B1–B4 are still open:** apply them or record why not. B2 (the open FLAG blocker) comes first.
5. **The stale "0.44" stays in the findings JSON and the config `prior_looks`.** Note the correction in the amendment; do not edit the research record.

## Addendum: rerun under amendment 2
Reviewer: a fresh-context critic (`critic:A3-rerun`, Oct 6, from about 16:30 PDT). Scratch work is in `scratchpad/critic_a3_rerun_b/`.

**Verdict: the rerun is SOUND WITH CAVEATS.**
- It is the one rerun amendment 2(d) allows.
- The script change is the single declared check line.
- Steps 1 and 2 pass.
- My scratch replication reproduces the log and the JSON exactly.

This addendum clears tables (2)–(7) for quoting only under "Caveats to carry (rerun)" below. Every number stays report-only and defined post hoc (`experiments/configs/tti_secondary_metrics.json`). Nothing here promotes a model, nothing is set against 0.58, and no number makes a model a candidate.

### Runs of record (config `budget`: each rerun listed with its cause)
| Run | Time (PDT) | Script | Result |
|---|---|---|---|
| 1 | 14:59:00 start; log 15:00:09 | pre-amendment | exit 3. Step 2 failed 1 of 66 checks (exp046 SPR50 rep 0: 0.4346 against 0.44) |
| 2 (the rerun) | 16:26:26 start; log and JSON 16:27:38 | b839892 | exit 0. Steps 1 and 2 pass, tables (2)–(7) written |

**Cause of the rerun** is amendment 2(a). The step-2 reference had been rounded twice (0.4346 → 0.435 → 0.44), and the amendment corrects it to 0.435 (3 decimals, ±0.002).

### Exactly one rerun
- **Order of events:**
  - Script and config were saved at 16:25:34.
  - Amendment commit b839892 at 16:25:47.
  - Rerun started at 16:26:26.
  - The JSON's `generated_at` is 16:27:38.
- **Transcript search.** I searched every transcript of this session (main and all subagents) for commands that run the script. Exactly two ran against the repo:
  - the first run, at 14:59:00;
  - this rerun, at 16:26:26.

  Every other execution was a scratch copy with `OUT_JSON` redirected: the first critic (`critic_a3`, 15:06), a second, concurrent `critic:A3` (`critic_a3_v2`, 16:28) and this review (16:31).
- **No hidden earlier run is possible.** Only 52 s passed between the script edit and the rerun start, and one run takes 71 s or more. The log committed in b839892 is still the failing first-run log.
- **The rerun log against the first-run log** differs in three places only:
  - the per-model timing column;
  - the SPR50 line (0.44 FAIL becomes 0.435 OK);
  - the BLOCKED line, now replaced by the pass line and tables (2)–(7).
- **HEAD moved during the run** to 4bde887 (16:27:27, orchestrator). That commit touched only tracker code and the xlsx files. Script and config are byte-identical at b839892, at 4bde887 and in the working tree.

### Script diff (amendment 2(d))
- **How I diffed it.** The script was untracked at dd9c9c6, so `git diff dd9c9c6` shows the whole file. Instead I diffed b839892's script against the first critic's scratch copy (`critic_a3/rep_copy.py`, copied at 15:06 from the script that run 1 used).
- **What differs:**
  - the two scratch-path lines (`ROOT`, `OUT_JSON`);
  - lines 306–308: one comment, and `("50", 0.44)` under the 2-decimal rule becomes `("50", 0.435, "3")`.

  The 25 and 75 kW/m² checks keep the 2-decimal rule. That is exactly the amendment.
- **Amendments 2(b) and 2(c)** describe what the script already did: no round-2 columns, and wins on the 14 listed metrics. Neither needed a code change.

### Replication
- **Setup.** Scratch copy `rep_copy.py`. Its diff from the repo script is the pinned `ROOT` and `OUT_JSON` redirected to scratch.
- **Environment.** `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`, python 3.12.4, sklearn 1.7.1, numpy 2.2.6, pandas 2.3.3, scipy 1.16.1.
- **Result: exit 0, captured.** The executor could only infer its exit code, because zsh has no `PIPESTATUS`.
  - The log is identical to `experiments/runs/tti_secondary_metrics.log` except the timing column and the "wrote" line.
  - The JSON is identical to `experiments/runs/tti_secondary_metrics.json` except `generated_at` and `runtime_s`.

### Caveats from the first review (items 1 and 2), checked
1. **Tables, `n_clipped` and NaN cells.**
   - Tables (2)–(7) are in the fixed model order, and every header carries the report-only label.
   - Table (6) is labelled "legacy mask, NOT a secondary metric", and NOT25 is described as "non-25 rows pooled (15/35/50/75/100)".
   - The JSON keys match `outputs.json`. The extra keys `rkf_std`, `per_repeat`, `label`, `nan_cells` and `runtime_s` are harmless additions.
   - **NaN cells: none.**
   - **`n_clipped`: c5_ridge 9, every other model 0.** I checked where the 9 fall with a scratch diagnostic, `diag.py`. It recomputes only c5_ridge's OOF, which reproduces its pooled 0.2825, and computes no new metric.
     - Repeat 5 holds 6 of the pairs and repeat 8 holds 3, all at 75 kW/m².
     - They fall on 6 rows from 2 materials, whose labels are 1.25–11.25 s.
     - The raw predictions are negative, −0.004 to −0.265 s: the log1p-target linear model back-transforms below zero.
     - The declared 1e-6 s floor maps each of them to ln(1e-6) ≈ −13.8, about 14 nats below the label.
   - **These are exactly c5_ridge's two LOG_ALL misses.** It scores −0.283 and 0.016 against exp046's 0.387 and 0.263, and its LOG75 drops to −4.56 and −2.53 in the same repeats. The executor's "my reading" is confirmed.
   - **They are not script artefacts.** c5_ridge predicts physically impossible negative TTI, and the declared rule scores that. The count is 8/10, as declared.
   - **Never recompute the count with another floor.** The finder's 1e-3 floor is not equivalent for c5_ridge, because its predictions are not strictly positive. The config's "identical for strictly positive predictions" note does not cover it.
2. **t2 is degenerate on the seconds and log scales.** My scratch recompute reproduces its pooled −38.1081.
   - 662 of 6,810 (repeat, row) predictions sit at the 2500 s cap of `1/max(z, 0.02)²`: 53–76 per repeat, 550 of them at 25 kW/m².
   - 62–88 predictions per repeat exceed the largest label (1102.5 s).
   - Hence its S\* run from −44.7 to −323, LOG_ALL is −0.004, LOG25 −1.264 and NOT25 −96.
   - Its 50 and 75 kW/m² Spearman values are almost cap-free: 58 of 4,380 pairs are capped.
   - SPR25 and SPM25 are tie-affected: 550 of 2,230 pairs are tied at the cap.
   - t2's S\* and LOG\* cells may be quoted only as "degenerate (capped back-transform)", never as values. t2 is above exp046 on LOG_ALL in 0 of 10 repeats, so it is not in the headline.
3. **The headline-template conditions.**
   - **Condition 1: at least 8/10 repeats above exp046 on LOG_ALL.**
     - Met: exp047 10, b1 10, b2 10, et_alone 10, c5_ridge 8, t3 10, t4 10.
     - Not met: t1 4, t2 0.
     - So the main template applies, not the empty case. The executor's filled values match the JSON.
   - **Mechanical fix.** The config's `aggregation` rule says "the headline cell is the 10-repeat mean +- std", but the executor's filled template gives means only. Carry the ± std, at least for c5_ridge (0.418 ± 0.285).
   - **Optional label.** The template names et_alone and c5_ridge. That is allowed, since naming a model is not calling it a candidate. Still, add "(diagnostic only)" after each.
   - **Condition 2: `allowed_interpretation`.**
     - The condition is met literally (et_alone 10/10, c5_ridge 8/10), so the sentence MAY be added.
     - **This critic advises against adding it.** The decision is the orchestrator's or Arnav's, for three reasons:
       - (a) c5_ridge sits exactly on the threshold, and both of its misses come from its negative predictions. One more such repeat would have closed the gate.
       - (b) "separates materials better … on these report-only metrics" claims more than the data shows. At 25 kW/m², et_alone is above exp046 on S25, SPR25 and SPM25 in only 1, 5 and 4 of 10 repeats, and c5_ridge in only 2, 7 and 7. The claim does hold at 50 and 75 kW/m² and on LOG25.
       - (c) LOG_ALL pools all fluxes, so much of it reflects the spread between fluxes rather than separation between materials.
     - If the sentence is used anyway, use the exact declared sentence, placed after the headline, with c5_ridge's `n_clipped` disclosure beside it. No variant and no substitute.
4. **The 0.58 rule.** Nothing in the log or the JSON compares a number with 0.58.
   - **Numbers sitting at the bar:**
     - b1 LOG_ALL 0.579, which appears in the headline itself;
     - et_alone S50_75 0.580;
     - b2 0.572, b1 0.569 and t3 0.566 (S50_75).
   - **Numbers above the bar:** et_alone LOG_ALL 0.633, t3 LOG_ALL 0.629, et_alone S75 0.613.
   - Write b1 as 0.579 and never round it to 0.58. Keep the template's closing sentence, "None of these numbers is compared with the 0.58 bar."
   - Repeat-0 CIs such as et_alone S50_75 [0.434, 0.722] are descriptive and are never compared with any bar.
5. **Step-1 diagnostics** (C5 Ridge 0.2825, ET alone's repeat-0 CI [0.158, 0.463]) are never quoted. The executor's message names 0.2825 as a diagnostic. That is fine in an internal report, but it must not reach any doc.

### Wording and files (`reporting_rules`)
- **No report file was written, which is correct.** The config's outputs are the log and the JSON only, and `not_written` bans report edits by the executor. So there is no executor report file to fix, and **I changed nothing.** The log and the JSON are script output and must not be hand-edited.
- **"Wins".** The executor's message says "wins" for the step-2 win-count checks. That is the config's own step-2 term. In any doc text, though, "win", "beats" and "record" must not be attached to a secondary metric. Say "repeats above exp046", as table (5) does.
- **"Artefacts".** The executor's message calls c5_ridge's misses "artefacts". In any doc, say "negative predictions clipped at the declared floor" instead.
- **No doc quotes a number from this run yet.**
  - No tracked `.md` has changed since 16:26.
  - `reports/model_cards/TTI.md` and `reports/macfp_wood_crosslab.md` cite only prior-look numbers or the definitions.
- **B4 must not draw on this run.** et_alone and t3 look strong at 50 and 75 kW/m², but this run may not be cited to choose, add or drop any B4 candidate. B4 is `tti_round3.json`, committed in dd9c9c6, before these tables existed.

### Process note
**A second A3 critic is running in parallel.** It is agent af09909e812171abb, labelled `critic:A3`, from the relaunched workflow wf_ed8c4600-36e.
- It started at 16:27:17, while the rerun was still writing its log. Its `critic_a3_v2/log_snapshot.log` is a partial log that ends at t3.
- It ran a scratch replication at 16:28. That copy redirects `OUT_JSON`, so it is not a run and does not change the run count.
- If it also writes to this file, the orchestrator should reconcile the two reviews and confirm it compared against the complete log.

### Caveats to carry (rerun)
1. **Quote only through the headline template:**
   - with "report-only, defined post hoc" and the config pointer;
   - in the fixed model order;
   - as mean ± std;
   - with et_alone and c5_ridge marked "diagnostic only".
2. **c5_ridge always carries** its ± std and "9 clipped predictions (repeats 5 and 8, 75 kW/m²)". Never recompute it with another floor.
3. **t2's S\* and LOG\* cells are degenerate.** Never quote them as values.
4. **`allowed_interpretation`** is permitted, but this critic advises against it. The orchestrator or Arnav decides. If it is used, use the exact sentence only.
5. **0.58:** b1's LOG_ALL 0.579 sits in the headline, so never round it to 0.58 and keep the closing disclaimer. et_alone's S50_75 0.580 is never set against anything.
6. **Step-1 diagnostics** (0.2825, [0.158, 0.463]) are never quoted.
7. **Nothing from this run may be cited** for B4, or for un-parking TTI.

## Addendum 2: second fresh-context critic (`critic:A3`, scratch `critic_a3_v2/`), reconciled
Reviewed Oct 6, 16:25–16:45 PDT. The review covers run 1, the rerun, the narrative corrections and the model-card text committed in 43fc65f.

**Verdict: SOUND WITH CAVEATS**, in agreement with both sections above, with three additions:
- **(A)** The model card's new secondary-metrics paragraph has three re-rounded values, one inaccurate sentence and an over-broad t2 clause. **Fix before anything else quotes it.**
- **(B)** A pre-existing bug in `train_tti_v5_physics.run()` explains why exp047's figures in the old logs differ from this run.
- **(C)** A5 and B1–B4 are now verified as exact.

I changed no file other than this one.

### Reconciliation with the process note above
- **I compared against the complete log.** `critic_a3_v2/log_snapshot.log` is a partial copy taken while the rerun was still writing. My comparison used `experiments/runs/tti_secondary_metrics.log` itself after the rerun ended.
- **Results:**
  - My scratch copy (exit 0) is identical to it apart from timings.
  - Its JSON is identical apart from `generated_at` and `runtime_s`.
  - My scratch runs redirect `OUT_JSON` and are not runs of record.
- **Independent code** (my own R², my own Spearman via average ranks, my own masks and medians; only `eval_protocol` shared):
  - **Every headline-template cell matches the JSON mean and std to 4 dp:** the exp046 per-flux and log cells, and LOG_ALL with its paired count for all 8 models in the template.
  - So do 35 further cells: S25, S50, S50_75, LOG_ALL, LOG75, SPR50 and SPM50 for exp046, b2, et_alone, c5_ridge and t2.
  - So do `n_clipped` and three CIs.
- **Run 1's SPR50 diagnosis is now exact.** The A3-protocol exp046 OOF is bit-identical to the skeptic's `skeptic_tti/oof_exp046.npy`: max abs difference 0.0 over 10 × 681.
- **Pre-amendment visibility.** Run 1's step-2 lines printed C5 Ridge's exact counts on five metrics, including LOG_ALL 8/10. So "before any table (2)–(7) was computed or seen" holds for the tables, but that one headline-relevant count was visible. Amendment 2 changes only one exp046 reference value, so it could not have been steered by it.

### (A) Model card, `reports/model_cards/TTI.md` (43fc65f): three fixes (items 1–3)
1. **Three values were re-rounded from the 4-dp JSON instead of taken from the log's 3-dp print.** That is the same double-rounding mechanism that blocked run 1. The config says "Printed to 3 decimals, stored to 4", so the log is the 3-dp source.

| Model | Card | JSON (4 dp) | Log (3 dp) |
|---|---|---|---|
| ET alone LOG_ALL | 0.632 | 0.6325 | **0.633** |
| t3 LOG_ALL | 0.628 | 0.6285 | **0.629** |
| exp047 LOG_ALL std | 0.035 | 0.0345 | **0.034** |

   All other values in the paragraph match the log.

2. **"At 25 kW/m² no model on the shelf is reliably above exp046 (ET alone: 1/10 repeats on seconds R²)" is wrong as written.** Repeats above exp046 at 25 kW/m², from table (5):
   - **S25:** b2 9/10, b1 7/10.
   - **LOG25:** b1, b2, ET alone, C5 Ridge and t3 10/10; exp047 and t4 9/10.
   - **SPR25 and SPM25:** b1 and b2 9/10 on each.

   The sentence is also outside `reporting_rules`, which allow only the template and the declared sentence. **Delete it.** If a 25 kW/m² statement is wanted, a purely descriptive one is accurate: "Seconds R² at 25 kW/m² is at most 0.08 for every model (10-repeat means)."
3. **Narrow the t2 clause.** "t2's secondary cells are degenerate" is too broad. Write "t2's seconds- and log-scale cells are degenerate". Its 50 and 75 kW/m² Spearman cells are almost cap-free (58 of 4,380 pairs capped; see the rerun addendum, item 2).
4. **The rest of the paragraph** follows the template with permitted additions: ± std, exp ids, "(diagnostic only)", the C5 clipping note and the closing disclaimer. It does not use the `allowed_interpretation` sentence. It sets nothing against 0.58 and contains no "beats" or "win".
5. **On `allowed_interpretation`** I take a neutral view: it is permitted (ET alone 10/10, C5 Ridge 8/10), and leaving it out is also fine.
   - The objection above, (b), cites seconds and rank cells at 25 kW/m².
   - The sentence is about these report-only metrics, and on log R² within each flux it holds: ET alone is above exp046 in 10/10 repeats at 25, 50 and 75; C5 Ridge in 10/10 at 25 and 50 and 8/10 at 75.
   - Either choice is defensible. Do not substitute any other sentence.

### (B) Pre-existing bug: `train_tti_v5_physics.run()` drops `log_target` (not A3's code; report only)
- **The bug.** The refit at line 265 is `repeated_group_kfold(model, X, y, g, repeats=1)`, with no `log_target=logt`. Its comment says it "== report()'s repeat-0 OOF", which is false for log-target candidates (C0 = exp047, C1–C5, R0).
- **What it affects.** The lines built from that refit describe a raw-seconds refit, not the scored model: SSE share, "R2 outside the 4 slowest", "OOF R2 in log1p space" and "seconds R2 on 25 kW/m2 rows / other fluxes".
- **What it does not affect.** The headline, CI and family-GKF numbers pass `log_target`. exp046 (C6, no log target) is unaffected. No number of record changes.
- **Checked in scratch** (repeat 0; at 25 kW/m², non-25 rows pooled, log1p overall):

| C0 fit | 25 kW/m² | non-25 pooled | log1p |
|---|---|---|---|
| without log target | 0.2319 | 0.3968 | 0.5355 |
| with log target, as scored | 0.0191 | 0.3724 | 0.5448 |

  The first row reproduces the logged values. This run's exp047 NOT25 repeat 0 is 0.372, which is the correct value for exp047.
- **Where the raw-refit numbers appear as exp047's:**
  - the config's `prior_looks` ("exp047 0.232 / 0.397, log1p 0.536");
  - `tti_narrative_corrections.md` fact 8 ("exp047 0.397"), which is in no applied text;
  - `reports/sprint_0929/tti_replicates.md` line 51, the C0 row (0.41 / 0.245 / 0.232 / 0.397 / 0.536);
  - in `reports/sprint_0929/tti.md`, the "top-4 SSE share", "R2 excl. top 4" and "R2 @25 / other" columns of the log-target rows (C0–C5, R0) in the tables at lines 150–158, 170–176 and 182–190. These columns are produced by the same refit. I did not recompute them;
  - possibly the "-0.14 to +0.23 across all 32 model-by-row-set runs" range in `reports/sprint_0929/tti.md` line 227 (line 223 before the A4 erratum insert; not checked).
- **The fix needs its own decision**, because this is a record-reproduction script. An erratum on `tti_replicates.md` line 51 is the minimum. Do not edit the logs.

### (C) Narrative corrections: all 11 files verified
- **Method.** I rebuilt each file from `git show dd9c9c6:<file>` plus the "Current" → "Replacement" blocks. Every "Current" block matched exactly at its cited line. The rebuilt files are byte-identical to b839892 (and to the working tree) for A2–A5, B1–B4, C1 and C2.
- **The two differences are both deliberate:**
  - `STATUS.md` line 9 carries the extra " (post hoc diagnostic, lever sweep tti-1)", which closes the first review's caveat 3.
  - Model-card line 19 quotes the A5 note verbatim, as the corrections file instructs.
- **Closed:** the first review's caveat 4 ("B1–B4 are still open"). A4(g), C3 and D1 remain unapplied, as recommended.
- **Further numbers checked:** 97.8% of exp046's repeat-0 SSE sits at 25 kW/m² (0.9779), and the Masonite rows run 398–488.5 s.

### Caveats to carry (addendum 2)
1. **Before anything else quotes the model-card paragraph:**
   - change 0.632 to 0.633, 0.628 to 0.629, and ± 0.035 to ± 0.034;
   - delete the 25 kW/m² sentence, or replace it with the descriptive one in (A);
   - narrow the t2 clause to its seconds- and log-scale cells.
2. **exp047's per-flux, log1p and SSE-share figures in the v5 logs are raw-target refits.** Quote this run's values for exp047. Correct `tti_replicates.md` line 51 by erratum under its own decision.
3. **All earlier caveats in this file stand.**
