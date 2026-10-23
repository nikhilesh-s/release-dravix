# Critic: A2 MaCFP-4 pine + Edinburgh PMMA external cross-lab check (report-only)
Verdicts:
- **Run: SOUND WITH CAVEATS.**
  - The script follows `experiments/configs/macfp_wood_crosslab.json` item by item. The one departure is the output location, which the orchestrator ordered and which is disclosed.
  - Step 1 reproduces all three records exactly. My scratch rerun reproduces every output: three files are byte-identical, and `tests.csv` differs only by about 1e-14 from ExtraTrees.
  - Labels and headlines also reproduce from the raw CSVs and READMEs, independently of the script.
  - No MaCFP or Edinburgh row enters any fit, and no exclusion depends on an outcome.
- **The caveats are about what the numbers mean, not about how they were computed.**
  - TTI R2 0.5242 is mostly two IMT tests at 25 kW/m² (81% of the SSE).
  - pHRR R2 0.2156 depends on which labs are in the pool (leave-one-lab-out ranges from −0.20 to 0.40).
  - Edinburgh is close to in-sample at 25 kW/m².
- **Report wording: SOUND.** It uses the quote-as label, makes no bar comparison, uses no win, SOTA or "validated" language, and states its limits plainly. One suggested edit to summary line 5 is below.

No SOTA claim is made or implied. No number here is set against 0.30, 0.58 or 0.82, and no number of record changes.

## Pre-declaration and order (Q1)
- **Order:**
  - The config was committed at 14:52:15 (dd9c9c6).
  - The script was created at 15:04:35 (file birth time).
  - First-run outputs were created between 15:05:24 (log) and 15:06:22 (summary.json, report).
  - The script was last modified at 15:07:15.
  - The second run wrote its outputs between 15:08:59 and 15:09:14. The hand-written summary was added to the report at 15:09:59.
  - Every run came after the config commit. OK.
- **The script was edited between the two runs.** The executor says only the report markdown changed, and that features.json and summary.json were byte-identical across runs.
  - The first-run outputs were overwritten, so that claim cannot be checked independently.
  - It does not matter for the result: I audited the final script against the config (below), and every number reproduces from it.
  - Next time, keep the first-run log.
- **Environment:** the log shows python 3.12.4, sklearn 1.7.1, numpy 2.2.6, pandas 2.3.3 and scipy 1.16.1, with OMP_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. These match the config.
- **Record files are untouched:** `git diff` is empty for numbers_of_record.json, leaderboard.csv and the xlsx files, and nothing was written to `experiments/results/`. No A2 number appears in STATUS, the model cards or the plans.

## Script vs config, item by item (Q2)
| Config item | Script | OK |
|---|---|---|
| source.pin_check | `rev-parse HEAD` must equal 0e4ad8b and `status --porcelain` must be empty, else block. I confirmed both | yes |
| readme_parsing | `normalize_name` is character-identical to `Cone_analysis.py` lines 534–549 at the pin (diff -w is empty). The mass and t_ign header rules match `get_reported_initial_mass` and `get_reported_ignition_time`. The name column is the union of MaCFP's two rules. Each test must match exactly one README row with all required values, else block. That is stricter than MaCFP's first match | yes |
| labs (m0, L, planform, t_ign, flameout, A_norm, orientation) | As declared for all 10 labs. UQ L = 25.3 mm from the README bullet; the table value is final thickness (2.15) and is not used. UMET planform comes from 'Solid size'. TUBS orientation comes from 'Grain orientation (-)'. FZJ mass = channel + README m0, as in MaCFP `read_cone_data`. The A_norm values equal MaCFP `get_cone_area` at the pin | yes |
| UCantabria / BUWfd / Gasification | UCantabria is not in LABS; its folder holds only TGA files. BUWfd 50 kW R1 is dropped (MaCFP line 45 drops it too), and the other 5 BUWfd tests appear only in the end-of-test table. `*Gasification*` files are filtered out | yes |
| row_sets / count_rule | All 9 declared counts are asserted, with a block on any mismatch: 80, the by-lab, by-flux and by-orientation counts, 18/62, tti_25 FSRI 6 + IMT 5, UMET 6, cr_flameout 53, end_of_test 89 | yes |
| mcc_curve_algorithm | Same arithmetic as `ingest_fsri_props.ingest_mcc_extras` lines 153–182: raw peak, `edge = n//20`, linear baseline between edge medians, trapezoid `cum_bc`, onset at 5%, same rounding. Files are opened with utf-8-sig. The self-test on Pine_Siding R1 gives 76.378 / 376.3 / 12.776 / 231.6, equal to the stored row | yes |
| mcc_ps_check | The 12 declared files. β comes from the header (NIST) or else the file name. Per lab: median. Reference: median of the 5 lab medians. Pass: ±10% on both HRC and h_c. The FZJ, TUBS and other-lab consequences are coded. No rescaling | yes |
| features.mcc | Peak = mean of FZJ raw 30K R1–R2 and TUBS 30K R1–R3 = 66.58 W/g. T_peak, total HR and onset come from FZJ raw only. Onset is the mean for exp074 and the median for exp046. Char yield comes from the README masses, and the parser asserts 4.0/0.64 and 4.01/0.54. The label-free table of all 88 Wood MCC files includes the FZJ processed files | yes |
| geometry G1 / G2 | Density = m0 / (planform·L) and mass/area = m0 / A_norm. G2 = medians of geometry over primary_80 (label-free) | yes |
| log_kpc primary / S1 | Primary is NaN. The exp046 imputer is asserted to carry a log_kpc indicator (it does). S1 = log(mean k · mean ρc · 1e6) over the 4 asserted TPS rows = 10.9505 | yes |
| sta_for_cr / category flags | The 7 declared FSRI STA files, last row / first row (the files hold absolute mg; I checked the first rows), the [−5, 105] gate, and the mean per rate. The 750 °C values are reported only. Primary flags are {structural, exterior finish}; S3 sets all 5 to 0 | yes |
| tti_phys_wood / edinburgh_tti | Column order is [log_q, log_dT, log_kpc, log_L, log_rho] with T0 = 20 °C. The Black_PMMA features come from `v5.df`, the same cached module that `data_tti()` imported with `--onset-bc --replicate-table`, so they are exp046's own frame. Uniqueness is asserted. log_L = log(0.006) | yes |
| sanity_gates / freeze | All 15 gates. features.json and its sha256 are written before step 4 | yes (wording nit, Q3) |
| models | exp074: the saved joblib, with its features asserted; a refit on the 191 rows gives max \|diff\| 0. exp046: `v5.linear(PoissonRegressor(0.01, 3000))` fit on 681 rows. exp117: `Avg([HGB, et(), ridge()])` fit on 567 rows, predictions unclipped | yes |
| step1_reproduce | ±0.002 on the mean and both CI ends, with the repeat-0 CI. Groups 64 / 76 / 61, as in `train_blend_round1` | yes |
| variants / reporting | S1 and S2 for pHRR and TTI, S2 and S3 for CR. Each changes one thing. Secondaries get headline pools plus the UMET MAE and bias only, with no per-lab, per-flux or per-orientation tables | yes |
| labels | pHRR = raw max over finite HRR rows. TTI = README t_ign as reported. CR = `np.interp` at the README flameout divided by README m0, plus a channel-m0 column. End-of-test = last finite mass / m0 (BUWfd uses the channel median over Time ≤ 5 s). Duration = last Time | yes |
| scoring | Every declared headline, table, stratum and extra is present. The lab-cluster bootstrap (2000 resamples, seed 0) is applied to the 80 and 62 pools only. R2 to 4 dp; kW/m², s and % to 1 dp | yes |
| outputs | The 4 declared files have the declared columns, plus 4 extra columns (`declared_sampling_s`, `channel_m0_g`, `in_cr_flameout`, `has_finite_mass`). **They are in `experiments/runs/macfp_wood_crosslab/`, and the log is `macfp_wood_crosslab.log`.** | departure: ordered by the orchestrator, disclosed in the docstring and the report |

- `summary.json` `counts` are literals, but each one is asserted earlier, and the 50 and 47 pools follow from the asserted per-lab cr_flameout counts. Cosmetic only.

## Leakage (Q3)
- **Fits:** the script calls `.fit` three times, on the FSRI frames Dp (191 rows), Dt (681) and Dc (567). No MaCFP or Edinburgh row enters any fit. Step 1 uses `repeated_group_kfold`, which clones per fold, and reproduces the records exactly.
- **Inputs are label-free:** README m0, L and planform; flux from the file name; MCC, STA and TPS values. G2 uses geometry only.
- **Freeze wording:** strictly, step 0 has already parsed README t_ign and t_flameout and loaded the cone CSVs before the freeze.
  - The config requires this: t_ign must be present, and the cr_flameout rule compares t_flameout with the end of the mass record.
  - No value derived from these reaches an input.
  - So the log's "[no label column read yet]" should read "no label value used yet". This does not affect validity.
- **Edinburgh is a known material, and the 25 kW/m² half is close to in-sample.**
  - FSRI Black_PMMA is in exp046's training rows: 14 rows at 25/50/75 kW/m², 8.4 mm thick, with a 25 kW/m² mean of 110.0 s.
  - The Edinburgh 25 kW/m² input differs from those rows only in log_L (6 mm vs 8.4 mm).
  - 65 kW/m² is an interpolation between the training fluxes 50 and 75.
  - The report's label "known material (probable), new lab" is correct. Add "near in-sample at 25 kW/m²" wherever R2 0.9424 is quoted.

## Replication (scratch, not the repo)
- **Setup:** I copied the script to scratch. The diff is 3 lines: `HERE` is pinned to the repo's `data/v2`, and `OUT` and `REPORT` point to scratch. Run settings: OMP_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. Result: status ok in 34 s.
- **Byte-identical:**
  - features.json (sha 8f2831f2…, the frozen sha in the log)
  - ps_check.json (f520bba5…)
  - summary.json (5bda0e17…)
  - the report body below the summary markers
- **tests.csv** differs only in `pred_cr_primary` (max 7e-15) and `pred_cr_S3` (1e-14), from ExtraTrees `n_jobs=-1` summation order. For the same reason, the pre-label tests.csv sha in the log (e8c2c82f…) is not bit-reproducible. That is harmless.
- **Step 1 in my rerun:**

| Record | numbers_of_record | Reproduced |
|---|---|---|
| exp074 | 0.6113 [0.516, 0.838] | 0.6113 [0.5155, 0.8376] |
| exp046 | 0.3133 [0.071, 0.496] | 0.3133 [0.0715, 0.4958] |
| exp117 | 0.6504 [0.4389, 0.7893] | 0.6504 [0.4389, 0.7893] |

- **Checks independent of the script:**
  - From the raw CSVs I recomputed obs pHRR (86 tests), end-of-test residue (89) and residue at flameout (53). There are 0 mismatches.
  - I read README rows by hand for FSRI, IMT, Aalto, FZJ, UDRI, TUBS, FPL, UQ, UMET and all 12 Edinburgh rows. t_ign, m0, L and t_flameout match tests.csv.
- **Headlines recomputed from tests.csv** with `eval_protocol.bootstrap_ci` (lab groups, 2000 resamples, seed 0). All match to 4 dp:
  - pHRR: 0.2156 [−1.8485, 0.6019] on 80, 0.6375 on FSRI-lab 18, 0.0496 [−2.6566, 0.6209] on the other 62.
  - TTI: 0.5242 [−0.5670, 0.7851] on 80 (log R2 0.7283), 0.8513 on FSRI-lab 18, 0.4644 [−1.9213, 0.4972] on the other 62, and −0.3593 on tti_25.
  - S1 and S2: all values match.
  - UMET: pHRR +97.9, TTI +23.7.
  - 86 tests: 0.5061. Edinburgh: 0.9424 / 0.8988.
  - CR MAE / bias: 5.8 / +1.6 on 50, 6.1 / +2.1 on 53, 4.4 / +3.5 on 47. Per-lab pHRR biases also match.
- **PS context (not part of the gate):**
  - HRC: computed UDRI is 1100.7–1166.1 W/g against its self-report of 1099–1166, and TUBS is 1109 J/gK against 1099.
  - h_c: TUBS is 40.6 kJ/g against its self-report of 38.8 (+4.6%), and UDRI is 40.6–40.8 against 41.2–41.8.
  - These differences come from the shared baseline-correction algorithm and stay well inside ±10%.

## Exclusions (Q4)
- Every excluded or separated row is named a priori in the config: BUWfd 50 kW R1, UCantabria, the Gasification files, the UMET stratum, the cr_flameout rule, and BUWfd being limited to end-of-test.
  - The two FZJ tests with all-NaN mass stay in the pHRR and TTI pools.
  - primary_80 is every cone file of the 8 labs. Nothing else is dropped in code.
- MaCFP's own outcome-based averaging exclusion (IMT 25 kW R2 and R5, chosen by ignition time) is **not** applied. That is correct, and it matters for TTI (Q5).

## What the headlines mean (Q5)
These are critic diagnostics computed from tests.csv. They are not results, and none may be quoted as a number for A2.

**TTI**
- IMT 25 kW R2 (observed 304 s, predicted 94 s) and R5 (observed 288 s, predicted 88 s) carry **81% of the 80-test SSE** (42.6% + 38.8%).
- On the 69 tests at 30–75 kW/m², R2 in seconds is −0.07 and log R2 is 0.47.
- So R2 0.5242 mostly says that ignition at 25 kW/m² is slow. The per-flux and per-lab tables are the result, as the config's interpretation says.
- Leave-one-lab-out R2 ranges from 0.46 (without FSRI) to 0.64 (without IMT).

**pHRR**
- No single test dominates the error. The largest lab × flux shares are FPL 50 (18%), IMT 75 (16%) and IMT 50 (12%).
- Leave-one-lab-out R2 ranges from **−0.20 (without FZJ) to 0.40 (without IMT)**.
- For this material, exp074 outputs three levels: about 193 kW/m² up to 30, about 276–288 at 35–60, and about 352–366 at 75 and above.

**Lab-cluster CIs**
- They come from 8 clusters (7 for the 62-test pool), so the percentile bootstrap is coarse. Read them as a width indicator only.
- The point estimates sit at the 53rd–72nd percentile of their bootstrap distributions, so there is no sign of a computation problem.

**FSRI-lab 18 ("new material, same lab")**
- MaCFP pine burns lower than both FSRI training softwoods at FSRI. Observed means at 25/50/75 kW/m² are 169/236/322 kW/m².
- The FSRI DB training targets are 259/313/376 for Pine_Siding and 290/276/338 for Wood_Stud. These are context only.
- The +33 kW/m² FSRI-lab bias therefore reflects the material, not the lab.

**Secondaries are not choices**
- S2 has the higher pHRR R2 (0.3583 vs 0.2156), and S3 has the lower CR MAE (4.3 vs 5.8).
  - The config bans selection on any MaCFP row.
  - Never quote S2 or S3 as an improvement, and never use them to change the G1 geometry rule or the CR category flags.
- S1's TPS log_kpc (10.95) is above FSRI's HFM values for its two softwoods (10.48 and 10.51) and above the GLM imputer median (10.83). That is the declared instrument-mismatch caveat.
  - S1's pHRR R2 of −1.09 shows that exp074's output for pine is very sensitive to log_kpc, so the primary result depends on how the HGB routes NaN.
- **Label-free check:** with the FZJ-only MCC peak (72.3 W/g instead of the pooled 66.6), every pHRR prediction would rise by about 9.5 kW/m². The declared TUBS pooling slightly reduces the over-prediction and does not change the reading.

**Char residue (diagnostic)**
- At FSRI-lab, 25 kW/m², exp117 predicts 26.8% against an observed 17.7%. This is what exp117 learned from the FSRI DB, whose end-of-test residue at 25 kW/m² is 31.9% (Pine_Siding) and 29.6% (Wood_Stud), against 13.5–18.2% at 50–75.
- So the stop-time issue already sits inside the training labels. That supports the report's warning, and it is still never grounds to change the CR definition.

**Stop-time claim (summary line 5)**
- The quoted extremes, FZJ 0.8% and UQ 44.4%, come from the two labs with mass-channel caveats:
  - FZJ's mass is reconstructed from a tared channel, and its residues go down to −0.3%.
  - UQ's mass channel is flagged in its README.
- The duration effect holds without them. On the other 66 tests, test duration and end-of-test residue have Spearman ρ −0.39 (p 0.001), with residues of 3.4–28.4%.
- **Suggested edit:** add "(3.4–28.4% per test without FZJ and UQ; residue still falls with test duration)".

**Config-internal tension**
- `row_sets.tti_86_secondary` pools UMET with the oxygen-consumption labs, which the config's own `never` list forbids.
- The config declares it as "a reported aggregate only". The executor followed the config and labelled it.
- **Do not quote 0.5061 outside the report.**

## Report wording (Q6)
- The report contains no "SOTA", "win", "beats", "validated" or "comparable". The only "0.58" match is inside "40.58 kJ/g".
- The quote-as label appears in the hand summary, the report header and the `summary.json` note.
- exp054's 0.588 is shown only with its exp043 label and is never pooled.
- Limits are stated:
  - one material, so R2 mostly measures flux response plus lab bias
  - TTI time zeros are not harmonised
  - 5 s sampling at Aalto and IMT
  - moisture and desiccation
  - the UQ flag
  - Edinburgh uses only two flux levels
- I checked every number in the 5-line hand summary against summary.json and tests.csv, and all are correct:
  - 0.2156 / 0.6375 / 0.0496, +39.1 / 44.3, and FPL +83.1, IMT +72.7, UDRI +53.3, FZJ +8.3
  - the 193–201 / 276–289 / 349–366 levels
  - TTI 0.5242 / 0.7283 / 0.8513 / 0.4644 / −0.3593 / −43.7, and Edinburgh 0.9424 / −7.0
  - CR 5.8 / +1.6 and 4.4 / +3.5, and FSRI-lab 26.8 vs 17.7 and 16.2 vs 16.9
  - end-of-test 0.8% to 44.4%, and FZJ durations of 3124–10770 s (52 min to 3.0 h)
- All markdown tables have consistent column counts, so the `|` fix holds.
- No README states a stop criterion, so "not stated" in the stop-rule column is accurate.

## Executor files
- **I changed nothing.** I found no clear mechanical error. The freeze-wording nit is in a run artifact, so it stays as written. Editing the script now would make it disagree with the log.
- **Output-path departure:** record it in the A2 commit message and here. Do not amend the committed config after the run: it binds from its commit, and the tracker ignores subdirectories in either location, so the location does not change any number.
- None of these files is gitignored, so all 7 can be committed.
- Not A2, so keep them out of the A2 commit: `data/v2/conformal*.py`, `experiments/runs/conformal*`, `models/v2/phrr_hgb_v4_conformal.joblib`, `data/v2/report_tti_secondary_metrics.py`, the A3 files, and the modified docs.
- The A3 narrative-corrections check does not apply to this item.

## Leaderboard row (Q7)
**Recommendation: no leaderboard row.** Reasons:
- It is report-only and nothing is a candidate.
- It covers one material.
- Its R2 CIs span roughly 2.5 R2 units ([−1.85, 0.60] for pHRR).
- `update_trackers.py` would label it rgkf5x10.

If a row is wanted for continuity with exp054:
- Hand-write it as "external cross-lab R2, MaCFP-4 pine (one material, 8 oxygen-consumption labs), report-only", with a pointer to the per-lab tables.
- Never write it through the tracker.
- Never put S1, S2, S3, the 86 aggregate or Edinburgh in it.

## Caveats to carry
1. Quote only as "external cross-lab check, MaCFP-4 pine (one material, 8 oxygen-consumption labs), report-only", and always together with the per-lab and per-flux tables.
2. pHRR: R2 0.2156 [−1.8485, 0.6019].
   - It over-predicts at every lab, by +39.1 kW/m² on average. The bias is largest at FPL, IMT and UDRI and smallest at FZJ.
   - The R2 depends on lab composition (leave-one-lab-out −0.20 to 0.40).
   - exp074 outputs three flux-step levels for this material.
3. TTI: R2 0.5242 [−0.5670, 0.7851] mostly reflects the 25 kW/m² level. Two IMT 25 kW tests carry 81% of the SSE. At 25 kW/m² the predictions are on average 43.7 s early, and lab time-zero definitions differ.
4. Edinburgh 0.9424 is a known material at two flux levels, near in-sample at 25 kW/m². It is never pooled and never placed next to the exp054 PMMA 0.588 except with labels.
5. CR is a diagnostic against residue at flameout, never R2 and never a CR target.
   - The 25 kW/m² over-prediction mirrors the FSRI DB's own end-of-test residues.
   - The stop-time table is a warning about the definition, never grounds to change it.
6. S1, S2, S3 and the 86-test aggregate stay inside the report. None of them is evidence for changing an input rule.
