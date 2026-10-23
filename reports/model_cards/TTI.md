# Model card: TTI (time to ignition, s)
Source of every number: `experiments/numbers_of_record.json` (exp046) = `experiments/leaderboard.csv` row exp046. No promoted joblib.

**Intended use.** Research baseline for cone time-to-ignition of FSRI products. Parked; not for use as a predictor.

**Data.** Canonical per-replicate cone table (`data/v2/cone_replicates_fsri.csv`) minus 45 did_not_ignite and 9 early_stop rows: 681 rows / 76 materials (replicate level). Left-joined to fixed MCC extras, material properties, thermal conductivity and specific heat. FSRI public database.

**Features.** log q'', log(MCC t_onset_bc_C - 20 C), log(k*rho*c), log cone thickness, log density; missing values -> in-fold median + missing indicators.

**Model.** Physics Poisson GLM: SimpleImputer(median, add_indicator) + StandardScaler + PoissonRegressor(alpha=0.01, max_iter=3000), log link, raw-seconds target. Fixed a priori, not tuned.

**Protocol.** Repeated GroupKFold 5x10 by material, pooled OOF R2 in seconds; material-bootstrap 95% CI.

**Number of record.** R2 0.3133, CI [0.071, 0.496]. GKF std 0.037. Family-grouped 0.356 +- 0.032 [-0.070, 0.566]. Legacy GroupShuffleSplit x10 mean 0.1411 (std 0.451, min -1.178).

**Competitor bar.** 0.58 (pomazi_2026, their ANN result; epoxy composites). comparable: pending-human.

**Status (verbatim).** PARKED, SHORT OF SOTA
Note (verbatim): "Parked per plan B5. Physics Poisson GLM on the canonical per-replicate table. Not good away from 25 kW/m2 either: pooled 50+75 R2 0.368, within-flux log R2 ~0 (0.025 / 0.018 / -0.006 at 25 / 50 / 75); the old 'other fluxes 0.80' pooled every non-25 row and was mostly 3 Masonite rows at 15 kW/m2 (lever sweep tti-1, post hoc, report-only). 25 kW/m2 replicate ceiling (LORO) 0.558. Shelf-wide secondary metrics: experiments/configs/tti_secondary_metrics.json."

**Known limitations** (leaderboard note; no critic file for TTI)
- CI lower bound 0.071; does not reach the bar.
- Beats same-rows C0 on GKF (0.261) and family (0.346), not on legacy (0.162).
- Legacy split instability: Latex_Pillow_Foam (no density or k*rho*c) predicted up to 414 s vs 1.0-21.5 s observed.
- Row set differs from exp039 (0.3475), so not paired with it.

**Per-flux skill (post hoc diagnostic, report-only; lever sweep 2026-10-06, `plans/LEVER_SWEEP_2026-10-06_findings.json` tti-1).** exp046 is not "good away from 25 kW/m²". The often-quoted "R² ≈ 0.80 at other fluxes" (repeat 0: 0.798; 10-repeat mean 0.768 ± 0.070) pools every non-25 row (15, 35, 50, 75 and 100 kW/m²), and 3 Masonite_Board rows at 15 kW/m² (398–489 s) hold 74.5% of that pool's target sum of squares; without them it is 0.294 ± 0.205. Within one flux (10-repeat means) seconds R² is 0.042 at 25, 0.274 at 50 and 0.283 at 75 kW/m², and 0.368 pooled over 50+75 (repeat-0 CI [−0.42, 0.59]). Log-space R² is 0.025 / 0.018 / −0.006, so the model barely separates materials at any single flux; rank skill is moderate (material-median Spearman 0.51 / 0.50 / 0.44). The pooled-seconds headline mostly measures the 25 kW/m² rows, which hold 85.2% of the target sum of squares and 97.8% of exp046's repeat-0 out-of-fold SSE. These numbers are descriptive, were found after the fact, and are never compared with the 0.58 bar. The same metrics for every TTI model on the shelf are declared in `experiments/configs/tti_secondary_metrics.json` (report-only; never promotes).

**The 25 kW/m² ceiling (label-only; tti-3).** On the 223 rows / 69 materials at 25 kW/m², predicting each replicate from its siblings in the same material × flux cell (leave-one-replicate-out) gives R² 0.558 (in-sample cell mean 0.776); at 50 and 75 kW/m² the same ceiling is 0.958 and 0.955. A reference model that already knows each material's own measured 50 and 75 kW/m² ignition times scores 0.29–0.49 at 25 kW/m², depending on how replicates are aggregated, or 0.50–0.62 overall when combined with the leave-one-replicate-out cell means at the other fluxes. That is a reference point, not a strict upper bound, but the pooled-seconds bar (0.58) sits at roughly the level of that oracle. Ignition times read from the HRR curve shrink the 25 kW/m² replicate scatter (90th-percentile within-cell CV 0.48 → about 0.36) but leave it large.

**V3 round 1 (Oct 4; record unchanged).** Four pre-declared candidates (`experiments/configs/tti_round1.json`, `data/v2/train_tti_round1.py`): GLM + log(q″ − q_cr) 0.298; HGB Poisson + critical-flux terms 0.212; HGB log1p 0.267; Ridge on TTI^-1/2 −38 (its declared back-transform blows up). 0/10 paired wins for each; exp046 reproduced exactly. The critical-flux term is ~95% explained by log q″ and log ΔT, which exp046 already has. R² at 25 kW/m² stays ≤ 0.06 for three model families (repeat 0). The weakness is not confined to 25 kW/m² (see "Per-flux skill" above). Critic: `experiments/critic/tti_round1_critic.md`.

**Round 2 (Oct 5; record unchanged).** FSRI integrating-sphere emissivity (`data/v2/emissivity_fsri.csv`, 383/681 rows, 42 materials; thermal-join ban lifted for this source only): GLM + log eps(T_heater) 0.283, + emissivity-corrected critical flux 0.271, HGB 0.157, has-emissivity-only control 0.287. 1/10 paired wins at best. Emissivity shows no signal even on covered materials (critic `experiments/critic/tti_round2_critic.md`). Parked.

**Secondary metrics (A3, Oct 6; report-only, defined post hoc; `experiments/configs/tti_secondary_metrics.json`, log `experiments/runs/tti_secondary_metrics.log` (values as printed there, 3 dp), critic `experiments/critic/tti_secondary_metrics_critic.md` SOUND WITH CAVEATS).** Within a single heat flux exp046 separates materials poorly (seconds R² 0.042 / 0.274 / 0.283 at 25 / 50 / 75 kW/m², 0.368 pooled over 50+75; log R² 0.025 / 0.018 / -0.006). On overall log R², exp047 0.535 ± 0.034; b1 (exp123) 0.579 ± 0.029; b2 (exp124) 0.567 ± 0.028; ET alone (diagnostic only) 0.633 ± 0.023; C5 Ridge (diagnostic only) 0.418 ± 0.285 (8/10, exactly on the threshold; 9 negative predictions at 75 kW/m² clipped to the declared floor, which cause both of its misses); t3 (exp098) 0.629 ± 0.019; t4 (exp099) 0.530 ± 0.038 score above exp046 (0.391 ± 0.047) in at least 8 of 10 repeats. The seconds-scale TTI number of record is unchanged: exp046 0.313 [0.071, 0.496]. None of these numbers is compared with the 0.58 bar. t2's seconds- and log-scale cells are degenerate (capped back-transform) and are not quoted. Not a reason to un-park TTI or to choose any future TTI candidate.

**Reproduce.** `python3 data/v2/train_tti_v5_physics.py --onset-bc --replicate-table` (log `experiments/runs/train_tti_v5_physics_replicate_table.log`; no exp046_*.json in experiments/results).
