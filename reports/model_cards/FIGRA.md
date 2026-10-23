# Model card: FIGRA (fire growth rate, kW/m2/s)
Source of every number: `experiments/numbers_of_record.json` (exp073) = `experiments/leaderboard.csv` row exp073 = `experiments/results/exp073_figra_rebaseline_exp012cfg.json`. No promoted joblib.

**Intended use.** Baseline estimate of FIGRA = pHRR / t_pHRR (from test start) for FSRI products. Not comparable to the published polymer-level result.

**Data.** `data/v2/figra_fsri.csv` replicate mean per material x flux (spark-ignited, ignition recorded) + MCC, MCC extras, material properties; materials with MCC. 197 rows / 67 materials. FSRI public database.

**Features (9).** heat_flux_kWm2, mcc_peak_hrr_Wg, mcc_char_yield_pct, density_kgm3, cone_thickness_mm, cone_mass_per_area_kgm2, mcc_hrc_JgK, mcc_total_hr_kJg, mcc_t_peak_C.

**Model.** HistGradientBoostingRegressor, direct, log1p target: max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=0.

**Protocol.** Repeated GroupKFold 5x10 by material, pooled OOF R2; material-bootstrap 95% CI.

**Number of record.** R2 0.4563, CI [0.245, 0.639]. Previous exp012 0.503 [0.324, 0.663] predates the MCC fix. Two-stage variant (exp072) rkf -2.790, not adopted.

**Competitor bar.** 0.9724 (phan_morgan_2025, 15 polymers, real cone data). comparable: false.

**Status (verbatim).** BASELINE, NOT COMPARABLE
Note (verbatim): "Re-baselined 2026-10-02 (Arnav, option A): exp012's config on post-f063d48 data, 197 rows / 67 materials. Previous exp012 0.503 [0.324, 0.663] predates the MCC fix. Two-stage (exp072) not adopted. Not comparable to phan_morgan 0.9724 (different population). Same-polymer table: reports/figra_same_polymer_table.md"

**Known limitations** (record note; no FIGRA critic file in experiments/critic)
- Different population and task (FSRI commercial products vs 15 pure polymers); do not read the gap to 0.9724 as a deficit or a win.
- Per-polymer comparison (e.g. HDPE 16.72 measured vs 6.61 predicted; Nylon 6.93 vs 13.63) in `reports/figra_same_polymer_table.md`.

**Reproduce.** `OMP_NUM_THREADS=1 python3 data/v2/train_figra_round1b.py` (code field of the results JSON; log `experiments/runs/train_figra_round1b.log`; also rewrites reports/figra_same_polymer_table.md).


**Blend round 1 (Oct 6; record now exp119).** Average of the exp073 HGB (log1p), ExtraTrees and Ridge: **0.498 [0.346, 0.648]**, 10/10 (+0.041). Critic SOUND WITH CAVEATS: the gain is small and not established (top 5 materials carry 116% of it; without them -0.007, 3/10). Recorded per the pre-declared rule. Still not comparable to 0.9724. Reproduce: `OMP_NUM_THREADS=1 python3 data/v2/train_blend_round1.py --target FIGRA`.
