# Model card: pHRR (peak heat release rate, kW/m2)
Source of every number: `experiments/numbers_of_record.json` (exp074) = `experiments/leaderboard.csv` row exp074. Model: `models/v2/phrr_hgb_v4.joblib`.

**Intended use.** Screening estimate of cone-calorimeter pHRR for an FSRI-style product at a given heat flux, from MCC, specimen geometry and density. Not for materials far outside FSRI products (e.g. FR composites), not a certification tool.

**Data.** FSRI cone replicates (`data/v2/cone_replicates_fsri.csv`), replicate mean per material x flux, excluding non-ignitions and early-stopped tests; joined to MCC, material properties, HFM. 191 rows / 65 materials; 64 groups (PlasticC aliased to HIPS for grouping only, no rows dropped). Provenance: FSRI public database, ingest scripts in `data/v2/ingest_*.py`.

**Features (10).** heat_flux_kWm2, mcc_peak_hrr_Wg, mcc_char_yield_pct, cone_mass_per_area_kgm2, cone_thickness_mm, density_kgm3, mcc_total_hr_kJg, mcc_t_peak_C, mcc_t_onset_C (baseline-corrected), log_kpc (NaN allowed).

**Model.** HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, random_state=0) on log1p(pHRR), predicts kW/m2. Fixed settings, not tuned.

**Protocol.** Repeated GroupKFold 5x10 by material (merged groups), pooled OOF R2; 95% CI = material bootstrap (2000, seed 0) of repeat 0.

**Number of record.** R2 0.6113, CI [0.516, 0.838] (leaderboard 0.5155-0.8376). Pooled-repeat CI [0.435, 0.822]. Category-grouped (family held out): 0.392 [-0.149, 0.529]. Paired vs exp043 RF on same merged splits: +0.065, 8/10 repeats (exp043 merged: 0.546).

**Competitor bar.** 0.3 (pomazi_2026, held-out RFR R2; epoxy composites, random split). comparable: pending-human.

**Status (verbatim).** AHEAD, CI CLEAR, DIFFERENT POPULATION
Label (verbatim): "ahead of the published number on a material-grouped protocol (harder than the competitor's random split), different population (FSRI products vs epoxy composites); NOT ahead when whole material families are held out (0.392 [-0.149, 0.529])."

**Known limitations** (record note + `experiments/critic/phrr_round1_critic.md`, verdict SOUND WITH CAVEATS)
- Not shown to improve on exp043 (paired gain CI includes 0 in the critic's run); c1/c2/c4 statistically indistinguishable.
- CI clears 0.30 on material-grouped GKF only; every family-grouped CI straddles 0.
- Gain over exp043 is concentrated in House_Wrap, PETG, HDPE.
- In-sample rows (e.g. demo lookups) are not held-out predictions. Needs thickness and mass per area of the specimen.

**Reproduce.** `python3 data/v2/train_phrr_promote_c2.py` (writes experiments/results/phrr_promote_*.json and the joblib; log `experiments/runs/train_phrr_promote_c2.log`). Demo: `python3 demo/predict.py --material PMMA --heat-flux 50`.
