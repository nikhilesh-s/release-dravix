# Model card: THR (total heat release, MJ/m2)
Source of every number: `experiments/numbers_of_record.json` (exp045) = `experiments/leaderboard.csv` row exp045. No promoted joblib.

**Intended use.** Sanity-check relation THR ~ chi * fuel load * heat of combustion for FSRI products. Not a predictive claim: the inputs nearly define the target.

**Data.** Per-replicate cone table, per material x flux mean of thr_MJm2 over replicates that ignited and were not stopped early (truncated tests kept); rows with flux-adjusted fuel load m''_flux and MCC h_c: 191 rows / 65 materials. FSRI public database.

**Features.** One: phys_MJm2 = m''_flux x h_c (MCC total_hr_bc_kJg).

**Model.** LinearRegression(fit_intercept=False) on that single feature (chi fit on training folds only); 1 fitted parameter.

**Protocol.** Repeated GroupKFold 5x10 by material, pooled OOF R2; material-bootstrap 95% CI.

**Number of record.** R2 0.953 (leaderboard test 0.9530; text 0.9531), CI [0.918, 0.975] (taken from the CSV test_metric_name text). GKF std 0.0024. Legacy GroupShuffleSplit x10: 0.9490.

**Competitor bar.** 0.67 (pomazi_2026, XGBoost held-out; epoxy composites). comparable: false.

**Status (verbatim).** ABOVE BAR, NOT LIKE-FOR-LIKE
Note (verbatim): "Inputs (fuel load, measured heat of combustion) nearly define THR; not a SOTA claim. CI from the exp045 CSV test_metric_name text."

**Known limitations** (record note and leaderboard note; no critic file for THR)
- Inputs nearly define the target; high R2 reflects physics, not learned structure.
- Different population from the competitor (FSRI products vs composites).
- +1 row vs exp040 (Masonite_Board 15 kW/m2); non-igniting replicates now excluded rather than averaged in.
- Needs a measured MCC heat release and a flux-adjusted fuel load; not available for a new product without testing.

**Reproduce.** `OMP_NUM_THREADS=1 python3 data/v2/train_thr_v1_replicates.py` (thin wrapper over `train_thr_v1.py`; no exp045_*.json in experiments/results).
