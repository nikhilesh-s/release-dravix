# Model card: Char Residue (cone residue, %)
Source of every number: `experiments/numbers_of_record.json` (exp081) = `experiments/leaderboard.csv` row exp081 = `experiments/results/exp081_cr_r1b_c2_hgb_inorg_flag_merged.json`. No promoted joblib.

**Intended use.** Estimate cone-calorimeter mass residue of FSRI products from STA/MCC char and product-category flags. Research baseline only.

**Data.** `char_residue_cone_fsri.csv` (spark-ignited, residue present) x STA / MCC / material properties. 567 rows / 62 materials; 61 groups (PlasticC merged into HIPS for grouping, no rows dropped). FSRI public database. STA is N2 only (no air runs).

**Features (11).** heat_flux_kWm2, density_kgm3, mcc_char_yield_pct, sta_res_3K, sta_res_10K, sta_res_30K, plus 5 category flags (interior finish, exterior finish, insulation, roofing, structural) as a coarse inorganic proxy.

**Model.** HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0). Nothing tuned.

**Protocol.** Repeated GroupKFold 5x10 by material; material-bootstrap 95% CI.

**Number of record.** R2 0.5491, CI [0.342, 0.693]. Re-baseline (merged) 0.4831 [0.132, 0.683]. Paired gain +0.068, CI [-0.041, +0.249], 9/10 wins. Family-held-out: 0.378 vs baseline 0.069.

**Competitor bar.** 0.82 (pomazi_2026; epoxy composites). comparable: pending-human.

**Status (verbatim).** SHORT OF SOTA
Note (verbatim, abridged to the caveats): "Promoted 2026-10-02 after critic (experiments/critic/cr_round1b_critic.md, SOUND WITH CAVEATS)... Caveats: paired gain CI includes 0 (+0.068 [-0.041, +0.249]); ~84% of the gain from EPS_Foam_Board; gain is mostly the model swap (flags +0.010)."

**Known limitations** (critic `cr_round1b_critic.md`)
- Gain over baseline is mostly one material (EPS_Foam_Board ~84%); without EPS and Nylon the gain is -0.014 (2/10 wins).
- Do not attribute the gain to the inorganic flags: HGB without flags scores 0.5401; flags add +0.010.
- Flags are a poor inorganic proxy (insulation includes EPS/PIR foam). Underpredicts mineral wool (73.4 vs 96.9).

**Reproduce.** `python3 data/v2/train_char_residue_round1b.py --merge-groups` (code field: `data/v2/train_char_residue_round1b.py`; log `experiments/runs/train_char_residue_round1b_merged.log`).


**Blend round 1 (Oct 6; record now exp117).** Equal average of the exp081 HGB, ExtraTrees (500, min_samples_leaf 2) and Ridge (alpha 10, impute + indicators + scaler), same features and merged groups: **0.650 [0.439, 0.789]**, 10/10 paired wins (+0.101). Critic `experiments/critic/blend_round1_critic.md` SOUND: positive in every repeat, +0.053 without the top 5 materials (mineral wool 29% of the gain), +0.157 with categories held out, multiplicity-adjusted gain lower bound +0.030. Ridge predicts outside 0-100% on ~8% of rows; clipping changes nothing. Below the 0.82 bar. Reproduce: `OMP_NUM_THREADS=1 python3 data/v2/train_blend_round1.py --target CR`.
