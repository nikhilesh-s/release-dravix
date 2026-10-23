# Model card: Pomazi reconstruction (pHRR, TTI, THR, Char Residue)
Source: `experiments/numbers_of_record.json` "pomazi_recon" = leaderboard rows exp076-079. No promoted joblib.

**Intended use.** An in-house stand-in for the Pomazi, Magyar and Toldy (2026) fibre-reinforced thermoset population, to learn what their protocol rewards. Not their data, not a claim about their models.

**Decision (verbatim).** "Arnav 2026-10-02: reconstruction does NOT count as the competitor's population until Toldy's real data arrives. Status wording: 'AHEAD ON RECONSTRUCTION, NOT ESTABLISHED'. Not a SOTA win."

**Data.** `data/v2/pomazi_recon_inputs.csv` (from `pomazi_recon.csv`, inputs derived by `derive_pomazi_inputs.py`): literature rows 2017-2023 (pz2023 Table 1, thesis, polym2018, pds2020, jtac11770), hemp excluded, duplicates dropped. 72 rows for pHRR, 60 for the others; 5 resin-system groups for the primary protocol; formulation groups by tolerance 0.1.

**Features (14).** 4 ratios (epoxy, hardener, FR1, FR2), 8 composition (C, H, O, N, P %, aliphatic/cycloaliphatic/aromatic %), heat_flux_kWm2, fibre_wtpct_input.

**Models** (`data/v2/train_pomazi_round1b.py`, fixed). pHRR, TTI: RandomForest(500, min_samples_leaf=2). THR: sklearn GradientBoosting(500, lr 0.01, depth 3, subsample 0.9, max_features 0.9; stand-in for their XGBoost). Residue: HGB(lr 0.05, 300 iter, 7 leaves, min leaf 5).

**Protocol.** Primary: leave-one-resin-system-out, pooled OOF R2, formulation bootstrap CI. Secondary: formulation-grouped GKF 5x10 pooled CI. "Their protocol A": random 80/20 x50.

| target (exp) | primary holdout R2 [CI] | secondary pooled CI | their protocol A | bar | status (verbatim) |
|---|---|---|---|---|---|
| pHRR (exp076) | 0.448 [0.136, 0.578] | [0.382, 0.734] | 0.604 | 0.3 | AHEAD ON RECONSTRUCTION, NOT ESTABLISHED |
| TTI (exp079) | 0.336 [-0.041, 0.690] | [0.491, 0.793] | 0.615 | 0.58 | AHEAD UNDER THEIR PROTOCOL ONLY, ON RECONSTRUCTION |
| THR (exp078) | 0.679 [0.486, 0.792] | [0.753, 0.914] | 0.828 | 0.67 | AHEAD ON RECONSTRUCTION, NOT ESTABLISHED (helped by fibre/heat-flux split) |
| Char Residue (exp077) | 0.695 [0.559, 0.812] | [0.712, 0.902] | 0.807 | 0.82 | BEHIND |

**Known limitations** (`experiments/critic/pomazi_round1_critic.md`, round 1; round 1b fixed the grouping bug and dropped hemp)
- THR is largely fibre/heat-flux separation: a 2-feature GBR reached 0.677; within 25 kW/m2 R2 was -0.03; without pz2023 it fell to 0.494.
- TTI pooled CI sat on the bar; TTI primary CI here includes 0. Bars are best-of-3 algorithms on one split plus 4 external samples.
- pHRR depends on the PER-base group (dropping it cost 0.125). Heat flux is a near-identifier of system.
- Roughly 3 composites probably not in their 75 rows; their per-sample targets are unpublished. Ours are a 50-split mean on a partly different population.

**Reproduce.** `python3 data/v2/derive_pomazi_inputs.py` then `python3 data/v2/train_pomazi_round1b.py` (results exp076-079 JSON; log `experiments/runs/train_pomazi_round1b.log`).
