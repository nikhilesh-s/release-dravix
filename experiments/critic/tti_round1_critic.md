# Critic: TTI round 1 (t1-t4 vs exp046) and BLOCKER-20261004-1156-tti-round1-no-gain
Verdict: **SOUND WITH CAVEATS**. The round ran as declared, exp046 reproduces exactly, and "nothing replaces exp046, park TTI" is correct. Two of the blocker's explanations are overstated, and one is wrong as written. Fix the wording before the blocker is cited.

## Pre-declaration and implementation
- Order: the config was committed at 11:51:11 (bd467e0). train_tti_round1.py mtime is 11:51:41, the log 11:56:10 (runtime 252 s), and results were committed at 12:06 (40f581e). OK.
- The code matches the config:
  - t1 is the v5 linear() pipeline (median impute + indicators + scaler) with PoissonRegressor(alpha 0.01) on PHYS + log_q_excess.
  - t2 is TransformedTargetRegressor(Ridge 1.0, t^-0.5, inverse 1/max(z,0.02)^2) on the 5 declared features.
  - t3 is HGB poisson with v5.HGB on F3 + both CHF terms.
  - t4 is HGB with log1p on F0.
  - q_cr uses sigma(T^4-T0^4) + 10(T-T0), with eps 1, T_ig = baseline-corrected MCC onset (--onset-bc), and NaN when q'' <= q_cr.
  - Selection rule, triggers and family-GKF / R2@25 reporting are as declared. No leakage: every feature is material-level bench data or the set flux, and v5's leakage assert covers F3 and PHYS.
- Deviations from plans/V3_PLAN (the config wins, but record them):
  - The plan's t1 (1/sqrt target, GLM) and t2 (the same with HGB) became a Ridge on 1/sqrt (t2) and a GLM + log(q''-q_cr) (t1).
  - The plan's t3 (per-material q_crit from training folds plus did-not-ignite rows) was dropped. This was correct: under GKF-by-material a held-out material has no training-fold q_crit.
  - The script name differs from the plan (v8_critical).
- Forking paths: t1 and t3 are v5's post-hoc exploratory X1/X2. I found no logged X1/X2 scores, so I cannot tell whether their results were seen before the declaration. This matters little because neither won.

## exp046 reproduction
- GKF 0.3133 ± 0.037, CI [0.071, 0.496], identical to the record. The 681 rows / 76 materials assert passes.

## Blocker claim 1: "0 rows have q'' <= q_cr, so the CHF term is only a smooth reshaping of log q''"
- "0 rows" is correct, but only for the 592 rows / 65 materials that have an onset. 89 rows have no T_ig, and log_q_excess is NaN there (imputed + indicator).
- "Reshaping of log q''" is wrong.
  - q_cr ranges from 3.6 to 21.1 kW/m2 (median 6.1). At 25 kW/m2, q''-q_cr runs from 4.6 to 21.4, a 4.6x material-dependent spread. That is exactly the near-critical effect the term was meant to capture.
  - What is true: log_q_excess is a smooth function of log q'' and log dT, which are both already in PHYS. A linear fit on (log q, log dT) explains 94.6% of its variance (corr with log q alone is 0.93).
  - So it adds no new information, only about 5% of new functional form, and the GLM did not use it (-0.015, 0/10).
- Correct wording: "q_cr from MCC onset is below every tested flux, and log(q''-q_cr) is ~95% collinear with log q'' and log dT already in exp046. MCC-onset q_cr does not explain the 25 kW/m2 failure." This agrees with v5's own check: the Spearman between predicted and observed TTI(25)/TTI(50) ratios across materials is only +0.15.

## Blocker claim 2: "t2 failed by the declared back-transform cap"
- Mostly correct.
  - In z-space (repeat 0 split), the Ridge reaches R2 0.31 on z = TTI^-1/2, about exp046's level. The model itself is not broken.
  - 49 OOF z_hat values are <= 0 and 76 are < 0.02. All 76 are mapped to 2500 s, while the maximum observed TTI is 1102 s (only 2 rows exceed 1000 s). Those 76 rows carry 93% of the SSE.
- But the cap is not the whole story.
  - Excluding capped rows, R2 is still -4.7. A tighter floor helps, but only partly: R2 is -8.0 with z >= 0.03 and -0.41 with z >= 0.05.
  - The inverse-square back-transform amplifies any small-z error. An additive Ridge cannot represent the multiplicative q''/(sqrt(kpc) dT) structure.
- The blocker is right that it was not re-tuned. "Design flaw in the declaration" is fair: the floor sat 2.3x above the data maximum, and z_hat <= 0 was foreseeable.

## Blocker claim 3: "The ceiling is in the data, not the model. Four model families all land at R2 <= 0.05 at 25 kW/m2"
- Overstated.
  - t2 is not a meaningful fourth family.
  - R2@25 is repeat-0 only, on a subset that has a heavy right tail (21 materials with TTI25/TTI50 > 6 hold 59% of that SS).
  - The supportable claim: "no model using the current material-level bench features, including MCC-onset q_cr, predicts the 25 kW/m2 blow-ups". Missing physics (absorptivity, melt/drip, char) is a plausible hypothesis, not a demonstrated one.

## Caveats to carry
1. exp046 stays the record: 0.313 [0.071, 0.496]. Parking TTI is justified.
2. Correct claim 1 in the blocker/STATUS. The CHF term is near-collinear with existing features, not a reshaping of log q'' alone.
3. A future inverse-sqrt target needs a floor on z tied to the data, such as 1/sqrt(max train TTI). A floor fixed at 2500 s is unusable.
