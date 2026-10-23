# Critic: blend round 1 (b1/b2 vs incumbents; CR and FIGRA, the two that pass)
Verdicts:
- **Char Residue b1: SOUND.** It should become the CR number of record: **0.6504 [0.4389, 0.7893]**, replacing exp081 0.5491 [0.342, 0.693].
- **FIGRA b1: SOUND WITH CAVEATS.** The declared rule is met, so b1 becomes the FIGRA number of record: **0.4975 [0.3464, 0.6475]**, replacing exp073. However, the gain is small, it comes from a few materials, and it does not survive selection adjustment.

No SOTA claim is implied for either target.

## Pre-declaration and implementation (Q1)
- Order:
  - The config was committed at 12:01:17 (512a363) and the script at 12:02:09 (83b3427).
  - The log mtimes are FIGRA 12:08:12, CR 12:08:33, TTI 12:08:40 and pHRR 12:09:46. FIGRA's runtime was 333 s and CR's was 355 s, so both runs started at about 12:02:39 and 12:02:38, after both commits.
  - The results were committed at 12:10:00 (8a48ee0). OK.
- Data prep:
  - `data_cr()` is line-for-line the `train_char_residue_round1b.py` loading code, with `--merge-groups` (PlasticC->HIPS), BASE, the 5 category flags and the same HGB.
  - `data_figra()` is verbatim from `train_figra_round1b.py` (row set A, F2, the same HGB with min_samples_leaf 5, log1p).
- Step 1 reproduces exactly:
  - CR: 0.5491 [0.3423, 0.6929] = exp081.
  - FIGRA: 0.4563 [0.2454, 0.6393] = exp073.
  - The script hard-stops if either fails.
- Candidates match the config: ET 500/leaf 2/rs 0, and Ridge alpha 10 after median impute + indicators + StandardScaler. n_jobs=-1 is harmless because the models are deterministic.
- Cosmetic bug (same as LOI r8): the `model` field is written as "Avg(incumbent, blend" in the result JSONs and the leaderboard. Fix it in the tracker regen, not by hand.

## Leakage (Q2)
- The imputer and scaler sit inside the Ridge pipeline. `_fit_predict` clones the Avg per fold, and Avg.fit clones each member again.
- FIGRA: `log_target=True` wraps the whole Avg, so the members are fit on log1p(y), averaged in log space, then expm1'd. That is consistent with the incumbent.
- y, groups and split seeds are identical across candidates. No target-derived feature.

## Replication (scratch, not the repo)
I re-ran the HGB, ET and Ridge OOFs separately (all deterministic). Avg = mean of the members, and it reproduces exactly:
- CR: b1 0.6504, b2 0.6067. Single models: ET 0.618, Ridge 0.631.
- FIGRA: b1 0.49751, b2 0.49748. Single models: ET 0.491, Ridge 0.439.

## CR: impossible values (Q2)
- 6.6% of Ridge OOF predictions are below 0 and 1.3% are above 100. b1 is below 0 for 4.2% of rows. Note that the observed y itself goes down to -4.1.
- Clipping does not matter:
  - b1 clipped to [0, 100]: 0.6505.
  - Ridge clipped before averaging: 0.6483.
  - Both are diagnostics only.

## Robustness (Q3)
Per-repeat dB, b1 - incumbent:

| Repeat | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| CR | +.101 | +.086 | +.097 | +.078 | +.077 | +.181 | +.123 | +.112 | +.093 | +.066 |
| FIGRA | +.035 | +.029 | +.048 | +.017 | +.062 | +.045 | +.057 | +.039 | +.050 | +.030 |

**CR:**
- 43 materials improve and 18 get worse.
- Top contributors: Mineral_Wool_Insulation is 29% of the gain. The incumbent SSE on it was 12573 and b1 brings it to 2865, which fixes the incumbent's third-largest error. The next four are Nylon carpet 9%, FRP 8%, Feather pillow 7% and Rubber foam 6%.
- The two largest incumbent errors (PC, House_Wrap) are not fixed.
- Removing materials from the evaluation set:

| Removed | dB | Wins |
|---|---|---|
| Top 1 | +0.084 | 10/10 |
| Top 3 | +0.066 | 10/10 |
| Top 5 | +0.053 | 10/10 |

- Category hold-out (GKF 5x10 by first FSRI category; 11 categories): incumbent 0.378 vs b1 0.535, +0.157, 10/10. b2 gets +0.098, 10/10.
- The gain is broad. Ridge alone (0.631) beats the HGB incumbent, so the incumbent was underfitting a mostly linear signal.

**FIGRA:**
- 42 materials improve and 25 get worse.
- The top 5 (Roof_Felt, Cotton_Rug, House_Wrap, PE_Foam_Pipe, Cellulose) are 116% of the net gain. The worst is the Overstuffed PU chair, +453.
- Removing materials from the evaluation set:

| Removed | dB | Wins |
|---|---|---|
| Top 1 | +0.034 | 10/10 |
| Top 3 | +0.013 | 8/10 |
| Top 5 | -0.007 | 3/10 |

- Category hold-out: 0.375 -> 0.432, +0.057, 10/10.
- No single huge error is fixed. Roof_Felt, the incumbent's largest error, only goes from 2724 to 2391.
- These removals are diagnostic only (exclusion is banned).

## FIGRA b1/b2 tie (Q4)
- The tie is not exact: b1 0.497507 vs b2 0.497478.
- `max()` picks b1, and would also pick b1 on an exact tie because it is listed first.
- Both satisfy the rule (10/10, ci_lo 0.346 and 0.325, both >= 0.245).
- Record b1 per the mechanical rule. Note that b2 is equivalent, and b2 is the simpler model with no Ridge.

## Selection inflation (Q5)
Method: paired material bootstrap (1000 resamples) of the repeat-mean dB, max-T over {b1, b2}, plus Bonferroni over the 4 targets (8 tries).

| | dB | P(dB <= 0) | Unadjusted LB95 | Adjusted LB |
|---|---|---|---|---|
| CR b1 | +0.101 | < 0.001 | +0.060 | **+0.030** |
| FIGRA b1 | +0.041 | 0.11 | -0.013 | **-0.090** |
| FIGRA b2 | +0.041 | 0.03 | +0.005 | -0.052 |

- CR: the gain survives selection.
- FIGRA: "10/10" counts repeats over the same 67 materials, not independent evidence. Material resampling does not separate b1 from exp073, and after multiplicity adjustment the gain is indistinguishable from zero.

## SOTA wording (Q6)
- CR: ci_hi 0.789 < 0.82 (Pomazi), so even the upper bound is below the bar. No exit trigger, no claim.
- FIGRA: 0.9724 is banned as a comparison (different population).
- The leaderboard rows contain no "SOTA", "beats" or "comparable" text. Keep it that way.

## Caveats to carry
1. CR b1 becomes the record: 0.6504 [0.4389, 0.7893]. It is an HGB+ET+Ridge ensemble. Mineral wool is 29% of the gain, but the gain holds without the top 5 materials (10/10) and under category hold-out.
2. FIGRA b1 becomes the record only because the declared rule says so: 0.4975 [0.3464, 0.6475].
   - The gain depends on 5 materials (3/10 without them).
   - The selection-adjusted lower bound is negative.
   - Describe it as "small, not established".
3. Fix the truncated `model` field via `scripts/update_trackers.py`.
