# Critic: LOI round 8 (r8a-r8d vs exp108, snapshot 2)
Verdict: **SOUND WITH CAVEATS**. The round ran as declared, exp108 reproduces, there is no leakage, and the adoption rule was applied correctly. r8d **may become the LOI number of record (0.425 [0.312, 0.552])** under the pre-declared rule. Three caveats must travel with it:
- The gain is concentrated in a few papers. M079 alone is 31% of it, and without M079 the result is 7/10.
- Protocol C regresses. All of the regression comes from Ridge extrapolating onto held-out P-additive / intumescent classes.
- 9/10 includes two near-ties (+0.003, -0.001).

## Pre-declaration and implementation (Q1)
- Order:
  - The config was committed at 21:39:36 (2a1e251) and the script plus the `--script` option at 21:40:15 (9781304).
  - The log mtime is 21:48:02 and the script's runtime was 456 s, so the run started at about 21:40:26, after both commits. Results were committed at 21:48:17 (d88b8e4).
  - The working-tree script is identical to 9781304. OK.
- Prefix: a diff against train_loi_round6.py shows the code is verbatim up to the round-6 fit section. Only the docstring and the fit/summary/JSON block differ.
- Step 1: the exp092 config re-fit inside the round gives B 0.3775, CI [0.2887, 0.5114]. That equals exp108, and the round has a hard stop if it does not. OK.
- Candidates match the config text:
  - ET 500 trees, leaf 2, rs 0.
  - Ridge alpha 10 in the pipeline Median imputer (add_indicator) -> StandardScaler -> Ridge.
  - Equal-weight Avg of RF (exp092 config), ET and Ridge.
  - n_jobs=-1 is not in the config. It is harmless because the models are deterministic with rs 0.
- Ridge alpha 10 is not justified in the config. Round 1 used alpha 1.0, and I found no log, result or config of any off-book alpha search. I can't rule out such a search, but there is no evidence of one.
- Selection = max B mean over r8a-r8d. r8d has 9/10 strict wins and rep-0 ci_lo 0.3122 >= 0.2887, so it is ADOPT pending critic. The rule is applied exactly as written: strict `>` per repeat, and rep-0 CI compared with rep-0 CI.
- Cosmetic bug: `model=DESC[cid].split(" ")[0]` writes "mean(RF" (r8c/r8d) and "Ridge(alpha" (r8b) into the result JSONs and leaderboard rows. Fix this in the next tracker regen, not by hand.

## Leakage (Q2)
- Imputer and scaler are inside the Ridge pipeline. Every fold refits a `clone()` (ep._fit_predict), and Avg.fit clones each base estimator again, so no fold sees test-fold medians or scales.
- y, g and dom are identical across candidates. The fold seeds are identical (GroupKFold shuffle rs 0-9), and the A protocol uses the same 50 seeds.
- F7 class priors are medians over all 544 FR mentions, test papers included. They are label-free (no LOI), transductive only in the unsupervised sense, and identical for the incumbent. Not a concern.
- No LOI-derived feature. Same 51 exp092 columns.

## Replication (scratch copy, not the repo)
- I rebuilt snapshot 2 (`--build-only --keep`, 571 rows / 102 papers) and ran the round-8 prefix plus my own suffix. That gave OOF predictions for RF, ET and Ridge separately, for both B and C.
- Single models reproduce:
  - RF: B 0.3775, C 0.237
  - ET: B 0.3663, C 0.308
  - Ridge: B 0.2350, C -1.359
- Every base model is deterministic, so the Avg OOF is the mean of the base OOFs. The blends reproduce: r8c 0.4052, r8d 0.4249.

## Is the gain robust? (Q3)
Per-repeat dB, r8d - exp108:

| Repeat | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| dB | +.041 | +.078 | +.033 | +.139 | +.048 | +.044 | +.003 | +.019 | -.001 | +.069 |

- Two repeats are within the RF tie-break noise of about 0.005 estimated in round 6. "9/10" could as easily be "8/10" or "10/10". It is not fragile with respect to the ≥ 8 rule, but 9 overstates the precision.
- The paired paper-bootstrap 95% CI of the mean dB is [+0.008, +0.092], with P(dB <= 0) = 0.003.

Per-paper dSSE (mean over repeats; total -318 = +0.047 R2):
- 61 papers improve and 41 get worse.
- The top 5 (M079 -98, M087 -54, M015 -43, W005 -23, N077 -23) are 76% of the net gain. The top 10 are 103%.
- Removing papers from the evaluation set:

| Papers removed | dB | Wins |
|---|---|---|
| M079 | +0.034 | 7/10 |
| Top 3 | +0.022 | 7/10 |
| Top 5 | +0.015 | 7/10 |

- M079 is the same paper the round-6 critic found driving c3 vs c4. In the old typing, its "unknown" polyol / reporting style acted as a cluster label. Where RF is biased on a paper-level offset, a linear model fixes it. With M079 out, the rule would fail (7/10). This is diagnostic only; acting on it is banned (exit condition 5).

## Is it Ridge extrapolation? (Q4)
- **In B, no.**
  - Only 0.6% of Ridge OOF predictions fall outside the observed LOI range [17.0, 36.7].
  - The 15 rows where |Ridge - RF| > 10 (papers M032, M046, W006) contribute -7 of the -318 total dSSE.
  - Clipping Ridge to the y range would *raise* r8d to 0.433. That is a diagnostic, not a candidate.
  - The gain comes from Ridge's smooth, additive correction on in-range papers where both trees are biased. M079, M087 and M015 each improve under Ridge alone.
  - Ridge's big failures (M046 +483 SSE, M063 +154, M001 +103) are mostly damped by the 1/3 weight.
- ET also contributes. RF+ET without Ridge (undeclared, diagnostic only) gives B 0.397, 9/10.
- Ridge adds about +0.028 on top of RF+ET.

## Protocol C drop (Q5)
- r8d C is 0.195 vs 0.237 for exp108, with 4/10 per-repeat wins.
- By held-out dominant class (r8d - RF dSSE):
  - Losses: P-additive +563 (Ridge alone +5095), intumescent +218, metal_hydroxide +128, silicon +107.
  - Gains: expandable_graphite -814, none -105.
- Excluding the P-additive class, r8d C is 0.308 vs 0.261 for RF.
- So yes: on a genuinely new FR class, the blend inherits Ridge's linear extrapolation of the cls_/el_ loadings. That is a real generalisation weakness, and P-additive is the largest FR class (71 rows / 23 papers).
- ET alone has the best C (0.308). Diagnostic RF+ET has C 0.300 with 10/10 wins.
- This should be **a caveat, not a block**:
  - The pre-declared rule names only B (mean, ≥ 8/10, ci_lo). Adding a C gate now would change the win rule after seeing the result, which is banned.
  - B (new papers) is the declared primary protocol.
- r8d must not be described as better for unseen FR chemistries. When citing r8d, report C 0.195 next to exp108's 0.237.

## Selection among 4 candidates (Q6)
- The four candidates are highly correlated (r8c and r8d share RF+Ridge), so the max-of-4 inflation is small.
- Test: paired paper bootstrap (1000 resamples) of dB for all four candidates, studentised max-T.
  - q95 = 1.98.
  - The simultaneous one-sided 95% lower bound for r8d's dB is **+0.006**. The unadjusted bound is +0.014.
- The gain survives selection, but its size is uncertain: +0.047 is the point estimate, and it could plausibly be anywhere from about +0.01 to +0.09.
- For a selection-adjusted headline, treat r8d as "about 0.41-0.42, with the improvement over exp108 small but positive". The repeat-0 CI [0.312, 0.552] stays the official CI.

## Caveats to carry
1. r8d becomes the record per the rule: B 0.425 [0.312, 0.552], A 0.736. Always quote C 0.195 (exp108 0.237) with it. It is an RF+ET+Ridge ensemble, not an RF.
2. The gain is paper-concentrated: the top 5 papers give 76%, and without M079 the result is 7/10, +0.034. The selection-adjusted lower bound on dB is only +0.006.
3. The C regression is caused by Ridge extrapolating onto held-out FR classes (mainly P-additive). Next round, pre-declare RF+ET (diagnostic B 0.397, C 0.300) and/or range-clipped Ridge rather than tuning anything post hoc.
4. Fix the truncated `model` field ("mean(RF") via the tracker script.
