# Critic: LOI round 8 supplementary (s8 = r8d blend on proxy-free c3, snapshot 2)
Verdict: **SOUND WITH CAVEATS**. s8 may replace numbers_of_record LOI.proxy_free (exp109 RF c3 0.371 [0.277, 0.523]) **as 0.415 [0.327, 0.572]**, as the config declared. It is reporting only: no selection, no promotion. The caveat that must travel with it: **protocol C is 0.050**, against 0.226 for exp109 and 0.195 for exp113.

## Pre-declaration and order
- Config and script were committed together at 22:57:44 (626cdbc). The log and result mtimes are 22:59:48 and the runtime was 115 s, so the run started at about 22:57:53, after the commit.
- The working-tree script is identical to 626cdbc (empty `git diff`). The log was committed at 23:00:07 (ebd0ae6).
- The result JSON is untracked, which is expected because it is pending this critic.

## Implementation
- I diffed the script against train_loi_round8.py. The only differences are:
  - the docstring;
  - `MAKE = {"s8": Avg([rf(), et(), ridge()])}`, with rf/et/ridge byte-identical to r8d;
  - `X2 = DATA["c3"][0]`;
  - the output block.
- The round-6 prefix is verbatim. It still reproduces exp108 (B 0.3775, CI [0.2887, 0.5114], "Reproduction within 0.002: True"). The round-8 hard stop was removed, but that is acceptable because nothing is being compared or selected.
- No std_* leak:
  - `FEAT_C3 = FEAT_C1 + ["log_density"]`, built on `dC`. The std5 columns go only into d6/c5.
  - The result JSON lists 47 features, with no std_* among them.
- Rows match exp113: 571 rows / 102 papers, the same y and g (asserted), and the same seeds.

## What the low C means
- s8 loses 0.010 on B against r8d, but C falls from 0.195 to 0.050.
- With the lab proxy removed, Ridge's extrapolation onto held-out FR classes is no longer offset. Across unseen FR chemistries the model is close to predicting the mean.
- How to quote it:
  - Always quote "0.415 B / 0.050 C" together.
  - Never call it chemistry-transferable.
  - Note that the proxy-free RF (exp109, C 0.226) remains the better choice for out-of-class prediction.
  - Note that s8's B gain over exp109 (+0.044) is unpaired and was not tested.
- Minor: the pooled CI lower bound (0.240) is well below the repeat-0 CI lower bound (0.327). Quote both.
