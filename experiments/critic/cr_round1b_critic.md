# Critic: CR round 1b, c2 (exp069, HGB + 5 category flags) vs re-baseline exp071

**Verdict: SOUND WITH CAVEATS.** c2 can be the CR number of record. Two conditions: cite it with PlasticC merged into HIPS,
and describe the gain as a model swap that rests mostly on one material, not as an "inorganic proxy" effect.
The 0.82 bar is not in question. Scratch code: `/private/tmp/.../scratchpad/critic_cr/` (common.py, run_*.py).

## Reproduction and process
- Reproduced exactly: exp071 0.4825, c2 0.5502, 10/10 wins, per-repeat deltas +0.004 to +0.117. c1 0.5302.
- Pre-declaration holds. `cr_round1.json` was committed 231d5a1 (Oct 1 22:08:57). Round 1 ran at 22:10 and was BLOCKED at
  reproduction, before any candidate ran. `MINERAL_CATS` already sat in round1.py (mtime 22:10:13), so the list was fixed
  before any candidate result. `stale_rebaseline_round1b.json` was committed b5e0c00 at 08:42:16; the run wrote results at 08:43:39 and 08:44:06.
  The selection rule (top GKF mean, at least 8/10 wins, ci_lo at or above 0.234) was applied mechanically and passes.
- Banned moves were respected. The target is cone residue. STA N2 residue enters only as features (sta_res_*). The target is not STA, so MCC char yield is an allowed feature.
  No material was excluded.

## (1) Leakage
- The flags come from FSRI `material_props.categories`, which is product metadata, not anything derived from the target. They are deterministic per row.
  Each fold clones the model; no preprocessing is fitted outside a fold. STA and MCC values are per-material means of separate
  tests and carry no cone information. No leakage found.
- The flags are a poor "inorganic" proxy: `insulation` also covers EPS, PIR and PE foam, and `interior finish` covers foam pads.
- **Grouping defect: PlasticC is the same material as HIPS** (the same issue the pHRR critic found, and the pHRR record already merged them). Here: MCC char 4.25 vs 4.47,
  STA 4.19/4.76/5.97 vs 3.92/6.74/5.05, cone residue 3.96 vs 4.20. With the two merged into one group:
  base 0.4831 [0.132, 0.683], **c2 0.5491 [0.342, 0.693], 9/10 wins**. The rule still passes, with a wider margin on ci_lo.
  MDF and Particleboard share density 583 and a 3 K STA mean of 23.54, but their raw STA runs are separate (2021 vs 2022), so this is a coincidence, not a duplicate.

## (2) Paired gain CI (material bootstrap of the repeat-averaged gain, delta_ci method)
- c2 minus base = +0.068, **95% CI [-0.041, +0.249]** at seed 0. Seeds 0-4 give lower bounds of -0.038 to -0.043, so **the interval includes 0**.
  With PlasticC merged: [-0.046, +0.250]. The 10/10 paired wins come from repeats of the same 62 materials, so they do not
  establish that the gain holds across materials. The win rule does not require this test; it is reported as a caveat.

## (3) CI robustness (c2 vs base)
- Repeat-0 CI over bootstrap seeds 0-4: c2 ci_lo 0.313 to 0.330; base 0.216 to 0.240.
- ci_lo for each of the 10 repeats: c2 min 0.237 / median 0.333; base min 0.091 / median 0.158. **exp071's recorded ci_lo of 0.234 is its
  luckiest repeat**, so the rule's ci_lo threshold was easy to clear.
- Pooled over repeats (mean R2 for each resample): c2 [0.324, 0.698] vs base [0.174, 0.665]. With repeat-averaged predictions: c2 [0.396, 0.750] vs base [0.260, 0.718].
  c2 is ahead under every CI variant.

## (4) Category-grouped GKF (primary family held out, 11 families, 5x10)
- base 0.069, **c2 0.378**, HGB without flags 0.281, 9/10 wins; paired-gain CI [+0.066, +0.712]. LOFO: base 0.127 vs c2 0.389.
  Under this harder split, c2's advantage is large and its CI excludes 0. This is the strongest evidence for c2.

## (5) Concentration of the gain
- **EPS_Foam_Board alone accounts for about 84% of the gain** (0.057 of 0.068 in R2 units of SSE reduction). Mean OOF prediction for EPS
  (true residue -1.8%): base 44.4, c2 7.4. The probable cause is the baseline RF learning that STA rows with only the 10 K rate and NaN elsewhere mean high
  residue: mineral wool, FRP, vinyl tile and shingle share that NaN pattern. HGB routes NaN differently.
  Nylon adds a further 0.024 (base 25.1 vs c2 3.3, true 1.9).
- Diagnostic only, without refitting (exclusion is banned): drop EPS → gain +0.011 with 7/10 wins; drop EPS and Nylon → gain -0.014 with 2/10.
  c2 loses on mineral wool (73.4 to 61.6, true 96.9), FRP and rubber foam, and its exterior-finish error share rises from 0.145 to 0.186.
- By family: general polymer contributes +0.046 and insulation +0.036; exterior finish is -0.010 and sleeping product -0.004.

## Attribution
- HGB without flags scores 0.5401 (10/10 vs base). The flags add only **+0.010** on top of that (7/10; paired CI [-0.010, +0.034]).
  Nearly all of the gain comes from swapping RF for HGB and its NaN handling, not from the inorganic proxy. Under category-GKF the flags help
  more (0.281 to 0.378), but there they act partly as family identifiers.

## Required wording for the record
- "CR: HGB + category flags, rGKF 5x10 0.549 [0.342, 0.693] (PlasticC grouped with HIPS), 9/10 paired wins vs the exp033
  config re-baseline; paired gain +0.07, CI includes 0; the gain is mostly one material (EPS)." The 0.82 bar is not cleared.
- Do not describe the gain as coming from the inorganic flags.
