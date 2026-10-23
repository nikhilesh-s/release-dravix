# Critic: pHRR round 1 (exp055–059), selected candidate c1 (exp043 minus MCC extras, RF, log target)

**Verdict: SOUND WITH CAVEATS.** The computation is honest and reproducible. The selection rule was applied as declared, and c1's
repeat-0 CI lower bound clearing 0.30 holds up under every perturbation I tried. But c1's *gain over exp043* is not statistically
established, and the CI only clears on the material-grouped protocol (not on family/category-grouped). See caveats.
Reviewer: Opus 5.5, fresh context. Scratch code: `$SCRATCHPAD/critic/{critic.py,merge.py}`. I recomputed everything independently by calling eval_protocol; it matches the log to 4 dp.

## 1. Leakage, grouping, target
- No feature uses the target. MCC, props and log_kpc are per-material aggregates of independent tests. c1 has no imputer, and RF handles NaN natively.
  The c5 imputers and one-hot encoder are fit on training-fold materials only (`impute_fit_predict`, `tr_mat = dfa.iloc[tr]`). OK.
- Target is `replicate_mean_target()`, the same function and data files (all unmodified since 2026-09-29 16:16) as exp043. It gives 191 rows / 65 materials. OK.
- **Grouping defect (probable duplicate):** `PlasticC` (no material.json, labelled "uncategorized") and `HIPS` are almost certainly the same plastic. In the raw cone files,
  HIPS is labelled "Plastic B" and PlasticC "Plastic C". Both were cone-tested on 220218 at 2.91 vs 2.86 mm. Their MCC values nearly match: peak 447 vs 441 W/g, char 4.47 vs 4.25 %,
  THR 39.1 vs 38.8, T_peak 441.3 vs 441.1. Their pHRR nearly matches too: 1010/1441/1738 vs 1032/1387/1746. They can land on opposite sides of a fold.
  Regrouping them as one material (64 groups) changes the numbers as follows. c1 goes from 0.664 to **0.635**. Its rep-0 CI is [0.498, 0.839] and its pooled CI [0.414, 0.761].
  exp043 goes from 0.567 to 0.546, with rep-0 CI [0.322, 0.830] (its ci_lo *rises*, because the splits change). c2: 0.611 [0.516, 0.838]. c4: 0.599 [0.458, 0.839].
  So the duplicate inflates means by about 0.03–0.06 but does not flip c1's CI clear. Other near neighbours (OSB/MDF, PMMA/Black_PMMA) are distinct products.

## 2. Incumbent reproduction and pairing
- exp043 reproduced exactly: 0.5672, sd 0.0617, CI [0.283, 0.836], GSS 0.3750, category-grouped 0.3151. All match the leaderboard.
- Every candidate uses the same `df` row order and groups. GroupKFold(shuffle, random_state=r) splits depend only on the groups, so the folds are identical.
  The paired counts are valid. c5 pairs on the 191 rows using the incumbent's own folds (asserted).
- Process: the config was committed in a518568 at 21:48:26. The script's mtime is 21:50:25 and the results' 21:51:26, so both come after the config.
  The script is untracked, so git cannot show which code actually ran. The current file does reproduce the log.
  The prose timestamps ("declared 22:20 PDT", blocker "22:35") disagree with the git and file clocks. This is bookkeeping only.

## 3. Forking paths (main caveat)
- History: c1's feature set first came from a post-hoc drop-one run (exp017: 0.640). It was then scored as exp038 (0.589), where it failed
  the rule that the paired-gain CI must exclude 0 ([-0.055, +0.287]). The data it was discovered on (175 rows / 60 materials) shares about 58–60 of today's 65 materials,
  so this is a third look at mostly the same data, not fresh evidence.
- **Paired gain vs exp043 (delta_ci, material bootstrap, repeat-averaged):** c1 +0.097, **CI [-0.057, +0.294]**. c2 +0.094, [-0.024, +0.243].
  c4 +0.094, [-0.027, +0.250]. Under exp038's stricter rule, **none would be preferred over exp043**. With the merged HIPS group: c1 [-0.065, +0.295].
- Selection among the top three is noise. c1 vs c2 is +0.003, CI [-0.078, +0.088]. c1 vs c4 is +0.003, [-0.079, +0.089]. The rule picked c1 by 0.003.
- Mitigation: c2 and c4 use the full incumbent feature set with no feature selection, and they also clear. Rep-0 ci_lo is 0.440 / 0.407 and pooled 0.373 / 0.368.
  So "some pre-declared model clears 0.30" does not rest on c1's post-hoc history alone. HGB is not history-free either: exp023 already showed ci_lo 0.307 on older rows.
- How much to discount: treat c1's 0.463 as a *selected* estimate. The defensible statement is that several pre-declared models on these rows have ci_lo between 0.41 and 0.46.
  Do not claim that c1 is better than exp043.

## 4. Stability
- c1 repeated GKF per repeat: 0.703, 0.620, 0.690, 0.690, 0.648, 0.604, 0.700, 0.684, 0.718, 0.584 (sd 0.044). exp043's ranged 0.452–0.660.
- c1 GSS x10: mean 0.442, sd 0.371, range [-0.383, 0.786]. Two of 10 splits are negative (seeds 5 and 9). On GSS, paired wins vs exp043 are only **7/10** (c2 and c4 6/10).
- **Category-grouped GKF** (family held out): c1 0.406, rep-0 material-bootstrap CI **[-0.063, 0.532]**. c2 0.392 [-0.129, 0.547]; c4 0.396 [-0.057, 0.550]; exp043 0.315 [-0.384, 0.499].
  **No candidate clears 0.30 when whole families are held out.**
- Leave-one-material-out on c1's pooled OOF R² (diagnostic only): range 0.617–0.713. Top-down: PP -0.047, LDPE -0.035; House_Wrap +0.045, PETG +0.049.
  The absolute level does not depend on any single material.
- The *gain* over exp043 is concentrated in a few materials. Of the squared-error advantage, House_Wrap carries 63%, PETG 41% and HDPE 26% (other materials offset).
  For House_Wrap (true 49/59 kW/m²), exp043 predicts about 1900–2200 and c1 about 1270–1460. Remove the top 1 and c1 is still +0.036 (8/10). Remove the top 3 (chosen post hoc) and c1 is **-0.040 (1/10)**.

## 5. CI method consistency
- Identical to exp043: `bootstrap_ci(y, oof[0], material, n_boot=2000, seed=0)` in both `score()` and `T.evaluate()`. No method change.
- Sensitivity of c1's rep-0 CI lower bound:
  - bootstrap seeds 0–4: 0.463, 0.474, 0.484, 0.469, 0.461
  - pooled over the 10 repeats: [0.425, 0.806]
  - ci_lo computed on each repeat's OOF: 0.463, 0.396, 0.466, 0.477, 0.342, **0.277**, 0.506, 0.401, 0.466, **0.261**, so 8/10 clear 0.30 and repeat 0 sits mid-pack
  - exp043's per-repeat ci_lo, for comparison: -0.223 to 0.406, only 1/10 above 0.30
- The rep-0 basis is noisy at the incumbent level: regrouping a single pair moved exp043's ci_lo from 0.283 to 0.322. For c1 the margin (0.46 vs 0.30) is large relative to that noise.

## Caveats a human must see before choosing wording
1. The clear is on **material-grouped** GKF only. Category-grouped CIs straddle 0 for every model, including c1's [-0.06, 0.53]. "Harder protocol" is true only relative to a random split.
2. c1's improvement over exp043 is **not significant** (gain CI [-0.057, +0.294]). It would fail the exp038-era rule. c1, c2 and c4 are statistically indistinguishable.
   If the post-hoc history matters, c2 (ci_lo 0.440, pooled 0.373) clears without feature selection.
3. **PlasticC ≈ HIPS** is a probable duplicate material split across groups. Fixing it lowers c1 to 0.635 [0.498, 0.839]. It should be fixed (by grouping, not by exclusion) before any number of record is cited.
4. The gain comes mostly from House_Wrap, PETG and HDPE. In 2 of 10 repeats c1's own ci_lo falls below 0.30.
5. Population: these are FSRI products, not Pomázi's epoxy composites. Comparability and the "SOTA" label are human decisions (H5 wording only). This critic makes no comparability judgement.
