# Critic: C1 FSRI composite snapshot (fsri_composite_snapshot.json, D5; pHRR and Char Residue)

Fresh-context critic. I did not run this experiment. Sources: the committed config, `data/v2/rescore_fsri_composite_snapshot.py`, the two logs, result JSONs, diag JSONs and OOF npz files (all in 954ad20), the incumbent modules, and my own scratch re-runs in `.venv312` (OMP 1, LOKY 3). I changed no file except this one.

| Target | Verdict |
|---|---|
| pHRR: exp074 config on C1, **0.6328 [0.4902, 0.8674]**, 205 rows / 70 materials / 66 groups | **SOUND WITH CAVEATS** |
| Char Residue: exp117 config on C1, **0.7091 [0.5499, 0.8294]**, 603 rows / 66 materials / 62 groups | **SOUND WITH CAVEATS** |

Bottom line:
- The implementation follows the config literally. Every new row's features and labels match my own recomputation from the raw CSVs and material.json. No old row's X, y, material or flux changed.
- Both incumbents reproduce exactly in the same run. I reproduced every reported number from the npz and re-ran S0/S2 to the same OOF (pHRR bit-identical; CR within 3e-14).
- I found no leakage and no banned move.
- Under the snapshot rule, both S2 numbers become the numbers of record. The caveats are about process order and record wording (sections 1 and 8), not about any number.

## 1. Pre-declaration and order (git and file times, PDT)
- The config was drafted in dd9c9c6 (Oct 6 14:52). The D5 approval amendment came in 69ffda8 (Oct 7 22:16:27). The results came in 954ad20 (Oct 8 01:52:54). **The config was committed before any score.**
- The script was committed together with the results, so I cannot check from git that it was written before the run.
  - Its last mtime is 01:50:45.
  - The pHRR run started about 01:51:10 and took 13 s. The CR run started 01:51:12 (the layers-CSV write) and took 64 s. The two runs were in parallel, with LOKY 1 + 2, so within the 3-core limit.
  - I found no trace of an earlier scoring run. No `--selftest-only` log was committed.
  - None of this affects the numbers, because the builders are deterministic and assert the label-free values the config fixed.
- **Order deviation (immaterial).** `precondition_order` says C1 runs "after that target's round-9 adoption decision and critic".
  - The round-9/10/CR-round-2 critic (`overnight_2026-10-08_critic.md`) was last modified at 01:54:15 and committed in db05669 at 01:55:22. That is about 3-4 minutes after the C1 runs.
  - The gates (all `adopt_if_critic_passes: false`) existed beforehand. That critic then confirmed no adoption, so exp074 and exp117 were the right incumbents.
  - The order should still be recorded as a deviation.
- **Environment.** The config pins Python 3.12.4 with "stop BLOCKED" if anything differs. The run used `.venv312`: Python 3.12.12 with the identical sklearn/numpy/pandas/scipy pins.
  - The worker flagged this before scoring (`fsri_composite_flags.md`).
  - The erratum was committed only with the results. db05669's timestamp note now says it was written after the run.
  - It does not matter numerically: S0 reproduces exp074 and exp117 exactly.
- Amendment clock times disagree with git. "22:45" appears in a commit made at 22:16. "~02:00" appears in a commit made at 01:52. db05669's note says git times govern. Nothing further is needed.

## 2. Implementation vs the config (item 1)
- **Layer map.** I recomputed it from every `material.json` with my own code. It equals `composites.layers` exactly. The only other `component material` lists are Bamboo_Rug, Overstuffed_Chair and Overstuffed_Sofa (`not_mapped`). The side CSV has the 12 layer rows, 1-based, in material.json order.
- **Aggregation.** `map_layers` is `agg.reindex(layers).mean(axis=0, skipna=True)`, as declared.
  - It runs after the stage-1 per-material groupby means: per heating rate for STA via `pivot_table` mean, and per material for MCC.
  - It appends one row per product only if some layer has a value, and it asserts that no product is already in the table.
  - My own two-stage computation (per-layer replicate mean, then unweighted mean of the non-NaN layers) matches the config's `mapped_values` to within 4.4e-5 (4-dp rounding). It matches the builder's X on every new row to 1e-9.
  - No clipping. The gypsum core's -0.038 kJ/g enters as is, and Gypsum_Wallboard's onset is the paper layer's alone.
- **Mapped columns.**
  - pHRR maps only mcc_agg (peak, peak_norm, char yield) and x_agg (`total_hr_bc_kJg`, `t_peak_C`, `t_onset_bc_C`).
  - CR maps only sta_wide (3/10/30 K) and mcc_agg (char yield, peak).
  - Density, thickness, mass per area, categories and log_kpc come from the product's own rows, as `not_mapped` requires. log_kpc is 10.4447 for Gypsum_Wallboard and 10.0409 for Lightweight_Gypsum_Wallboard (their own Dry HFM), and NaN for the other three.
  - STA is N2 only, and every layer's MCC is at 0.5 C/s.
- **Builders vs the record loaders.**
  - `build_phrr_c1` is `T.load(replicate_target=True)` with the `--bc` extras, line for line, plus the two `map_layers` calls.
  - `build_cr_c1` is `data_cr()` line for line, plus the two `map_layers` calls before the outer merge.
  - Left merges keep the cone-table order, so the new rows land at the declared positions: pHRR 56-60, 81-83, 109-111, 150-152; CR 180-188, 255-263, 353-361, 468-476.
- **Old rows unchanged.** With the mapping on, the 191 / 567 non-product rows are bit-identical to `data_phrr()` / `data_cr()` in X (equal_nan) and y, in order. I re-checked this outside the script's self-test.
- **Rows added are exactly those named.**
  - pHRR, 14 rows: R30_PF 25/50/75; Gypsum_Wallboard 50/75 (its HF25 replicates did not ignite); 3 each for the other three products. This equals all 14 ignited, non-early-stop product x flux cells in `cone_replicates_fsri.csv`. Each label equals my own replicate mean.
  - CR, 36 rows: 9 each for Gypsum_Wallboard (including the 3 non-ignited HF25 rows, as declared), Lightweight_Gypsum_Wallboard, the countertop and the chair assembly. The labels equal the raw spark, non-null `cone_residue_pct` rows in CSV order.
  - R30_PF's 9 raw CR rows are absent, because neither of its layers has STA (pre-declared entry rule).
  - The complete-layer rule changes no row set (asserted).
- **Counts and NaN.** The NaN counts match the config. The category counts are 11 / 11, and all four chair-group materials take 'upholstered furniture'.

## 3. Incumbents unchanged and reproduced (item 2)
- `train_phrr_v3_robust.py`, `train_blend_round1.py` and `eval_protocol.py` were last changed on Oct 2 and Oct 6, and none has working-tree edits. The only modified tracked files today are LOI scripts, which C1 does not import.
- pHRR uses `T.hgb()` with log1p via `eval_protocol`. CR uses `Avg([data_cr()['inc'] HGB(300, 0.05, 15, 10), et(), ridge()])` on the raw % target, unclipped. Both match the records.
- S0 in the same run: exp074 **0.6113 [0.5155, 0.8376]** and exp117 **0.6504 [0.4389, 0.7893]**, both exact.
- My re-run of S0 and S2 gives the npz OOF bit-for-bit for pHRR, and to max |d| 2.8e-14 for CR (ET/Ridge float ordering).

## 4. Leakage (item 3)
- The mapped features come only from bench-scale MCC/STA tables of the layers. No cone label of any product or layer enters any feature.
- Groups follow material, then PlasticC -> HIPS, then the four chair materials -> `Overstuffed_Chair_Assembly`. I checked this against the npz in both targets: 66 and 62 groups.
- `repeated_group_kfold` asserts disjoint groups in every fold, so the assembly and its layers are never split across folds. The category-GKF never splits them either, because all four share one category.
- The only layers with cone rows are the three chair layers, so no other layer is ever tested.
- Residual, declared and not a bug:
  - Near-twin products stay in separate groups, as in every prior round: Gypsum_Wallboard vs Lightweight_Gypsum_Wallboard (both new), and Particleboard vs the countertop's particle-board layer.
  - This pair is not named in `groups.not_linked`, but it is covered by its general rule.
  - It does not visibly inflate the score: both gypsum boards are under-predicted on CR by 13 and 28 points.

## 5. Banned moves (item 4)
- **No thermal-property join.** log_kpc is exp074's existing feature, product-own and not layer-mapped.
- **The pHRR pool grows**, from 65 to 70 materials.
- **No exclusion after seeing influence.** R30_PF's absence from CR follows the pre-declared entry rule.
- **CR target.** y equals `cone_residue_pct`, and STA residue is a feature only.
- **No variants or other numbers computed.** No other layer rule, no unmerged chair and no leave-out scores appear in any output. I computed none either: my decomposition below uses only the pre-declared old/new subsets.
- **No record files touched.** numbers_of_record, the leaderboard and the xlsx files are untouched by 954ad20.

## 6. Re-computation from the npz (item 5; `eval_protocol.bootstrap_ci`, n_boot 2000, seed 0)
| | Reported | Recomputed |
|---|---|---|
| pHRR S2 repeat-mean R2 / repeat-0 CI | 0.6327724266 [0.4901560114, 0.8674152205] | identical |
| pHRR pooled CI / cat-GKF / GSS (re-run) | [0.5003, 0.8460] / 0.3414 [-0.3074, 0.4925] / 0.3104 | identical |
| pHRR S0 / S1 | 0.6113 [0.5155, 0.8376] / 0.6296 [0.1244, 0.8230] | identical |
| CR S2 repeat-mean R2 / repeat-0 CI | 0.7091150773 [0.5498769391, 0.8293619354] | identical |
| CR pooled CI / cat-GKF | [0.5517, 0.8289] / 0.5704 [0.4613, 0.7687] | identical |
| CR S0 / S1 | 0.6504 [0.4389, 0.7893] / 0.6489 [0.4466, 0.8009] | identical |

The per-repeat R2, the old/new subset R2, the SSE shares and the per-composite table all match too.

## 7. What moved the numbers (item 6; not paired, folds differ)
**Char Residue: +0.059 over S1 is a population change, not a model gain.**
- The 36 new rows have mean residue 56.4% (old rows 24.6%) and variance 1474 (old rows 575). The two gypsum boards at about 92-95% raise the total SST by 27% (326,088 -> 413,389).
- **The new rows are predicted worse per row than the old ones.** New-row MSE is 303 vs 193. The gypsum boards are 3% of rows but carry 8.1% of the SSE.
- R2 rises anyway because the denominator grows. If the old rows' SSE is held at its S1 level, the snapshot R2 is 0.697. So about +0.048 is variance enlargement. The remaining ~+0.012 is lower old-row SSE (more training rows plus fold reshuffle, unpaired).
- This is why the pre-registered expectation ("expect it to hurt") was wrong in sign: R2 is variance-relative. The gypsum boards are predicted badly, as the config foresaw.

**pHRR: +0.022 is almost all fold reshuffle.**
- S1 (old rows, chair merge only) already gives 0.6296 (+0.018). Old-row SSE is slightly higher in S2 than in S1.
- The 14 new rows are low-pHRR (mean 292 vs 663 kW/m2) and low-error. They add only 3.9% to SST.
- Per-repeat S2 R2 ranges from 0.36 to 0.74 (std 0.113 vs 0.089 for S0), so the change is inside the noise.
- The secondary protocols moved down:
  - GroupShuffleSplit x10 went from 0.494 to 0.310, with 3 of 10 seeds negative.
  - Category-GKF went from 0.392 [-0.149, 0.529] to 0.341 [-0.307, 0.493].
- The repeat-0 CI lower bound is fold-sensitive. S1, with the same model and rows and only the grouping changed, has ci_lo 0.124. S2's 0.490 and the pooled CI [0.500, 0.846] both clear 0.30, so the exit trigger correctly did not fire. "CI clear" should still be read with that sensitivity in mind.

## 8. Record-note caveats (to be carried into numbers_of_record)
Both targets:
- Use the config's `record_text` template.
- Name the old record and population, and say "population change, different population; not comparable to <exp>".
- Never call the move a model improvement or a regression.
- `comparable` stays pending-human. Neither number is comparable to the Pomazi 2026 (BME) bars.

**pHRR.** Keep "AHEAD, CI CLEAR, DIFFERENT POPULATION" (ci_lo 0.490 > 0.30). Update the stale figures in the current note and add:
- Category-GKF is now 0.341 [-0.307, 0.493]: not ahead when families are held out.
- Pooled CI [0.500, 0.846]; GSS 0.310.
- +0.018 of the +0.022 comes from the chair-merge reshuffle (S1 0.6296).
- 25 of 42 composite replicates behind the new rows are truncated_test.
- `models/v2/phrr_hgb_v4.joblib` is still the 191-row fit. The config forbids a refit, so the record CV population no longer equals the artifact's training set.

**CR.**
- State that the rise is variance enlargement from the near-inert gypsum boards. The new rows have higher per-row error (MSE 303 vs 193; 0.697 with the old-row SSE held at S1).
- **Fix:** the current note's "Below the Pomazi bar 0.82 even at the CI upper bound (0.789)" becomes false. The new upper bound, 0.829, lies above 0.82. Status stays SHORT OF SOTA (ci_lo 0.550).
- The exp117-vs-exp081 facts (10/10, mineral wool 29%, Ridge out of range) describe the old 567-row population and should be labelled that way.
- R30_PF is absent (no layer STA).

## 9. Anything to fix
- **Blocking: none.**
- **Record notes:** apply section 8. The CR upper-bound sentence is the one factual error waiting to happen.
- **Process (write into the amendments or record notes):**
  - C1 ran before the round-9/10 critic was final (section 1).
  - Next time, commit the `--selftest-only` log.
