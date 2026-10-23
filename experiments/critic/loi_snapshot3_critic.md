# Critic: LOI snapshot 3 (snapshot 2 + wave 5; exp113 / exp114 configs on 606 rows / 107 papers)
Verdict: **SOUND WITH CAVEATS**. The snapshot rule, the wave-5 data and the scoring script were all committed before the run. Both snapshot-2 and snapshot-3 numbers reproduce exactly. All 35 wave-5 rows match their sources and the extraction rules, and no wave-5 row needs a data fix.

The drop (0.425 -> 0.386) is not a data error. It has three parts:
- about half is GroupKFold reshuffle luck on the old rows;
- about half comes from a new batch with low LOI variance, predicted about as well as the old rows in absolute terms;
- the remaining ~0.004 is the shifted F7 priors and the new training rows.

**Recommendation: snapshot 3, unchanged, becomes the LOI benchmark of record**:
- exp113 config: **0.3857 [0.3086, 0.5306]**, pooled CI [0.237, 0.487], C 0.204.
- proxy-free (exp114 config): 0.3834 [0.3030, 0.5469], C 0.129.

The record should describe this as a different population, not a regression. Use the caveats below as written. Do not drop or "fix" any row because of its effect on the score.

## Pre-declaration and order (git times)
- Discovery rules were pre-declared in ffaa023 at 21:40:40 (data_pass_oct4 `loi_wave5_oa`). The discovery manifest was committed in 37a6384 at 23:00:42.
- rows_w13, log_w13, QA5, the element/structure additions, the snapshot-3 amendment and train_loi_rescore_r8.py were all committed in f17818f at 23:09:23. File mtimes run from 23:05:57 to 23:09:10.
- The snap606 log and results were written at 23:22:01. With a 385 s runtime the run started at about 23:15:36, after f17818f. OK.
- 775a29c at 23:22:35 commits the log plus a one-line change to rescore_loi_snapshot.py: the log tag `_snap2` becomes `_snap{n_rows}`. That change was in the working tree during the run. It only renames the log; it is not a scoring change.
- The rows were extracted before the amendment, by the same agent, so the person who wrote the rule knew the LOI values. This is the same situation as snapshot 2. Both wave-5 rulings in the amendment are about population, not scores, and both hold up against the source:
  - V006 not_rpuf: the PDF shows TDI 48 php, a flexible slabstock-type recipe.
  - V005's collective "25" kept as printed.
- **Authorisation is weaker than for snapshot 2.** Snapshot 2 needed Arnav's explicit "re-score LOI now" and has it quoted. Snapshot 3 cites only the generic loop instruction ("keep working on making dravix better").
  - The snapshot rule ("regardless of direction") does cover later waves.
  - But the "when" condition was written for the library-PDF wave. Get Arnav's explicit go before numbers_of_record changes.
- The snapshot-2 self-check log was overwritten by the tag change. I reproduced it independently from the committed script's prefix in a scratch tree:
  - exp113: 0.4249 [0.3122, 0.5519]
  - exp114: 0.4146
  - Both are exact. Commit a self-check log next time.

## Reproduction (scratch trees only; nothing written to the repo)
- Built with `rescore_loi_snapshot.py --stage wave4 [--stage wave5] --build-only --keep` (TMPDIR = scratchpad). The trees have 571 / 102 and 606 / 107 rows / papers.
- I ran the committed train_loi_rescore_r8.py prefix (patched only by the tool's own patch_round6) and dumped OOF predictions for RF, ET and Ridge separately. The blend is their mean.

| Config | Snapshot 2 | Snapshot 3 |
|---|---|---|
| exp113 config (blend) | 0.4249 [0.3122, 0.5519] | 0.3857 [0.3086, 0.5306] |
| proxy-free blend | 0.4146 | 0.3834 [0.3030, 0.5469] |
| **RF-only exp092 config** | 0.3775 | **0.3362** [0.2535, 0.4857] (-0.041) |
| ET (exp092 features) | 0.366 | 0.341 |
| Ridge (exp092 features) | 0.235 | 0.186 |
| C, blend | 0.195 | 0.204 |
| C, RF | 0.237 | 0.231 |

- Every model drops by about 0.04. The drop does not come from the blend.
- On snapshot 3 the blend still beats the RF config in 9/10 repeats (mean +0.050), so exp113 stays the right record config.

## Why R2 drops 0.039 (exp113 config)
- SST rises only 6707 -> 6922, because the 35 new rows have LOI variance 4.1 vs 11.8 for the old rows, and their mean (21.5) is close to the overall mean (23.0).
- SSE rises 3857 -> 4251. Of that, +243 is SSE on the new rows and +152 is extra SSE on the old rows.
- **New rows are not badly predicted in absolute terms.**
  - Their MSE is 6.9 vs 7.0 on the old rows.
  - Replacing their SSE with old-row MSE gives 0.3854, the same R2. The new-rows-only R2 is -0.68 because their variance is small.
  - Holding old-row SSE at its snapshot-2 level gives 0.4077. So the low-variance batch costs about 0.017.
- Per new paper (SSE / SST3):

| Paper | Cost in R2 | Notes |
|---|---|---|
| V004 | 0.024 | 69% of new SSE. 14 sunflower-cake foams (+/- liquid glass, +/- TCPP) are over-predicted by about 3.1 LOI points. Actual LOI 18.9-21.7, predicted 20-26. |
| V001 | 0.006 | |
| V003 | 0.002 | |
| V005 | 0.002 | The collective-25 rows are predicted at about 23.7, so they are not what drives the drop. |
| V002 | 0.000 | |

- **Old rows get worse: R2 0.4249 -> 0.4023 (-0.023).** Decomposition:
  1. F7 class-prior shift: 11 old rows in 4 papers change el_P / el_metal / syn_P*N. Snapshot-2 rows with snapshot-3 features on snapshot-2 folds give 0.4241, so the effect is **-0.001**.
  2. Wave-5 rows as extra training data: snapshot-3 folds with V rows removed from training give old-row R2 0.4055 vs 0.4023 with them, so **-0.003**.
  3. The rest, **about -0.019, is fold reassignment.** Adding 5 groups reshuffles GroupKFold for every repeat. That is about 1.6 SE given a per-repeat std of 0.038.
- 63 old papers get worse and 39 get better. The largest losses are N022 +48, W006 +27, M083 +25, N017 +23 and M087 +21. The largest gains are M032 -33 and M001 -21. This pattern is consistent with reshuffle noise.
- Per-repeat blend values:
  - snapshot 2: 0.455 / 0.394 / 0.418 / 0.433 / 0.475 / 0.500 / 0.414 / 0.393 / 0.345 / 0.422
  - snapshot 3: 0.441 / 0.338 / 0.409 / 0.386 / 0.425 / 0.374 / 0.400 / 0.407 / 0.311 / 0.368
- The rep-0 ci_lo barely moves (0.312 -> 0.309). The pooled CI moves [0.280, 0.530] -> [0.237, 0.487]. Quote the pooled CI.

## Wave-5 rows vs sources (35/35 checked)
Sources:
- V002 PMC7659977 and V004 PMC9657363 from Europe PMC fullTextXML.
- V001 thesis PDF, V003 InTech PDF and V005 Polimery PDF via oa_url and curl. The V005 host returned 403 to bare curl; the second GET with a browser user-agent worked. No paywall.

Paper-level results:
- V001 (8 rows): Table 9 and Table 13 LOIs 17.0 / 22.0 / 22.2 / 21.0 / 22.0 / 22.0 / 21.4 / 21.0 are correct. The B3 and B2 controls are merged under R9.
  - L-HNT densities 75 and 72 come from Table 5, which is the L-HNT series of the same formulations. This is legitimate, the same as cross-table density reads in earlier waves.
  - Blank densities for the references and HNT-PPO rows are correct ("never guess": Table 3 ref 90 vs Table 5 ref 74; HNT-PPO is figure-only). The asymmetry (B3 ref blank while the L-HNT rows use Table 5) is conservative, not a violation.
  - The thesis keywords say RPUF.
- V002 (4 rows): LOI 20.2 / 19.8 / 20.8 / 21.6 (text) and Table 4 densities are correct. HS is 2 pbw (Table 2).
- V003 (3 rows): Table 7 OI 19.6 / 23.2 / 26.4 and Table 4 densities 33.4 / 40.3 / 41.5 are correct, ASTM D 2863-1970. **TMCP 46.9 g is in every foam (Table 3)**, so K0 is correctly not neat.
- V004 (14 rows): all Table 5 LOIs, Table 2 densities, TCPP 25 pbw and SFP wt% are correct.
- V005 (6 rows): **TMCP 54.0 g is in every foam (Table 2)**, so the extraction is correct.
  - M0 OI 21.4 and densities M0 36.3 / M5 39.5 are correct from the text. M1-M4 densities are only in the Fig. 2 bars and are correctly left blank.
  - The cake loading basis checks out: 15.9 g / (66.8 + 250.6) = 5.0%.

Questionable rows and rulings:
- **V005 "25": legitimate, keep.**
  - The text (p. 591) says the oxygen index of all foams modified with rapeseed cake was 25% (my paraphrase of the Polish).
  - It is a stated measured value for each foam. It is not censored (R7), not figure-read (R6) and not a guess.
  - It is integer precision and probably a summary statement, but under the rules that is a precision caveat, not a violation.
  - It was declared before scoring and costs only 0.002.
- **V001 TCPP basis: legitimate as copied.** The thesis says "15% FR" and "15 wt% liquid FR B2 contains" in a polyol-component context. The basis is ambiguous (it may be php). The rule says copy as printed, which gives "wt%". Note it as an ambiguity.
- **V003 / V005 unit strings: not a wave-5 error, but a pre-existing normaliser bug.**
  - `build_loi_rpuf.basis()` sends any unit containing "of polyol" to the php basis. That catches "R (chemical equivalent of polyol)" (V003 boron polyol: 0.025 / 0.12 wt% instead of about 1.7 / 8.5 wt%). It also catches "wt% of polyol+isocyanate" (V005 cake: cls_bio-based 1.2-5.8 instead of about 3.9-17).
  - The same strings already occur in old rows: N029 ×6, N049 ×6, W014 (R), W017 (chemical equivalent) ×4. The wave-5 extraction follows the existing convention.
  - Fix it for all rows as a separately declared data-prep fix with one re-score, like the kaolin blank fix. Do not patch only wave 5 now. The effect is unknown and should be small: features only, no labels.

## Caveats to carry
1. Snapshot 3 is a different population (+35 rows of a low-variance cluster), not a model regression. Always quote the pooled CI [0.237, 0.487] and C 0.204 (proxy-free C 0.129).
2. About half of the drop is fold-reshuffle luck; snapshot 2's 0.425 was partly luck too. Expect about ±0.02 swings at every future wave.
3. The normaliser's "of polyol" basis bug affects V003-2/3, V005-2..6 and about 17 old rows. Declare and apply the fix across all rows, then re-score once.
4. Record Arnav's explicit go for snapshot 3 in the config before numbers_of_record changes. Commit a snapshot-2 self-check log.
5. Disclose that the F7 prior shift changes 11 old rows in 4 papers (effect -0.001).
