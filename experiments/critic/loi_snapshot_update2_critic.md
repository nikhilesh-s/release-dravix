# Critic: LOI snapshot re-score (loi_snapshot_update2; exp102 / exp103)
Verdict: **SOUND WITH CAVEATS**. The re-score was declared and committed before it ran. The tool changes only what it says it changes, the self-test reproduces exp092 and exp093, and the 43 wave-4 rows I checked against the papers are correct. Two things must happen before numbers_of_record changes:
- Fix a data bug in the staged element-table additions. It sets every el_* value to NaN on 9 new rows, and the fix lowers both numbers by about 0.004.
- Stop reading the ci_lo jump (0.20 -> 0.30) as a gain. It is mostly luck in repeat 0.

There is also one undeclared side effect on old rows. It is harmless, but it should be disclosed.

## Pre-declaration and order (git times)
- Rules, data and tool:
  - f1d1710 at 18:10:18 committed the config (mtime 18:05:48, unchanged since), inclusion rules R1-R9, element/structure additions, the tool and the self-test log. The self-test log was written at 18:09:57, but it scores only the existing 460 rows.
  - 4f26ea8 at 18:17:53 added write_results() only.
  - The expanded log and exp102/103 were written at 18:22:08. At 250 s runtime the run started at about 18:17:58, after both commits. OK.
- Wave-4 rows (rows_w9-w12) were last modified on Oct 4 at 13:39-13:48 and committed in 89d739c at 13:48:58. The rules were locked a day later.
  - The rule writer knew the extracted LOI values (the same agent extracted them).
  - The rules are about population, not scores. Each cites a precedent, so I see no sign that any rule was tuned to the scores.
- **The "when" condition has no record in the repo.** The config, the README and STATUS (18:10) all say the re-score waits for the library-PDF wave or Arnav's "re-score LOI now".
  - No file records that Arnav said this. The 18:17 commit implies it did happen.
  - Quote Arnav's words in the config or STATUS before the record changes, as tti_round2.json does for its own authorisation.

## Tool and patches
- patch_round6 makes only two kinds of change:
  - the two `== 460 / == 83` asserts, which now take the snapshot's counts;
  - the two `sys.exit(2)` reproduction stops, which now print a message instead of stopping.
- Both are asserted to match exactly once or twice. Nothing else in train_loi_round6.py changes. patch_build only extends EXCLUDE_REFS with W032.
- The tool reproduces. My harness uses the tool's own copy_tree / apply_stage / patch_* functions and dumps features just before round 6's fits. It reproduces the expanded numbers exactly: exp092 config 0.3812 [0.296, 0.520], c3 0.3744 [0.2857, 0.5274].
- Cosmetic problems in the log. These are labels only; the counts are computed correctly on 571 rows (the polyol counts sum to 571).
  - After each "BLOCKED ... reported, not stopped" line, the log still prints "Reproduction within 0.002: True". This is false on the expanded run.
  - The headers "460" / "Summary (460 rows / 83 papers)" are hard-coded.
  - The round-6 "SELECTION ... exp092 stays the record" line does not apply to a snapshot re-score. Ignore it.
- exp102 records no pooled CI. exp103 does (0.225-0.486). Add exp092's pooled CI [0.238, 0.490] from the log.

## Inclusion rules vs what was included
- The counts are consistent:
  - 127 rows, minus 5 figure-read rows (W009), minus 2 censored "< 20" rows (W013) = 120.
  - Minus W032's 9 rows (R2) = 111 new rows from 19 papers, giving 571 / 102. W029 has 0 rows (NIPU, R3).
- W032 excluded is consistent with N019, which EXCLUDE_REFS already excludes as open-cell spray foam.
- W013 ("rigid/semi-rigid", not spray, not open-cell) is kept under R1. OK.
- **W010 kept (R4) conflicts with the N039 precedent, and R4 is not applied back to N039.**
  - The README's own open-questions list says "W010 single non-FR foam (N039 precedent says drop)".
  - N039 is the same Rzeszów group's metasilicic-acid rigid foam with no FR. It was logged no_loi_table / "not FR study". W010's paper cites it as ref [12].
  - R4 now keeps such papers for new waves only, so old and new papers follow different population rules.
  - The effect is 1 row. Either re-examine N039 under R4 or say why it stays out.
- R8 (W005 "GB 8624", W017 ASTM D1692) puts those rows in std_other_unknown, exactly as the round-5 regexes would. R9: W015's shared control appears once (confirmed by QA).

## Element-table and structure additions
- I checked all 14 formula entries by hand: H3PMo12O40 (P 1.70, Mo 63.08), ulexite (B 13.34, Na+Ca 15.56), Sb2O3, DAP, Al(OH)3, Al diethylphosphinate, the two heptanetriols, DBBD, Fyrol-6, the APP wording variant, TDEAMM/FTDEAMM/DFTDEAMM.
  - The formulas, molar masses and mass fractions are all correct to the printed precision.
  - The derived-from-name structures (heptanetriols, TDEAMM series) are chemically consistent.
- **Bug: the two class entries `^kaolin` and `^flax fib` have blank P..metal fields.** The committed class rows (perlite, lignocellulosics, ...) use 0.
  - build_loi_rpuf and fill_row both do `float(e[x])`, so a blank gives NaN, and every el_* value on the row becomes NaN.
  - That is the opposite of the stated intent ("0 with mineral flag").
  - Affected: 9 benchmark rows, W005-2, -4..-7 and W016-2, -3, -5, -6. W005-4..-7 is the APP + kaolin series with LOI 22.4-30.2, so the APP phosphorus signal is lost exactly where it matters.
  - el_imputed_share = 0 on these rows, which claims they are fully resolved.
  - Re-scored with 0s (my harness, same seeds):
    - exp092 config: 0.3812 [0.296, 0.520] -> **0.3775 [0.289, 0.511]**
    - c3: 0.3744 [0.286, 0.527] -> **0.3708 [0.277, 0.523]**
  - This is a data-entry fix whose direction is known (the scores go down), so applying it before the record breaks no rule. Re-run the tool after the fix.
- Minor: the "aluminum trihydroxide" wording variant misses the British spelling "aluminium trihydroxide (ATH)" (old rows N073-2/-3). Those rows stay prior-imputed, as before.
- fr_structures additions: 4 names accepted (Fyrol-6, TEP twice, pentaerythritol), with correct SMILES. No fr_* struct feature changes on any old row.
- The raw `b.write(a.read())` append works: the base file ends in CRLF and the columns align. But it is the raw-text CSV write that banned_moves warns against. Use the csv module.

## Old rows: the declared side effect is not the only one
I compared all 51 exp092-config features for the 460 old benchmark rows between the self-test build and the expanded build.
- **47 old rows in 16 papers change, not 2.** N069-2/-4 are the declared ones: DAP is now resolved, el_imputed_share goes 1 -> 0, and P and N change.
- The other 45 rows (M001, M032, M034, M045, M054, M055, M063, M089, M091, M155, M160, N022, N023, N056, N073) change through **the F7 class-prior table, which is a median over all FR mentions in all rows, wave 4 included** (429 -> 544 mentions). Examples:
  - P-reactive P goes 18.42 -> 10.90.
  - P-N N goes 14.44 -> 21.21.
  - The halogen class goes P 9.46 / hal 32.47 -> 0 / 19.41.
  - metal_oxide/salt goes 0 -> 46.48.
  - The global fallback P goes 0 -> 9.20.
  - So every old row with an unresolved FR in those classes gets new el_P / el_N / el_halogen / el_metal and syn_* values. M032-5's el_P goes from 1.42 to 0.
- No LOI is used, so this is not label leakage. And it is a fixed rule, not a choice, so "no model or feature choice depends on new rows" is technically still true. But "exp092 config unchanged" holds only for the code: the features of old rows now depend on what the new papers contain.
- Measured effect:
  - Freezing old rows at their old features: 0.3820 [0.300, 0.519], vs 0.3812 as scored.
  - Old 460 rows with the shifted priors: 0.3749, vs 0.3738.
  - So about 0.001. Negligible, but it must be declared. Every future wave will shift these priors again.
- Old y are identical, and the old row_ids are all still in the benchmark.

## Spot-check against the sources (9 papers, 43 rows)
Sources: Europe PMC fullTextXML (W021 PMC6647972, W036 PMC11064040) and oa_url PDFs (W002, W003, W004, W008, W010, W014, W016). I picked papers QA did not cover, except W036.
- W021: LOI 19/23/27 (Table 2), densities 30.29/50.65/82.65 (Table 3), PTFM 0/50/100 wt% of castor polyol. Correct.
- W036: 8 foams, LOI 19/23/24/26 ×2, densities 47.3/68.5/69.0/61.7/48.6/69.3/64.5/57.7. Correct. The 6 foams with LOI "-" are correctly left out.
- W002: LOI 19.4/24.1/21.3/23.9/22.9/22.1 (Table 4), densities (Table 6), APP/PMA grams (Table 1). Correct.
- W003: P-1..P-6 LOI 21.8/21.5/21.2/21.2/21.3/20.7, PN-EN ISO 4589-2, no FR. Correct.
- W004: RPUFc-75..150 LOI 17.8/17.8/17.8/18.0, synthetic control 18.5, ASTM D2863-97. Correct (from text).
- W008: LOI 21.9/21.9/23.8/24.2/24.4/24.6, densities 59.7/63.7/79.0/83.1/70.5/85.5. Correct.
- W010: unexposed foam LOI 20.4, density 73.17, described as "sztywna" (rigid). Correct. The exposed-foam LOIs are correctly excluded.
- W014: P3.0 24.1 / P3.6 26.0, densities 50.19 / 33.07. Correct.
- W016: RP150 and RP250 series LOI 22.4/22.0/21.8/23.5/23.2/23.0, flax 3/6 wt%, ISO 4589-2. Correct.
- 0 errors in 43 rows, consistent with qa4 (0/68).
- Duplicates: no new row shares an (LOI, density) pair with any existing row.
  - The existing data does contain cross-paper duplicates: M071/M082 (4 foams), M058/M062, M042/M044.
  - These are not caused by this update, but they leak under paper-grouped CV. Worth a separate audit.

## Why ci_lo jumped 0.20 -> 0.30
- The record CI is the paper bootstrap of **repeat 0's OOF only**.
  - Old snapshot: repeat 0 was below average (0.357 vs a mean of 0.374).
  - New snapshot: repeat 0 is the second-best repeat (0.419 vs a mean of 0.381). Adding papers reshuffles GroupKFold, so repeat 0 is a different split.
- ci_lo over all 10 repeats:
  - old: 0.072-0.274, median **0.200**
  - new: 0.129-0.326, median **0.214**
- The pooled-over-repeats CI moves [0.213, 0.488] -> [0.238, 0.490].
- So the real lower-bound gain is about +0.01 to +0.025. About +0.08 of the +0.095 jump is repeat-0 luck.
- The narrower width (0.269 -> 0.224) is roughly what 83 -> 102 papers should give (sqrt(83/102) = 0.90).
- Wave-4 LOI has lower variance than the old rows (7.8 vs 12.5), with a mean of 22.2 vs 23.2.
- In the 571-row model, the old rows score 0.372 and the new rows 0.386. The mean is essentially unchanged: +0.007 on a different population.

## Is "the expanded snapshot becomes the benchmark of record" applicable?
- Yes, under data_pass_oct4's snapshot_rule. It was pre-declared, it applies regardless of direction, the exp092 and exp093 configs ran unchanged, and this critic precedes any record change. Conditions:
  - (a) Record Arnav's "re-score now" in the repo.
  - (b) Fix the kaolin/flax blanks and re-run the tool. Expected: exp092 config about 0.378 [0.289, 0.511], c3 about 0.371 [0.277, 0.523].
  - (c) Report the pooled CI next to the repeat-0 CI.
  - (d) Disclose the prior shift on old rows.
- It is not a comparison. 0.374 -> 0.38 is a different population, not an improvement, and the record row should say so.

## Caveats to carry
1. Fix `^kaolin` / `^flax fib` to 0s (9 rows) and re-run. The numbers drop by about 0.004. Do not record 0.381 [0.296, 0.520] as it stands.
2. The ci_lo of about 0.29-0.30 is mostly repeat-0 luck. The median per-repeat ci_lo is 0.21, and the pooled CI is [0.238, 0.490]. Always quote the pooled CI.
3. F7 class priors are fitted on all rows, so each wave changes old rows' features (45 rows here, effect about 0.001). Either declare this or freeze the priors.
4. The R4 vs N039 inconsistency, and the missing in-repo authorisation for the run.
5. Log labels "460" and "Reproduction ... True" are stale or false on expanded runs. Fix them before the next wave.
