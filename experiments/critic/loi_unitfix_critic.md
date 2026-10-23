# Critic: loi_snapshot_update4_unitfix (LOI unit-basis correction re-score)

Fresh-context critic, 2026-10-08. Subject: `experiments/configs/loi_snapshot_update4_unitfix.json` (declared in
8a9509d at 04:01:47, results in dcb3233 at 04:37:17). I did not re-tune or re-score. I ran two label-free
`--build-only` builds (with and without `unitfix_staging`), recomputed the acceptance rule, recomputed the paired
numbers from the logs and read 6 source papers plus part of a 7th.

**Verdict: SOUND WITH CAVEATS.** Nothing blocks. The caveats below belong in the record note.

## 1. Acceptance rule: followed exactly
- I recomputed the rule from `unit_corrections_proposed.csv` (121 cells, 16 papers):
  - keep every `high` cell;
  - keep `medium` cells only in papers with no medium `*_loading` cell, which excludes N022 and M064;
  - drop `low` (N061).
- The result is 94 cells, 65 rows, 13 papers. It is identical to `unit_corrections.csv`: a row-for-row `equals` check
  passes, and an outer merge leaves nothing on either side. The paper list matches the config.
- Two inclusions follow the rule as written but should be named in the note:
  - **M080** (1 cell, `pph of oligodiol` -> `php`). The audit calls this a parser-gap coverage change, not an error
    fix ("drop it if only true errors should go in"). It turns one unparsed row (NaN) into 9.62 wt%. It went in
    because it is `high`.
  - **M055** changes 3 loading **values** (2/4/6 -> 20/40/60) at `high` confidence. The config excludes loading-value
    changes only at `medium`, so M055 is in.

## 2. Source spot-checks: new basis right in all 6 papers
I read each paper's open-access PMC page and ignored the LOI results.

| paper | source evidence | verdict |
|---|---|---|
| M032 (PMC12300640) | Table 1 in grams: Rokopol 100 g, FR 30 or 15+15 or 30+30 g, MDI 197 g. The text calls these "30% and 60% by weight", which is where the old `wt%` came from. | Polyol basis correct. Under the old foam basis, M032 showed up to 60 wt% FR and el_P 16.7. Now 9-17 wt% (iso parsed from 197 g). |
| M063 (PMC9041321) | HSBP + DOPO-HSBP + T-D add up to 100 in every row. AK8805 3, DBTDL 1 and H2O 1.5 sit on the same scale. Isocyanate is given only as an index of 1.1. | Polyol-blend basis correct. The figures are approximate, see §6. |
| M045 (PMC11314537) | Table 3 in grams: A + B + K-salt + DFD = 100 g in every row. The conclusions say "2 wt.% ... 10 wt.%". | Foam `wt%` correct. The old string parsed as a polyol basis. |
| M072 (PMC7730523) | Table 1 header reads "Parts by Weight (wt.%)": polyol 100, Purocyn B 160, BP 1/2/5. The abstract says "1, 2, 5 wt.%" without a reference mass. | Polyol basis is the better reading of the table. The effect is at most 2 wt%. |
| M042 (PMC11433397) | §2.2: "10 parts by weight based on polyol". | Correct; the source states it explicitly. |
| N052 (PMC8038432) | Table 4 "Content (wt.% to polyol)": polyol 100, SO 1/2/5, pMDI 160. | Correct. The old string was right in meaning; `basis()` misread "to polyol". |
| M055 (partial) | The bcpublication PDF did not extract. The abstract reads "P-polyol amount in white material was 40%", and white material is the polyol side. | Consistent with 20/40/60 wt% of polyol. The table itself was not checked. |

I did not check the medium-confidence papers M110 and V001. For V001, "15% FR" of the B-component has been recorded
as wt% of polyol, which slightly overstates the per-100-polyol loading. `medium` is a fair label.

## 3. Code path: only the accepted cells change
- `rescore_loi_snapshot.py` and `build_loi_rpuf.py` were last changed in 63e966d, before the declaration.
  - `correct_units()` asserts each `old_value`.
  - It checks that the rewritten file reads back equal to the expected frame, with only the target cells changed.
  - It asserts that every fix found its row.
- The run used a scratch mirror (`scratchpad/unitfix_mirror`, created 04:02, after the declaration). Its `data/` tree is
  byte-identical to the repo except `run_numbers_of_record.sh`, which this run does not use. The driver log shows the
  stage order wave4 -> wave5 -> chemfix (2 M123 cells) -> unitfix ("94 unit cells corrected").
- I rebuilt the snapshot both ways, label-free, and compared the two:
  - Exactly 94 raw cells differ, and they are exactly the accepted set.
  - `fr_element_table.csv` and `fr_structures.csv` are identical.
  - Exactly the 65 accepted rows change in the built CSV, and only in loading/feature columns (`total_fr_wtpct`,
    `cls_*`, `el_*`, `loading_ok`, `flags`).
  - `loi_pct` is identical everywhere. Row and paper counts are unchanged at 606 / 107.
- 64 of the 65 changed rows are benchmark rows (M045 has 4 rows, 3 in the benchmark).
- The `BLOCKED: exp089/exp092 not reproduced` lines in `loi_unitfix_rescore.log` are the expected message on an expanded
  snapshot, which is patched to not stop the run. They are not a failure.
- Self-test: `experiments/runs/repro_rescore_loi_snapshot.log` (04:14, between the two commits) reproduces exp159 r8d
  B 0.4123 [0.3389, 0.5481] and exp160 s8 0.4060 [0.3424, 0.5512] exactly.

## 4. Label exposure (audit §6): no plausible bias on acceptance
- **Which papers.** The exposures are a few LOI values, all on the input side, in M032, N051, N052, M042, W005, W006,
  W007, M071 and V005. No model output was seen.
  - Of the accepted papers, only M032, N051, N052 and M042 are exposed. All four are `high` cells whose basis is fixed
    by an explicit table header or sentence; I confirmed three of them independently above.
  - Seeing one LOI value does not tell anyone which loading basis would help a model.
- **The discretionary cells.** These are the medium M110 and V001, and the excluded N022, M064 and N061. None of them
  is among the exposures.
- **Flagging criteria.** All are feature-side: FR > 60, el_P > 10, unparsed basis, unit-string conflicts.
- **Prior per-paper score information.** The trigger papers M032 and N051-N053 came from the chemfix critic's §8.3 row
  notes, which are feature-side. That same critic also published per-paper score contributions naming M032 and N022.
  This is minor: M032's direction is fixed by its gram table, and N022 was excluded.
- **Timeline.** Nothing suggests any score on unit-corrected data before 8a9509d. The audit's scratch only applied and
  diffed the corrections.

## 5. Wording and the wider CI
- **Call it a data correction, not a model gain.** The model is unchanged and B moves from 0.4123 to 0.4253.
  - Paired per repeat (recomputed): mean +0.013, 7/10 repeats up, median +0.008.
  - Most of that comes from repeat 8 (+0.091), which was exp159's weakest repeat (0.323). Without it the mean is +0.004.
  - The change is within repeat noise (per-repeat std 0.03-0.04). Describe B as "essentially unchanged".
  - Proxy-free: 0.4060 -> 0.4339 (paired +0.028, 7/10).
- **CI.** The paper-bootstrap lower bound falls from 0.339 to 0.261 while the upper bound barely moves (0.548 -> 0.550).
  It is not a general loss of precision:
  - the pooled CI lower bound rises (0.275 -> 0.290);
  - the per-repeat std falls (0.039 -> 0.031).

  The bootstrap lower tail depends on which papers are resampled, and 13 papers changed features.
- **Does the wider CI matter under the snapshot rule?** No. The corrected result becomes the record whichever way it
  moves, and the only exit triggers are ci_lo > 0.84 or A > 0.84; neither fires.
  - The record must still quote [0.2607, 0.5503].
  - Any sentence that relied on exp159's lower bound (for example "lower bound above 0.3") must be withdrawn or
    restated.
- **Protocols A and C.** A goes 0.727 -> 0.750. C goes 0.082 -> 0.295 (proxy-free -0.073 -> 0.219). A C swing this
  large from a 64-row feature correction shows how fragile C is. Report it as a data-correction effect, not as an
  improvement.

## 6. Caveats for the record note (none blocking)
1. **The pre-declared 60 wt% check was not reported, and one row fails it.** The config's report list asks for a
   label-free check that no row exceeds 60 wt% total FR. Neither log shows that check. On the accepted snapshot,
   **N022-7 sits at 70.0 wt%**.
   - This is pre-existing: it was in exp159 too, and N022 was excluded by the rule.
   - The audit's §2 statement "no row has total FR > 60" holds only for the full 121-cell proposal, not for the
     accepted 94 cells.
   - The note should state: 1 row > 60 (N022-7, 70.0, unchanged, excluded paper).
2. **Composition of the 94 cells.** They are 93 basis/value corrections plus 1 parser-gap coverage change (M080).
   They touch 65 rows, of which 64 are benchmark rows.
3. **Remaining approximations on the corrected rows.** These are pipeline causes the config declares unchanged:
   - 48 of the 65 changed rows now use the 130-part isocyanate default; only M032 and M045 do not. Benchmark rows on
     the iso default go from 124 to 164.
   - For the reactive-polyol FRs in M063 and M055, the FR sits inside the 100 polyol parts but is added to the
     denominator a second time through `polyol_fr_sum`. That understates their wt% by roughly 20% relative at high
     replacement.
4. **Still unresolved and unchanged.** V004, W005, N077 and M092 (basis not stated); N022, M064 and N061 (excluded);
   plain-`wt%` papers the audit did not source-check (M014, M079, M088, M115, N023, W006, W009, W011, W016). The
   benchmark can still carry basis errors.
5. **`basis()` should be fixed at the root.** The lasting fix belongs in `basis()` / `iso_parts()`: `to polyol`, `pph`,
   `g per 100 g (wt%)`, the iso unit strings and the reactive-polyol denominator. A label-free range assertion should
   run before any future score. This is for a future pre-declared round, not this one.

Not checked: M110 and V001 against their sources; the M055 table itself.
