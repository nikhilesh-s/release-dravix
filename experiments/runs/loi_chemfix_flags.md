# loi_snapshot_update3_chemfix: flags and disclosures (worker, 2026-10-08)

## FLAG 0 (found after the score; needs an orchestrator/critic decision before the result becomes the record)
The corrected tables expose an existing unit-parsing bug on 2 benchmark rows (paper M123, Roflam F5):
- `fr1_unit` = `g (1.5 wt% P in foam)`, `fr1_loading` = 56.1 (M123-1) / 62.5 (M123-2): grams, with the P content in
  the parenthesis. `build_loi_rpuf.basis()` (snapshot-pipeline patch included) sends any unit containing "wt% p" to the
  P basis, so the gram amount is read as wt% P: w = 56.1 / 8.5 x 100 -> `total_fr_wtpct` 660 / 735 and `el_P` 56.1 / 62.5
  (benchmark maximum before: 70 and 16.7).
- Before the fix this was masked: Roflam F5 had no element entry, so "P-basis without P fraction" left the loading
  unparsed (NaN). C1 rule 1 (paper-stated P 8.5) unmasks it. The other 9 newly parsed "wt% P" rows are plausible
  (M126-1 27.2 wt% FR at 4 wt% P; M004-4..11 4.9-22.8 wt%).
- Effect seen (diagnostic re-run of protocol C only, `loi_chemfix_diag_protocolC.log` (script included), no result changed): exp113-config C goes
  0.248 -> -1.254 because Ridge extrapolates on these two rows (Ridge member C -0.98 -> -15.26; RF 0.25 -> 0.27,
  ET 0.32 -> 0.36; M123-1/-2 mean |blend error| 61 / 69 LOI units). Their effect on B and A was not isolated (that would
  be a second score).
- Not fixed here: the config allows only REPLACE extensions and one re-score. Precedent for a fix is the
  loi_snapshot_update2 basis() amendment ("BUG FIX, not a choice", declared before one re-score). Candidate fix for
  the orchestrator to declare or reject: units of the form `g (<x> wt% P in foam)` -> P basis with loading x (here 1.5),
  or leave such loadings unparsed.

All other items below were decided and written before any chemfix score existed; none of them stopped a piece.

## FLAG (non-blocking): two code changes beyond "a stage may REPLACE entries"
1. `data/v2/train_loi_rescore_r8.py`: one added print line, `<cid> per-repeat B: ...` (10 values per config).
   The config asks for the paired per-repeat difference vs exp115, and the trainer printed only mean +- std, so the
   comparison could not be made without it. Output only: no feature, model, seed or fold change; the self-test after the
   edit reproduces exp115/exp116 (see below), and its exp115 per-repeat values equal `round10_loi_oof.npz`
   `inc_per_repeat_r2` to 1e-4.
2. `data/v2/rescore_loi_snapshot.py`: besides `replace_entries()` (the allowed REPLACE extension), the scoring subprocess
   now inherits `LOKY_MAX_CPU_COUNT` if it is set (the orchestrator capped this worker at 3 cores; the tool otherwise
   passes a fixed environment). Unset -> identical behaviour. RF/ET results do not depend on the core count.

## Config wording vs the legacy table (moot here)
3. C2 lists "silica" among legacy zero-convention minerals, but `fr_element_table.csv` counts silica aerogel Si (46.74),
   and the table has no talc entry. No effect: both resolved silica rows are confidence low, so the legacy entry stays.

## Interpretations applied (stated in `data/v2/loi_rpuf_parts/chemfix_staging/README.md`)
4. Blank element columns in a resolved row that has an element basis count as 0 (16 rows, listed in the README);
   basis `nominal_zero` = all six 0 (adds GO x2); basis `none` gives no fractions (rule 2/3 apply, unless LEGACY CONFLICT).
5. LEGACY CONFLICT names with confidence low (L-HNT, MEG silicon-boron) keep their legacy entries: rule 1 is defined for
   high/medium rows only and low rows are "not used anywhere". ADPO2 (conflict, low, unresolvable) keeps its legacy entry (C4).
6. C1 applied to the 196 snapshot-3 names only; the 6 resolved names outside snapshot 3 would get the same numbers or none.

## Disclosed side effect (pipeline arithmetic, not a choice)
7. "wt% P" loadings are converted with the element table's P fraction (`build_loi_rpuf.basis`). 11 benchmark rows now parse
   (M126-1, M123-1/-2, M004-4..11), so their `total_fr_wtpct`, `cls_*` and dependent features become defined; 3 rows
   (W012-5/-6/-7, PFA) change loading value. F7 class priors move as in loi_snapshot_update2.

## Process
8. The tool writes `experiments/runs/rescore_loi_train_loi_rescore_r8_snap606.log`, which is the exp115 record log, and
   copies `loi_snap606_*.json` into `experiments/results/`. To leave both untouched, both runs were made from a scratch
   mirror of the repo (same files the tool itself copies); the logs were copied to `experiments/runs/loi_chemfix_*.log`
   and the result JSONs rewritten as `loi_chemfix_*.json`.

## Results (for reference; numbers in experiments/results/loi_chemfix_*.json)
- Self-test after the extension (no chemfix stage): exp113 config B 0.3875 [0.3124, 0.5289], A 0.717, C 0.248;
  exp114 config 0.3864 [0.3077, 0.5462]: exp115/exp116 reproduced exactly (`loi_chemfix_selftest.log`).
- Chemfix: exp113 config B 0.4052 +- 0.0403 [0.3030, 0.5409], pooled [0.262, 0.506], A 0.721, C -1.254;
  exp114 config B 0.3914 +- 0.0524 [0.3329, 0.5521], pooled [0.224, 0.504], A 0.705, C -1.701 (`loi_chemfix_rescore.log`).
- Paired per-repeat B vs exp115: +0.0177 mean, 10/10 higher; proxy-free vs exp116: +0.0050, 6/10 (report-only).
- Exit triggers (ci_lo > 0.84 or A > 0.84): none. Diagnostic: `loi_chemfix_diag_protocolC.log`.

## FLAG 0 resolution (orchestrator amendment a61c218) -> chemfix2
- `chemfix_staging/unit_corrections.csv` (M123-1/-2 `fr1_unit` 'g (1.5 wt% P in foam)' -> 'g (per 100 g polyol)'),
  applied by `rescore_loi_snapshot.correct_units()` only with this stage (old value asserted; every other cell checked
  unchanged). Label-free build: M123-1 total FR 19.27 wt%, el_P 1.64; M123-2 21.01, 1.79; no other row changes.
- `train_loi_rescore_r8.py` also writes the exp113-config protocol-B OOF (10 x 606) to
  `experiments/runs/loi_rescore_r8_r8d_oof.npz` in the working copy (output only; requested for the critic), copied
  out as `experiments/runs/loi_chemfix2_*_oof.npz`.
- Self-test repeated without the chemfix stage, then one re-score: `loi_chemfix2_*` results and logs.
- `build_chemfix_stage.py`: sort made deterministic for case-tied names (`melamine` / `Melamine` row order in
  `precedence.csv` only).

## chemfix2 results (amendment a61c218; candidate record after a fresh critic)
- Self-test without the chemfix stage: exp113 config B 0.3875 [0.3124, 0.5289], A 0.717, C 0.248; exp114 config 0.3864
  [0.3077, 0.5462]. Exact; OOF equals `round10_loi_oof.npz` `oof_incumbent` to 1e-14 (`loi_chemfix2_selftest*.{log,npz}`).
- Re-score: exp113 config B 0.4123 +- 0.0390 [0.3389, 0.5481], pooled [0.275, 0.513], A 0.727, C 0.082;
  exp114 config B 0.4060 +- 0.0390 [0.3424, 0.5512], pooled [0.251, 0.510], A 0.710, C -0.073.
- Paired per-repeat B (report-only): vs exp115 +0.0248, 10/10 higher; proxy-free vs exp116 +0.0196, 9/10.
- Protocol C is still below exp115 (0.248 -> 0.082; proxy-free 0.171 -> -0.073). Not diagnosed further (report-only metric).
- M123-1/-2 in the scored build: total_fr_wtpct 19.27 / 21.01, el_P 1.64 / 1.79. Exit triggers: none.
- OOF repeats 0-9, exp113 config: `experiments/runs/loi_chemfix2_exp113cfg_oof.npz` (row_id, groups, y, oof, per_repeat_B;
  recomputed per-repeat R2 equals the log).
