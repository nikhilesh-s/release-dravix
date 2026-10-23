# Critic: LOI round 4, candidate c2 (F6 base foam + F7 class-prior el_* fill + F8 synergy) vs exp087
Verdict: **SOUND WITH CAVEATS**. The pre-declared rule is met and no leakage was found, so c2 can replace exp087 as
the LOI number of record. The gain is small: the paired paper-bootstrap CI only just includes 0. Report it as +0.04 with that CI.
## Pre-declaration
- loi_round4.json was committed at 14:31:38 and the run log was written at 14:37. The script mtime (14:33) is after the commit, but the run follows the config. Results were committed at 14:41. OK. exp087 reproduces exactly: B 0.3109, CI [0.1459, 0.4666].
## (1) Leakage
- F6 (pt_*, iso_index_num, is_PIR) comes only from the polyol, isocyanate, iso_index and notes text. None of it is derived from LOI or from the neat control.
  The polyol type is constant within a paper for 80/83 papers, so it is paper-level metadata. Under GroupKFold by paper it cannot memorise a test paper, so it is not leakage.
- F7 priors are per-class medians of element mass fractions over 429 resolved FR mentions, taken from fr_element_table/fr_structures. They do not use LOI. They are computed on all rows and not per fold, but they contain no target information, so this is acceptable.
- neat_loi (the paper's neat-control LOI) appears only in the 'ba' variant. That variant is logged as not selectable and is not in FEAT_C2. Verified in the code.
- F8 is arithmetic on el_*, cls_EG, total_fr_wtpct and density. No LOI.
## (2) Paired gain vs exp087 (seed 0)
- Per-repeat dB: +.008 +.024 +.049 +.014 +.085 +.029 -.002 +.037 +.098 +.092. Mean **+0.043**, 9/10 wins.
- Paper bootstrap of the repeat-averaged gain (2000 resamples): **95% CI [-0.004, +0.102]**, P(dB <= 0) = 0.036. The interval includes 0 by a hair, so the gain is directional rather than established at 95%.
- The gain is not driven by a single paper. Dropping each of the top-3 SSE-reduction papers (N022, M058, M046) leaves dB at +0.033 / +0.034 / +0.038.
- Ablation: F7+F8 without F6 gives B 0.341 (dB +0.030, 8/10). F6 alone (c1) gives dB +0.026. The two parts each add a little.
## (3) RF seed stability (exp087 and c2 both re-run per seed)
| seed | exp087 B | c2 B | dB | wins | c2 rep0 CI | paired CI |
|---|---|---|---|---|---|---|
| 0 | 0.311 | 0.354 | +0.043 | 9/10 | [0.169, 0.466] | [-0.004, +0.102] |
| 1 | 0.313 | 0.352 | +0.038 | 9/10 | [0.176, 0.469] | [-0.009, +0.097] |
| 2 | 0.311 | 0.351 | +0.039 | 9/10 | [0.171, 0.466] | [-0.009, +0.097] |
| 3 | 0.314 | 0.351 | +0.038 | 9/10 | [0.178, 0.472] | [-0.010, +0.095] |
The result is stable across seeds: the rule passes on every seed. Seed 0 has the largest dB of the four, so the expected gain is about +0.039.
## (4) Regex polyol typing: 20-string sample plus a scan of all 117 unique strings
The regex is about right on named commercial grades (Rokopol, Voranol, LY-4110, Stepanpol, Isoexter). Systematic errors:
- Bio-first precedence: any blend with a bio component is typed bio, even a minor one. This includes the "RF-551 66.8 g + EPB **0 g**" control (pure polyether) and the soybean 15 wt% + polyether blend.
- Polyether under-recall: 80 rows are 'unknown', and many of them are polyethers (RF-551 without "Rokopol", PTMEG, PPG, PEG, Laprol oxypropylated, oligoetherols, Voracor). Glycolysed-PET oligodiol (an aromatic polyester) is also 'unknown'. PLA-polyol blends are typed polyether.
- "aromatic" sends "heteroaromatic (isocyanurate-based) polyether polyol" to polyester_aromatic, and the same string sets is_PIR=1.
- In iso_index, "70 (R 3.7:1, PU/PIR)" parses to 0.70 instead of 3.7.
These errors add label noise to F6 but do not leak. Fixing them is a separate pre-declared round, not a change to c2 after the fact.
## (5) Per-repeat and pooled CIs (c2)
B 0.354 ± 0.043. Repeat-0 CI [0.169, 0.466]; pooled-over-repeats CI [0.178, 0.479]. exp087 rep0 CI [0.146, 0.467], pooled [0.109, 0.442]. The c2 and exp087 CIs overlap almost completely, which is why the paired CI above is the right comparison.
## Caveats to carry with the number
1. The paired gain is about +0.04 with a 95% paper-bootstrap CI that just includes 0. Do not call it a significant improvement.
2. Protocol C (held out by FR class) gets worse: 0.345 -> 0.257. The class-prior fill helps within-class transfer and hurts cross-class transfer. Report C next to B.
3. F6 typing is noisy (item 4).
4. B is still far below the Sun/Fu reference of 0.84. Never phrase this as beating it.
