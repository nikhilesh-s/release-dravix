# Critic: LOI chemistry-correction re-score (loi_snapshot_update3_chemfix; exp113 / exp114 configs, snapshot 3)

Verdict: **SOUND WITH CAVEATS.**

- The stage implements C1-C4 as written. I checked all 202 per-name outcomes and all 66 formula-based element rows independently.
- The tool extension does nothing unless the chemfix stage is given. The self-test reproduces exp115 and exp116 exactly.
- The M123 fix corrects a real bug. It was not tuned to the score.
- B, the CIs and the paired difference all recompute exactly from the npz.
- Under the config's rule, the corrected exp113-config result becomes the LOI number of record: **0.4123 ± 0.039 [0.3389, 0.5481]**, pooled [0.275, 0.513], A 0.727, C 0.082. The proxy-free supplementary number is 0.4060 [0.3424, 0.5512], C -0.073.
- The record must call this a data correction, not a model gain. The note has to carry the caveats in the last section.

## 1. Order (git and log times, 2026-10-08 PDT)
| time | event |
|---|---|
| 01:45:12 | config committed (e3ba49a) |
| 02:03 | first self-test |
| 02:15 | first re-score (B 0.4052, C -1.254) |
| 02:21:51 | first results, the stage and the amendment committed together (a61c218) |
| ~02:27-02:41 | second self-test |
| ~02:45-03:01 | single re-score |
| 03:02:36 | committed (63e966d) |

- The config's `declared` text says "~02:05 PDT", but the commit is at 01:45. The text is wrong; the order is still right.
- **The stage files were not committed before the first score.** Only the config was. Two facts limit the damage:
  - The element and structure tables are byte-identical between a61c218 and 63e966d (md5 checked).
  - The first-run log prints the same stage counts (96 placed first, 5 unset, 29 structures).
- Between the two runs, the only input change is `unit_corrections.csv`. The other changes are a deterministic sort for `melamine`/`Melamine` in `precedence.csv`, an output-only npz dump, and README text.

## 2. Stage vs C1-C4 (independent checks in scratch builds; nothing written to the repo)
I built both trees with `rescore_loi_snapshot.py --build-only --keep`: wave4+wave5, and wave4+wave5+chemfix.

- **C1 precedence.** I looked up all 202 names of `precedence.csv` in the built element tables, before and after, using the pipeline's own first-regex-match rule. Every value matches the file's `before_vals` / `after_vals`; there are 0 mismatches.
  - The 5 LEGACY CONFLICT names that have no resolved fractions (BEG, fGO, HALM, IFR blend, MEG DOPO/A-151) resolve to no entry. This costs 14 element-complete rows. That is what the literal rule says, not a choice.
- **C2 mapping.** `resolved_six()` maps exactly as declared:
  - el_halogen = Cl + Br;
  - el_metal = Al + Mg + Zn + the metals in `other_elements_wt_pct` (asserted to be metals);
  - S is dropped.
  - The mineral convention (Si and metal set to 0) is applied to kaolin, vermiculite W, woody ash, HNT-PPO, walnut/halloysite and lavender/kaolinite. The README states the convention per name for all 40 inorganic names before scoring.
  - The config's "silica" example is wrong about the legacy table. The README discloses this, and it has no effect because both silica rows are low confidence.
- **Element arithmetic.** I recomputed the wt% values of all 66 `element_basis = formula` rows from the formula strings using RDKit atomic weights: 0 errors over 0.06 wt%. All 37 SMILES parse, and each RDKit formula equals the stated formula.
- **C3 structures.** The 29 structure replacements are exactly the set with `smiles` + `molecule` + high/medium confidence (24 high, 5 medium; 28 in the snapshot plus AD). No accepted structure is replaced; the tool exits if asked to replace one.
- **Low confidence.** None of the 15 low-confidence names gets a resolved element entry or structure.
- **Feature effect.**
  - Labels, row ids and paper ids are identical, and so are the GroupKFold groups.
  - 86 rows change element, class or loading features; 182 rows change any model feature, including the F7-imputed values.
  - RDKit coverage goes from 56 to 156 rows. Element-complete rows go from 437 to 485.
- **Spot check: 15 names against external sources.** All 15 agree with the CSV.
  - PubChem formula and IUPAC name by CID for AAM-DOPO, DFD, DTE-DOPO, ETA-DOPO, PN-Cl, TBPBP, trisilanol phenyl POSS, TCPP (the TCPP and Roflam PLO rows), Exolit/ADP, Mg(OH)2, and the Roflam F5 representative component.
  - Roflam F5 P 8.5 wt%, from the M123 paper itself (Europe PMC full text).
  - PTOP P 1.85 and PTOSi Si 1.95, from Table 1 of the paper.
  - Roflam B7 is halogen-free per the PCC product page. The paper's own label "brominated" is the error, and the resolved el_halogen = 0 is correct.
  - One open point: Roflam PLO is taken to be the PCC TCPP grade "Roflam P LO" (medium confidence). That is plausible, but the paper does not identify it.

## 3. Tool extension and self-test
- `replace_entries()` and `correct_units()` return early unless the stage has the replace, unset or unit files. wave4_staging and wave5_staging have none of them.
- The other code changes are output-only:
  - a per-repeat print;
  - the npz dump;
  - passing LOKY_MAX_CPU_COUNT to the scoring subprocess.
- **Self-test after the amendment reproduces exp115 and exp116 exactly:**
  - exp113 config: B 0.3875 [0.3124, 0.5289], A 0.717, C 0.248.
  - proxy-free: 0.3864 [0.3077, 0.5462], C 0.171.
  - Its OOF matches `round10_loi_oof.npz` `oof_incumbent` to 1.1e-14. I recomputed this.
  - My own scratch build without the stage reproduces C 0.248 exactly.

## 4. The M123 fix
- **It is a real bug.**
  - `basis()` matches the substring "wt% p" inside the description "g (1.5 wt% P in foam)". It then reads 56.1 / 62.5 g as wt% P, which gives 660 / 735 wt% FR. That is physically impossible; the benchmark maximum is 70.
  - Table 1 of the paper (doi 10.3390/ma16227217, PMC10673181) lists RF-551 100 g and Roflam F5 18.7 / 37.4 / 56.1 g (S1) and 20.8 / 41.6 / 62.5 g (S2). These are grams in a 100 g-polyol formulation.
  - The pipeline already treats a plain "g" unit as polyol basis (165 rows), so the fix follows existing behaviour.
- **The reading is a choice among faithful options. It is disclosed, and it does not matter for B.**
  - The paper defines P "in relation to the weight of the reference foam". That implies a reference foam of about 318 g (S1) and 354 g (S2), so pMDI is about 206 / 247 g, not the pipeline's 130-part default.
  - The paper-consistent loading is therefore about 15.0 wt% FR in the final foam, el_P about 1.3.
  - The scored values are 19.27 / 21.01 wt% and el_P 1.64 / 1.79. These overstate the loading through the same iso-default approximation used on 124 benchmark rows.
  - The worker's alternative (P basis at 1.5) would give 17.6 wt% and el_P 1.5.
  - Both M123 rows are predicted at 21.64 / 21.55 against measured 21.5 / 21.7. Their own squared-error change is about 0 R2. The +0.007 between the first and final run comes from removing two leverage points from everyone's training sets.
  - The amendment was written after the first score, as in the loi_snapshot_update2 precedent. Only one re-score followed, and B was already above exp115 in the first run, so the fix was not needed to clear anything.
- **No other row has this failure mode.** Every unit that `basis()` sends to the P basis:

| unit | rows | loading | verdict |
|---|---|---|---|
| "g (1.5 wt% P in foam)" | M123 ×2 | 56.1 / 62.5 | the bug |
| "wt% P" | M004, W012 ×19 | 0.5-3 | plausible as wt% P |
| "wt% P in foam" | M126 ×1 | 4.0 | plausible as wt% P |

  After the fix, no row exceeds the old maxima (total FR 70, N022-7; el_P 16.7, M032-13).
- **The bug could have been caught before scoring without labels.** The README coverage table listed M123 among the newly parsed rows, but nothing checked their range.

## 5. Recomputation from the npz
`loi_chemfix2_exp113cfg_oof.npz` (10 × 606, 107 papers):
- Per-repeat B: 0.4629 0.3658 0.4398 0.4281 0.4388 0.4382 0.4024 0.4127 0.3226 0.4115. Mean 0.4123, std 0.0390; these equal the stored values.
- `eval_protocol.bootstrap_ci` on repeat 0: **[0.3389, 0.5481]**. Pooled-repeat paper bootstrap: [0.2754, 0.5132].
- Paired against the self-test OOF: +0.0218 / 0.0298 / 0.0228 / 0.0422 / 0.0097 / 0.0632 / 0.0037 / 0.0058 / 0.0112 / 0.0375. **Mean +0.0248, 10/10.**
- Proxy-free paired against exp116, from the logs: +0.0196, 9/10.

**Where the +0.025 comes from** (mean-over-repeats SSE / SST):
- **Concentrated in a few papers.** M034 alone gives +0.0135, 54% of the gain: the phosphazenes PN-Cl, PN-Br and PN-H now have P/N/halogen and RDKit features, and their mean |error| falls from 5.2 to 2.8.
- The top 5 papers (M034, N049, M065, N022, M032) give +0.0294. **Without them the change is -0.005.**
- The 86 rows whose features changed give +0.015. The unchanged rows give +0.010, through F7 priors and training composition.
- Some corrected papers got worse:
  - M004, the 8 newly parsed DOPO/PLF140 rows: -0.006, mean |error| 0.86 -> 1.76;
  - M087, MEG unset: -0.003.
- The CIs overlap heavily: ci_lo is 0.339 vs 0.312. The result remains far below the 0.84 bar.

## 6. Protocol C (0.248 -> 0.082): a Ridge extrapolation artefact, not a loss of transfer
This is a report-only diagnostic. I re-ran C on the scratch builds with the unchanged exp113 models and seeds. It reproduces 0.248 before and 0.082 after exactly.

| features / class grouping | blend | RF | ET | Ridge |
|---|---|---|---|---|
| before / before | 0.248 | 0.250 | 0.322 | -0.976 |
| after / after (scored) | 0.082 | 0.273 | 0.363 | -2.423 |
| after / before grouping | 0.085 | 0.264 | 0.371 | -2.379 |
| before / after grouping | 0.235 | 0.253 | 0.323 | -1.053 |

- **Both tree members transfer better after the correction.** The RF+ET mean goes from 0.313 to 0.343. The whole drop is the Ridge member.
- **The regrouping of the 11 newly parsed rows (none -> P classes) explains at most 0.013.**
- **The loss sits on a few rows.**
  - The held-out intumescent (APP-based) class gives 0.149 of the 0.166 loss.
  - W005 (7 rows) gives 0.094, and W005-3 alone 0.047. Ridge predicts LOI -14 for W005-3; measured is 31.4.
  - Excluding W005 gives 0.271 before and 0.198 after. Excluding W005 and W006 gives 0.295 before and 0.248 after.
  - Ridge's mean prediction is below 10 LOI on 13 rows after, against 0 before.
- **Mechanism.**
  - When the APP class is held out, `cls_intumescent` has only 5 nonzero training rows (sd 0.32). A 40 wt% APP row therefore sits about 120 SD out.
  - Ridge's coefficient on that near-constant feature takes the contribution from -15 to -50 LOI.
  - Restoring the before values of the el_*/syn_* block alone brings W005-3 back from -24 to 0.
  - The W005 rows' own features are unchanged. The corrected element data elsewhere tips an already unstable extrapolation; Ridge's C was already -0.98 before the correction.
- **It does not block the record change.** The config makes C report-only, and the rule applies whichever way B moves. The record must quote C 0.082 (proxy-free -0.073) with this explanation.

## 7. Wording and caveats for the record note
Suggested note:

> Data correction (loi_snapshot_update3_chemfix, owner-delegated): exp113 config unchanged on the same 606 rows / 107 papers, folds and labels. FR element fractions and single-molecule structures were corrected and extended from the 2026-10-08 resolution file (element-complete rows 437 -> 485; RDKit rows 56 -> 156), plus one unit-parsing bug fix (M123-1/-2). 0.4123 [0.3389, 0.5481], pooled [0.275, 0.513], A 0.727, C 0.082; proxy-free 0.4060 [0.3424, 0.5512], C -0.073. Not a model gain: +0.025 paired (10/10), concentrated in M034 (phosphazenes; 54%), and about -0.005 without the top 5 papers. C fell 0.248 -> 0.082 because the Ridge member extrapolates when the APP class is held out (W005 alone gives 0.09; RF and ET C improved). The first chemfix re-score (0.4052, C -1.25, before the M123 fix) is history. Previous: exp115 0.3875 [0.3124, 0.5289].

Also record:
- Authorisation is the owner's general overnight delegation, not an explicit go for this change, which is the same gap the snapshot-3 critic noted.
- Element values from ideal formulas ignore grade and hydration.
- Roflam PLO's identity is assumed.
- M123's loading is overstated by the iso-130 default.

## 8. Things to fix (none blocks the record)
1. Commit stage directories before the first score, not with it. Add a label-free range assertion to stage builders (0 <= total_fr_wtpct <= 100; el_* <= total) so that a case like M123 stops before scoring.
2. Correct the config's `declared` time (~02:05) to the commit time (01:45) in an erratum.
3. Out of scope here, but the same family of bug (a unit cell that disagrees with the source). These rows were identical in both runs, so they do not bias this comparison. They need their own pre-declared unit audit:
   - **M032 (14 rows):** unit "wt%", but the row note says "FR amounts g per 100 g Rokopol RF551" with pMDI 197 g. If the note is right, the loadings are overstated about 2.5-3x, and so is el_P on M032-13 (16.7).
   - **N051 / N052 / N053 (16 rows):** "wt% to polyol" parses as a foam basis.
4. For any future pre-declared round, not now: the blend's protocol C is dominated by Ridge extrapolation. Report the tree members' C next to the blend's.

Diagnostic scripts and builds are in the session scratchpad (`diag/extract.py`, `run_c.py`, `analyse_c.py`). The only repo file written is this one.
