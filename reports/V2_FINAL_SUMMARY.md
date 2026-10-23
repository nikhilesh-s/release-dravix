# Dravix V2 — final summary (2026-10-03 lock; docs touched 2026-10-04)

Source of truth: `experiments/numbers_of_record.json`. Every number below reproduces from a fresh clone with
`sh data/v2/run_numbers_of_record.sh && python3 scripts/check_numbers_of_record.py` (≈ 7 min; THR needs the FSRI clone in `data/v2/raw/`).
R² is out-of-fold on repeated GroupKFold (5 folds × 10 repeats), grouped by material (FSRI), source paper (LOI) or resin system (Pomázi reconstruction). 95% CIs are group bootstraps. A competitor bar counts as beaten only if the protocol is comparable or harder **and** the CI lower bound clears it; comparability is decided by humans.

## 1. Numbers of record

| Target | Dravix R² [95% CI] | Data | Competitor bar | Status |
|---|---|---|---|---|
| Peak HRR | **0.611 [0.516, 0.838]** (exp074) | FSRI, 191 rows / 64 materials | 0.30 (Pomázi 2026) | Ahead, CI clear, **different population** |
| Time to ignition | 0.313 [0.071, 0.496] (exp046) | FSRI, 681 rows / 76 materials | 0.58 (Pomázi 2026) | Behind; parked |
| Total heat release | 0.953 (exp045) | FSRI | 0.67 (Pomázi 2026) | Not like-for-like (near energy-balance identity) |
| Char residue (cone) | 0.650 [0.439, 0.789] (exp117) | FSRI, 567 rows / 61 materials | 0.82 (Pomázi 2026) | Behind (CI upper bound below bar) |
| LOI | 0.388 [0.312, 0.529] (exp115) | RPUF literature reconstruction, snapshot 3: 606 rows / 107 papers | 0.84 (Sun & Fu 2026) | Behind; first same-population number |
| FIGRA | 0.498 [0.346, 0.648] (exp119) | FSRI, 197 rows / 67 materials | 0.9724 (Phan & Morgan) | Not comparable (different population) |

## 2. Ahead but not established
- **Peak HRR (FSRI).** The CI clears 0.30 on a material-grouped protocol, which is harder than the competitor's random split. It is a different population (building products vs. epoxy composites), and with whole material families held out the model is not ahead (0.392 [−0.149, 0.529]). Critic: `experiments/critic/phrr_round1_critic.md`.
- **Pomázi reconstruction.** 72 epoxy / carbon-fibre samples rebuilt from the BME group's open papers and Toldy's 2017 thesis, inputs derived by a mixture rule validated to ≤ 0.02 points against their published table. Under their random-split protocol, R² is above their published held-out values for pHRR (0.604 vs 0.30), TTI (0.615 vs 0.58) and THR (0.828 vs 0.67); char residue is below (0.807 vs 0.82). With formulations kept together, pooled CIs clear for pHRR [0.382, 0.734] and THR [0.753, 0.914]. With whole resin systems held out, no CI clears. **Decision (Arnav, Oct 2): a reconstruction is not their dataset; status "ahead on reconstruction, not established" until their data arrives.** Critic: `experiments/critic/pomazi_round1_critic.md`.

## 3. What changed in this sprint (Oct 1–3)
- Two numbers of record (char residue, FIGRA) were found to predate a Sep 29 data fix and were re-baselined on current data (Arnav, option A).
- Peak HRR promoted from exp043 (RF) to exp074 (HGB) after merging a duplicate material (PlasticC = HIPS). New artifact `models/v2/phrr_hgb_v4.joblib`; v3 kept.
- LOI moved from 38 pure polymers (wrong population) to an independently compiled rigid-PU-foam set (exp064, 0.351), then to exp085 (0.363) by adding PubChem/RDKit structure features for the 16 flame retardants that resolve to a single molecule. The gain is small (paired CI [−0.002, +0.030]). A third extraction wave through Europe PMC added 27 papers (460 rows / 83 papers); under the pre-declared rule the same model was re-scored on the larger set and adopted even though R² fell to 0.311 [0.146, 0.467] (exp087): the first 56 papers flattered the model slightly. Better features, not more papers, are the lever: on Oct 3 (LOI reopened by Arnav after the freeze) one pre-declared round added base-foam type (polyol class, isocyanate index, PIR flag), class-prior element fill and P/N synergy terms, raising R² to 0.354 [0.169, 0.466] (exp089; under the competitor's random split 0.700 vs their 0.84). The gain's paired CI just includes 0 and the model is weaker on unseen FR classes (0.257); critic `experiments/critic/loi_round4_critic.md`. A second approved round fixed the polyol typing the critic flagged and added the LOI test standard and log density: 0.374 [0.201, 0.470] (exp092; random split 0.723 vs 0.84). About two-thirds of that last gain comes from the test-standard feature, which partly acts as a lab proxy; over both rounds the gain vs exp087 is +0.063, CI [+0.017, +0.133] (`experiments/critic/loi_round5_critic.md`).
- Every round was pre-declared in `experiments/configs/`, results go through one tracker writer (`scripts/update_trackers.py`), and any CI above a bar stopped work for a human decision (`blockers/resolved/`).

## 3b. Workflow before / after (plan §2)

| | Sep 27–30 (before) | Oct 1–3 (after the Fable plan) |
|---|---|---|
| Experiments logged | 54 in 4 days | 38 (exp055–exp092) |
| Rounds pre-declared in a committed config | 0 (prose only) | 13 of 13 |
| Repeats of a known dead end | 6 thermal-property joins (11% of runs) | 0 (`experiments/banned_moves.md`) |
| Wins announced, then retracted | 1 (exp006, corrected next day) | 0 — 5 possible claims stopped for a human decision |
| Stale or non-reproducible numbers of record | undetected | 2 found and re-baselined; 42/42 reproduced at the lock; 48/48 after V3 (Oct 4) |
| Tracker corruption / parallel-session renumbering | exp053 row broken; 5 experiments renumbered | single writer (`scripts/update_trackers.py`), validated on every run |
| Blockers surfaced to a human | outreach emails sat unsent 3 days | 7 blocker files, each with options and a recommendation; 5 resolved within hours (plus 1 recorded human decision reopening LOI) |

## 3c. V3 one-day push (Oct 4, after the lock; plan `plans/V3_PLAN_2026-10-04.md`)
V2 was tagged `v2-oct3` at `d7d65b4` first, so nothing below changes the V2 release. Three pre-declared rounds, each with a critic (all SOUND WITH CAVEATS):
- **LOI round 6** (`experiments/critic/loi_round6_critic.md`): with the test-standard (lab-proxy) feature removed, LOI is **0.359 [0.183, 0.474]** (exp093). Adding the critic's polyol-typing fixes gives 0.342 (exp094), so the honest proxy-free range is 0.34–0.36. The "fixes" score lower mainly because of one paper (M079): the "unknown" label had been acting as a cluster tag. Nothing beat exp092, which stays the record at 0.374.
- **LOI round 7** (`experiments/critic/loi_round7_critic.md`): a different, practical task. Given the paper's own untreated-foam LOI, the model predicts flame-retarded LOI with **0.376 [0.102, 0.538]** (exp100; 313 rows / 63 papers), against 0.296 on the same rows without the control (10/10 paired wins, CIs overlap). Supplementary only; never compared to Sun & Fu.
- **TTI round 1** (`experiments/critic/tti_round1_critic.md`): a critical-heat-flux term, a 1/√t target, and HGB variants. None beat exp046 (best 0.298, 0/10 paired wins). R² on the 25 kW/m² rows stays ≤ 0.06 across three model families (repeat 0) and bench-derived physics does not fix it. The weakness is not confined to 25 kW/m²: within 50 or 75 kW/m², exp046's log R² is about 0 and its pooled 50+75 R² is 0.368; the 25 kW/m² rows dominate the headline because they hold 85% of the seconds variance (post hoc diagnostic, lever sweep tti-1). TTI parked (`blockers/BLOCKER-20261004-1156-tti-round1-no-gain.md`).

## 3d. Oct 5: more data (`experiments/configs/data_pass_oct4.json`, `loi_snapshot_update2.json`, `tti_round2.json`)
- **LOI snapshot 2.** 20 new open-access papers (OpenAlex search, 37 readable of 109 found; blind QA 0 errors) -> 571 rows / 102 papers. The exp092 config, unchanged, scores **0.378 [0.289, 0.511]** (exp108; was 0.374 [0.201, 0.470]); proxy-free 0.371 [0.277, 0.523] (exp109). The mean barely moves (+0.004, noise); most of the CI-lower-bound jump is repeat-0 bootstrap luck (median ci_lo over 10 repeats 0.200 -> 0.214). Random split 0.710. Critic SOUND WITH CAVEATS after one bug fix (kaolin/flax element entries). 72 more OA papers blocked automated access -> `data/sources/LOI_LIBRARY_PULL.md`.
- **LOI round 8** (`experiments/critic/loi_round8_critic.md`, SOUND WITH CAVEATS): averaging RF, ExtraTrees and Ridge on the same features gives **0.425 [0.312, 0.552]** (exp113; 9/10 paired wins over 0.378), random split 0.736. Caveats: a third of the gain is one paper (M079); held out by FR class it drops to 0.195 (vs 0.237); selection-adjusted gain is small (~+0.006 lower bound). New record.
- **LOI snapshot 3** (`experiments/critic/loi_snapshot3_critic.md`, SOUND WITH CAVEATS): wave 5 (non-English OpenAlex, Crossref, Semantic Scholar; 6 readable papers, 35 rows, QA 0 errors) -> 606 rows / 107 papers. Same model, **0.388 [0.312, 0.529]** (exp115), lower than 0.425 because the new rows have little LOI spread and the folds reshuffle (snapshot 2's 0.425 was partly luck). The pre-declared rule makes the bigger snapshot the record anyway. A unit-parsing bug (polyol+isocyanate, equivalent ratios) found by the critic was fixed first. Proxy-free 0.386.
- **TTI round 2** (FSRI integrating-sphere emissivity, 42 of 76 materials): no gain (best 0.283 vs 0.313, 1/10 wins). Emissivity carries no detectable signal even on the covered materials; the overall drop is ~85% the missing-value indicator. TTI stays parked. NIST FCD has no cone-scale data.

## 3e. Oct 6: blend round 1 (`experiments/configs/blend_round1.json`, critic `experiments/critic/blend_round1_critic.md`)
The LOI round-8 averaging recipe (incumbent + ExtraTrees + Ridge; and incumbent + ExtraTrees), fixed in advance, tried once on four targets.
- **Char residue: 0.549 -> 0.650 [0.439, 0.789]** (exp117, 10/10, critic SOUND). Robust: positive in every repeat, holds without the top 5 materials and with whole categories held out. Still below 0.82.
- **FIGRA: 0.456 -> 0.498 [0.346, 0.648]** (exp119, 10/10, SOUND WITH CAVEATS): recorded per the rule, but the gain is small and not established (it vanishes without the top 5 materials).
- TTI 0.332 (7/10, not adopted; R2 at 25 kW/m2 0.06 -> 0.13). pHRR worse (0.560).

## 4. Parked
TTI and THR on FSRI (coverage-limited; THR is near an identity; TTI round 1 on Oct 4 found no gain). FIGRA two-stage model (−2.79). LOI: round budget (3) plus both human-approved extra rounds used.

## 5. Open human actions
- Emails to Andrea Toldy (their 75-sample database: would turn the reconstruction into a like-for-like test) and Madeleine Bussemaker.
- Optional library pulls for LOI: `data/sources/LOI_LIBRARY_PULL.md`.

## Artifacts
Figures: `reports/figures/v2_final/`. Model cards: `reports/model_cards/`. Demo: `python3 demo/predict.py --material PMMA --heat-flux 50`. Same-polymer FIGRA table: `reports/figra_same_polymer_table.md`.
