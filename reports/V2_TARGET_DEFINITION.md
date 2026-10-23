# Dravix V2 — Prediction Target Definition (Task 3 / Plan Step 4)

Decided 2026-09-27, based on the verified SOTA board (`benchmarks/sota_board.csv`) and the two data-source catalogs (`data/sources/leads_hunterA_papers_institutions.md`, `leads_hunterB_repositories.md`). Rule from the plan: never claim a paper is beaten unless the comparison is genuinely apples-to-apples. Everything below is chosen to make that rule easy to honor, not to inflate a headline number.

## Why not one single target

Real data volume differs enormously by target right now, and the competitive bar is not uniformly hard — in fact one of the SOTA Scout's biggest findings is that some "impressive" published numbers collapse under true held-out evaluation. So targets are tiered by (a) how much real data we can actually get before Oct 3, and (b) how honest/beatable the existing published bar actually is.

## Tier 1 — Primary target: Limiting Oxygen Index (LOI)

- **Definition:** vol% oxygen concentration at which a material sustains candle-like burning (ASTM D2863 / ISO 4589). Continuous scalar, standardized test, the single most commonly reported flammability metric across every candidate paper we found (USTC, Surrey/Nanjing×2, IMDEA, and our own legacy corpus already carries an LOI column).
- **Competitor to beat:** `sun_fu_2026_rpuf` (Sun, Fu, Hao, Bussemaker, Jiang, Zhou, Liu — Polymer Degradation and Stability 244:111838) — **VERIFIED**, XGBoost, LOI R²=0.84, 435 RPUF formulations, 348/87 train/test split. This is a real, confirmed, comparable number — the cleanest target to chase first.
- **Why primary:** most cross-paper comparable, most existing real rows in hand already (legacy corpus has an LOI column, even if its provenance for the *resistance target* is weak — LOI itself as a raw lab-measured property is a different, more trustworthy claim than the constructed resistance score), and a real verified bar that looks beatable with a genuinely grouped (formulation-family-held-out) split.
- **Data sources:** existing legacy descriptor corpus (LOI column, needs provenance check — task 4-adjacent), email requests to Sun/Liu (Surrey-Nanjing, 435 rows) and De-Yi Wang (IMDEA), USTC group (Li et al., pending author response).

## Tier 2 — High-leverage targets: TTI, Peak HRR, THR, Char Residue

These have the **weakest genuinely-honest published bars found so far**, which makes them the highest-leverage targets even with modest data volume:

| Target | Competitor (Pomázi et al. 2026, BME, real held-out numbers) | Real data sources |
|---|---|---|
| Time to Ignition (TTI) | R²=0.58 (their best, ANN) | FSRI MaP DB (cone calorimeter), MaCFP matl-db, Toldy group's own 79-row set |
| Peak HRR (pHRR) | R²=0.30 (their best, RFR) — the weakest real bar on the entire board | FSRI MaP DB, MaCFP matl-db |
| Total Heat Release (THR) | R²=0.67 (their best, XGBoost) | FSRI MaP DB |
| Char Residue (CR) | R²=0.82 (their best, RFR) | FSRI MaP DB (via STA mass-loss), MaCFP matl-db |

**Critical methodology note carried over from task-2/SOTA-scout findings:** Pomázi's abstract-level numbers (0.96 pHRR, 0.95 TTI) are full-dataset fits including training data, not real generalization — do not cite those as the bar. The bar is the paper's own Table 9 held-out numbers above. When Dravix reports its own numbers here, they must be **held-out only**, computed on a split at least as hard as theirs (ideally harder — grouped by material family, not random), or the comparison is invalid by our own rule.

**Why this tier matters:** pHRR at R²=0.30 real-held-out is a low bar. Even a modest real dataset (FSRI + MaCFP combined is probably a few hundred rows once merged) evaluated honestly has a real shot at clearing it — much better ROI than chasing Tier 3 out of the gate.

## Tier 3 — Stretch target: Fire Growth Rate / FIGRA

- **Competitor:** `phan_morgan_2025` — **VERIFIED**, R²=0.9724 (RDKit descriptors), the single most honestly-earned high number on the board (real held-out polymers, clearly described methodology) — but trained on synthetic data with only 15 real polymers in the cone-calorimetry test set.
- **Status:** aspirational. FIGRA needs full HRR-vs-time curves (cone calorimetry), which none of our near-term real sources supply in volume yet. Revisit once Tier 1/2 pipelines are working and if FSRI/MaCFP raw curve data turns out to support FIGRA derivation.

## Explicitly out of scope for now

- **UL-94** — categorical (V0/V1/V2/HB/fail), a different problem type (classification, not regression) from everything else here. USTC and IMDEA both use it; revisit only if Tier 1/2 land early.
- **Flammability Index (FI)** — Phan/Morgan's own target is trained almost entirely on synthetic (SDV-augmented) data with only ~32 real polymers behind it. Not a target we want to imitate the methodology of, even though it's on their SOTA board.
- **`lv_2026`** — not adopted as a target-defining competitor; the SOTA Scout could not find this paper at all across ~10 search variants. Treated as unconfirmed/possibly non-existent per the sota_board.csv note.

## Ground rule carried forward

Any Dravix-vs-competitor row added to `benchmarks/benchmark_results.csv` must specify: which tier/target, which competitor row from `sota_board.csv`, the exact split method used on both sides, and why the split is comparable-or-harder. No row goes in on a full-dataset-fit or random-split comparison against a competitor's grouped/held-out number.

## Next task this unlocks

Task 4 (dataset-lineage audit) and the "Data Engineer" role can now proceed with a concrete target list to canonicalize toward, instead of collecting generically.
