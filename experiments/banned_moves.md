# Banned moves

Every round config in `experiments/configs/` must state it checked this file.
A move here needs a human decision to un-ban. Sources: `HANDOFF_TO_NEXT_CLAUDE.md` §4, `plans/FABLE_PLAN_2026-10-01.md` §6.

## Data and features
- **Thermal-property joins** (conductivity, specific heat, thermal inertia, handbook k·ρ·c) added on top of the FSRI tables, for pHRR or TTI. Tried five times (exp009/048, 010/049, 012/051, 013/052, 053); the join always shrinks the material pool. Allowed again only with coverage on *more* materials, not fewer.
- **Any feature that shrinks the material pool** for pHRR (exp043's 65 materials is the floor).
- **Excluding a material** from a pHRR / CR evaluation after seeing its influence. Diagnostic tables are fine; acting on them is exit condition 5.
- **Mixing FAA pure-polymer LOI rows** into the RPUF LOI benchmark.
- **HKPoly MaCFP PMMA rows** in cross-lab checks (reads ~2–2.75× other labs; unresolved).
- **STA N₂ residue as the reported Char Residue target.** Canonical CR is cone residue (exp033). STA residue may be a feature.
- **MCC char yield as a feature for an STA-residue target** (near-tautological, r = 0.986).
- **The `max HRR(t)/t` FIGRA definition.** FIGRA is pHRR / t_pHRR from test start (`097ef71`, `figra.md` §3).
- **Re-deriving the STA mg-vs-% fix** from scratch. Read `data/v2/train_char_residue_v2.py` and `reports/sprint_0929/sta_fix.md`.

## Evaluation and claims
- **Picking a model by its CI**, or by a single seed / single split. Selection is by repeated-GKF mean plus the paired 8/10 rule.
- **Changing the win rule or the CI method** after seeing a result. Supplementary CIs never promote.
- **Writing "beats SOTA"**, or setting `comparable: true` in `numbers_of_record.json`. Humans only.
- **Citing THR 0.953 vs 0.67 as a win** (near energy-balance identity; different problem).
- **Claiming anything against FIGRA 0.9724** (different population, 15 neat polymers).

## Process
- **Hand-editing** `experiments/leaderboard.csv` or any xlsx. Only `scripts/update_trackers.py` writes them.
- **Raw string concatenation** into any CSV. Use the `csv` module.
- **Running a round** without a committed config in `experiments/configs/`.
- **Exceeding a round budget** (plan §6). Plateau is a FLAG.
- **A third workaround** for the same access failure.
- **A second top-level session** working the repo in parallel.
- ~~**More compute on TTI or THR** before Oct 3 (parked).~~ Expired; V3 (Arnav, Oct 4) grants TTI one round. Thermal-property joins stay banned, except `data/v2/emissivity_fsri.csv` for TTI round 2 only (Arnav, Oct 5; `experiments/configs/tti_round2.json`).
- **Committing PDFs** or anything under `data/sources/pdfs/`.
