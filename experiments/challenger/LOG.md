# USTC challenger log (branch `ustc-challenger`, cut from e950b39)

Target: USTC-matched pHRR MAE below 125.5 kW/m2 (Li et al. 2025, PDS 234:111209; abstract-level, dataset differs).
Start point: Arnav's amendment 13, MAE 171.8 [149.4, 207.6], R2 0.58 on B' (EP + P-FR rows with a same-paper, same-flux neat control; 183 rows, 54 papers).

Protocol (unchanged from amendments 10-13): GroupKFold-5 by FR identity, fold map = seed-r permutation of the test population's FR groups, seeds 0-9. Pooled out-of-fold MAE (mean +- sd over the 10 seeds). 95% CI = group bootstrap (2,000 draws) over the seed-0 OOF. R2 = mean over seeds. Each attempt was declared in `attempts.json` and committed before it was scored. Harness: `run_attempt.py`. Logs: `runs/<id>.log`.
Sealed data: `data/v2/cone_lit_parts/oa_staging` (wave 2) was never read or used.

## Dev round (model choice made without scoring B')
The d1-d14 runs were scored on a dev population only: EP rows with a control whose FR has no P (305 rows, disjoint from B'), seeds 0-2.
- More training data was the only large lever. Training on every matrix's non-control rows that have a control: d1 187.2 -> d2 163.3. Adding EP-only rows did little (d11 184.4).
- The neat-control baselines (the same paper's control TTI/THR/avgHRR/t_pHRR/char/TSR/EHC) helped a little: d7 164.9 -> d10 160.0.
- Dropping the 100 per-name `a_` columns helped a little (d14 157.6, chosen).
- No gain from: ECFP on dev (dev FRs mostly lack SMILES), the L1 ratio target (d5, d8), ET alone (d12) or HGB alone (d13).

## Attempts on B'
| id | declaration (short) | groups | MAE | 95% CI | R2 | verdict |
|---|---|---|---|---|---|---|
| A0 | reproduce amendment 13 | first-FR (45) | 171.8 +- 6.5 | [149.4, 207.6] | 0.583 | reproduced exactly |
| A1 | d14 config: all-matrix pool, P load, ctrl baselines, no `a_` | first-FR (45) | 129.2 +- 2.2 | [111.8, 150.8] | 0.769 | superseded (synonym leak in groups) |
| A2 | A1 + ECFP4 of main FR | first-FR (45) | 126.2 +- 2.2 | [106.8, 147.9] | 0.776 | won A1/A2 by the pre-declared rule; superseded (same leak) |
| **A3** | **A2 with strict groups (blind synonym map + InChIKey)** | **strict (37)** | **127.8 +- 3.9** | **[113.5, 149.9]** | **0.774** | **HEADLINE. Run Checker: approve with caveats** |
| A3 fresh | A3 with seeds 10-19 | strict (37) | 128.0 +- 3.7 | [115.4, 148.2] | 0.774 | no selection overfit |
| A0s | Arnav's A0 with strict groups | strict (37) | 170.3 +- 10.2 | [136.5, 197.6] | 0.580 | synonym leak was worth ~1.5 MAE to him |
| A4 | A3 + test-fold papers also held out | strict | 141.7 +- 4.5 | [130.9, 164.1] | 0.726 | diagnostic: ~14 MAE comes from paper-mates |
| A5 | A3 without ctrl baselines | strict | 133.1 +- 4.7 | [115.3, 152.0] | 0.756 | ablation: ctrl baselines worth ~5 |
| A6 | A3 config on population v2 (bulk, organophosphorus, n=168) | strict (65) | 155.2 +- 3.5 | [129.9, 170.8] | 0.512 | closest scope to USTC; well above 125.5 |
| RC1 | A3 + drop training rows sharing ANY FR/synergist identity with the test fold | strict | 134.1 +- 4.3 | [120.5, 161.5] | 0.747 | Run Checker: ~6 MAE from non-first-FR overlap |

Baseline (control x training-median ratio): ~195 for every attempt.

## Outcome
- Not below 125.5. The best fair point is 127.8 (A3). Its CI [113.5, 149.9] overlaps 125.5, so this is "comparable", not "beats".
- The one point under 130 on the old split (A2, 126.2) uses Arnav's synonym-leaky groups, so it is not quoted.
- Budget: 8 attempts used (A1-A6, A0s, RC1). The loop stops here.
- Required footnotes: see `RUN_CHECK.md`.
