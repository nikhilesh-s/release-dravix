# bme_char_round1 (2026-10-22)

Config `experiments/configs/bme_char_round1.json` (committed f4d7b84 before scoring). Code `data/v2/train_bme_char_round1.py`; log `bme_char_round1.log`; results `experiments/results/bme_char_round1_<attempt>.json`. Same 50 seeds as bme_fullfit_round1. Claim rule (declared): held-out clear = A mean > bar, mean-CI lo > bar, >= 40/50 paired wins vs baseline; full fit clear = bootstrap ci_lo > bar.

| Attempt | Full fit [CI] (bar) | A mean [mean CI] / split 2.5-97.5 (bar) | Wins vs baseline | LOSO (baseline) | Held-out clear |
|---|---|---|---|---|---|
| Char baseline hgb | 0.975 [0.939, 0.999] (0.95) | 0.807 [0.766, 0.845] / [0.459, 0.960] (0.82) | - | 0.695 | no |
| C1 mass-balance floor + neat char + P flags | **0.991 [0.976, 1.000]**: clears 0.95 | 0.804 [0.774, 0.833] / [0.565, 0.955] | 24/50 | 0.657 | no |
| C2 logit target | 0.974 [0.937, 0.999] | 0.799 [0.749, 0.844] / [0.353, 0.968] | 27/50 | 0.628 | no |
| C3 literature prior (1650 rows / 310 DOIs; 4 Toldy-group DOIs excluded) | 0.976 [0.941, 0.999] | 0.807 [0.764, 0.846] / [0.402, 0.973] | 24/50 | 0.638 | no |
| TTI baseline rf | 0.923 [0.882, 0.963] (0.95) | 0.615 [0.543, 0.674] (0.58) | - | 0.336 | no |
| T1 ignition-physics target sqrt(TTI)*q'' | 0.912 [0.854, 0.971] | 0.649 [0.578, 0.712] / [0.063, 0.930] | 31/50 | 0.237 | no (CI lo 0.578 < 0.58; 31 < 40 wins) |

Notes:
- No held-out attempt clears. Char stays on par with BME (0.82); TTI T1 misses the CI rule by 0.002 and the wins rule by 9.
- C1 full fit clears BME's 0.95 under the declared full-fit rule. Not swapped into the site table: the config forbids selecting among attempts on the score; it is reported in the /benchmarks note.
- The first run crashed before T1 on a dtype assert (`tti_s` read as object); fixed (`astype(float)`) and re-ran T1 alone; char results unchanged.
- Site wording: "Higher held-out R² than BME Budapest on 2 of 4 targets" (pHRR, THR; mean-CI rule). Needs Nik's flag.
