# bme_fullfit_round1 (2026-10-22): checkpoint

Config `experiments/configs/bme_fullfit_round1.json` (committed d91fe2c before scoring). Code `data/v2/train_bme_fullfit_round1.py`; log `bme_fullfit_round1.log`; results `experiments/results/bme_fullfit_<target>.json` (ef92bbe). Python: `~/dravix/.venv312/bin/python` (uv env lacks pandas).

## Protocol A (what it is)
`pomazi_round1.json`: random 80/20 split, 50 times (seeds 0-49), mean test R2. BME Table 9 used one random split (80/20 per LIT_REVIEW and the round-1 critic, about 15-19 test rows), pooled with 4 external RDP validation composites. The ratio and split type match. The differences: theirs is one draw plus external rows, ours is the mean of 50 draws with no external rows.

## Results (reconstruction; comparable = pending-human)
| Target | BME fit | Dravix fit [CI] (pre-declared model) | BME held-out | Dravix A mean [CI of mean] / 2.5-97.5 split | LOSO |
|---|---|---|---|---|---|
| pHRR | 0.96 | 0.895 [0.841, 0.928] rf: **lower** | 0.30 | 0.604 [0.557, 0.650] / [0.213, 0.832]: higher, mean CI clear | 0.448 |
| TTI | 0.95 | 0.923 [0.882, 0.963] rf: **lower** | 0.58 | 0.615 [0.543, 0.674] / [-0.165, 0.902]: within noise | 0.336 |
| THR | 0.88 | 0.994 [0.991, 0.997] gbr: higher, clear | 0.67 | 0.828 [0.791, 0.861] / [0.575, 0.970]: higher, mean CI clear | 0.679 |
| Char | 0.95 | 0.975 [0.939, 0.999] hgb: higher, not clear | 0.82 | 0.807 [0.766, 0.845] / [0.459, 0.960]: gap is noise (on par) | 0.695 |

Notes:
- The pre-declared rule says "clear" on held-out only if the 2.5th split percentile clears the bar. Under that rule no held-out lead is clear.
- Context only, not used for selection: gbr_theirs full fit is pHRR 0.979 and TTI 0.990. Picking it now would mean selecting on the score, which the rules do not allow.

## Declared but not run
Nothing. Nik's char attempts are not yet declared.

## Next steps
1. Pre-declare and commit `experiments/configs/bme_char_round1.json` with up to 3 independent attempts against the hgb baseline. All use the same 50 seeds; also report LOSO and full fit. The bar for "clearly ahead" is: mean > 0.82, mean-CI lo > 0.82, and at least 40/50 paired wins.
   - (1) Mass-balance floor feature: fibre_wt*1.0 + (1-fibre/100)*neat-matrix char. The neat char is the in-fold mean of neat rows of the same system. Add P-per-FR flags (APP/RDP/TEDAP). There is no ATH/MDH/silica in these rows.
   - (2) Logit target on residue/100, clipped to [0.005, 0.995].
   - (3) A literature prior feature: HGB pretrained on `cone_lit_benchmark_v1.csv` char_residue rows (1687), using heat flux, fr_total, filler, is_EP, fibre_composite and a P-FR flag. Exclude any DOI with a component matching `T58|ERYSIS|GE-60|sorbitol polyglycidyl|pentaerythritol polyglycidyl|glycerol polyglycidyl|GFTE|DETDA`, plus polym11020303, ppch.19468, polym8090322 and ma10050467 (Toldy group).
2. Run the attempts and write results JSONs. Then do the site work (Part 2 of the brief). The likely wording is "higher held-out R2 than BME on 2 of 4 (pHRR, THR), on par on TTI and char". This needs Nik's flag.
