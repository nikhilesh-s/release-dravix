# External check: Juelich black/transparent PMMA cone tests vs the pHRR record (report-only)

Config `experiments/configs/external_juelich_pmma.json` (declared at 9b89ae4; amendments 1-2 before any label was read, amendment 3 after). Executor `data/v2/validate_external_juelich_pmma.py`, log `experiments/runs/external_juelich_pmma.log`, outputs `experiments/runs/external_juelich_pmma/`. Data: Zenodo 10.5281/zenodo.6301397 (CC BY 4.0), 24 tests: black and transparent 6 mm cast PMMA, 25/35/50/75 kW/m2, 3 replicates each, auto-ignition. Model: `models/v2/phrr_hgb_v4.joblib` (exp074 recipe, 191-row FSRI fit). No training, tuning or selection on any Juelich row. No number of record is touched. No bar is compared and no win language applies.

## Result as declared (primary): not usable, the label scale is wrong
The file column `HRR (kW/m2)` peaks at 6 to 16. PMMA at these fluxes is several hundred to over a thousand kW/m2 (MaCFP PMMA in this repo: 379-757 at 25 kW/m2, 965-1237 at 50). Scored as published, R2 is about -118,000 and carries no information.

## Post-hoc variant U (exploratory, label-informed): label / 0.0088 m2
The Zenodo description says HRR per unit area used a surface area of 0.0088 m2; the column behaves like total HRR in kW. I proposed dividing by exactly that documented area after seeing both the labels and that the predictions were of similar size, so this is not a confirmatory test. A check that uses no Dravix output: the rescaled Juelich 25 kW/m2 range (696-756) sits inside the MaCFP-lab range (379-757), and the rescaled 25 and 50 kW/m2 means are 1.18x and 1.22x the MaCFP medians.

| Variant (inputs) | n | R2 | MAE kW/m2 | Bias kW/m2 |
|---|---|---|---|---|
| A (6 mm geometry), all | 24 | 0.760 | 138 (12.2% of mean) | -56 |
| A, black (FSRI Black_PMMA) | 12 | 0.852 | 88 | +53 |
| A, transparent (FSRI PMMA, `log_kpc` NaN) | 12 | 0.684 | 187 | -164 |
| B (FSRI's own geometry), all | 24 | 0.868 | 100 | +68 |

## What this does and does not show
- Pooled R2 over 24 tests is driven by 8 (color, flux) cells and by the flux trend, which any monotone-in-flux model reproduces. It is not comparable to a material-grouped CV R2.
- Variant B (FSRI's 8.4 mm and 2.8 mm specimens, which do not match these 6 mm plates) scores better than variant A (6 mm). The two variants disagree on a physically less-correct input, so the 0.76 to 0.87 spread should be read as input sensitivity, not as skill.
- Black 35 kW/m2 is over-predicted by about 30% (pred 1102 vs obs 850) in both variants. The transparent tests are under-predicted at 50 and 75 kW/m2 under A.
- Both PMMAs are in the FSRI training set. This is a new lab, new specimens and a new instrument for a material the model has seen, not an unseen material. Same standing as exp054.
- The transparent predictions carry a missing input (`log_kpc` NaN, amendment 2).

## Needs a human
Confirm the units of the Juelich `HRR (kW/m2)` column with the dataset authors (De Lannoye / Arnold, Forschungszentrum Juelich). Until then, variant U is exploratory and nothing here should be quoted outside this report. A fresh-context critic pass (config `after`) has not been run.
