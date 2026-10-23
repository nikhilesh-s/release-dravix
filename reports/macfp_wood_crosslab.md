# A2: MaCFP-4 pine and Edinburgh PMMA, external cross-lab check (report-only)

<!-- summary:start -->
## Summary (plain English, 5 lines)

1. External cross-lab check, MaCFP-4 pine (one material, 8 oxygen-consumption labs), report-only. The frozen FSRI records exp074 (pHRR), exp046 (TTI) and exp117 (char residue, diagnostic) ran unchanged on 80 pine cone tests. UMET's 6 mass-loss-cone tests and 12 Edinburgh PMMA ignition times were kept separate. All three records reproduced on FSRI first, every count and sanity gate matched the config, and all 5 PS MCC labs passed the +-10% gate.
2. pHRR: R2 0.2156 on all 80 (lab-bootstrap 95% CI [-1.8485, 0.6019]), 0.6375 on FSRI's own 18 tests, and 0.0496 [-2.6566, 0.6209] on the other 7 labs' 62. The model over-predicts by 39.1 kW/m2 on average (MAE 44.3): most at FPL (+83.1), IMT (+72.7) and UDRI (+53.3), least at FZJ (+8.3). Its predictions move in three steps: about 193-201 kW/m2 at 25-30, 276-289 at 45-60 and 349-366 at 75.
3. TTI: R2 in seconds 0.5242 on all 80 ([-0.5670, 0.7851]; log R2 0.7283), 0.8513 on FSRI-lab 18 and 0.4644 [-1.9213, 0.4972] on the other 62. At 25 kW/m2 (FSRI 6 + IMT 5) it is -0.3593, with predictions 43.7 s too early on average. On Edinburgh PMMA (known material (probable), new lab), R2 is 0.9424 from only two flux levels, with a -7.0 s bias.
4. Char residue (diagnostic, absolute error only): against residue at flameout, exp117's MAE is 5.8 percentage points (bias +1.6) on the declared 50 tests, and 4.4 (bias +3.5) without UQ's flagged mass channel. It predicts a flux trend the MaCFP data do not show: at FSRI-lab, 26.8% predicted vs 17.7% observed at 25 kW/m2, and 16.2% vs 16.9% at 75.
5. End-of-test residue ranges from 0.8% (FZJ, whose tests ran 52 min to 3 h) to 44.4% (UQ). For a charring material, the FSRI end-of-test CR definition therefore depends on when a lab stops the test. This is a warning only. Nothing here is compared with any bar, no number of record changes, and a fresh-context critic pass is required before any number is quoted.
<!-- summary:end -->

Quote as: external cross-lab check, MaCFP-4 pine (one material, 8 oxygen-consumption labs), report-only. Nothing here is a candidate, nothing is compared with any bar, and no number of record changes. Config: `experiments/configs/macfp_wood_crosslab.json` (committed dd9c9c6 before this run). Script: `data/v2/validate_macfp_wood_crosslab.py`. Log: `experiments/runs/macfp_wood_crosslab.log`. Data files: `experiments/runs/macfp_wood_crosslab/` (ps_check.json, features.json, tests.csv, summary.json); the config names experiments/results/macfp_wood_crosslab/ and experiments/runs/validate_macfp_wood_crosslab.log, but this run's orchestrator forbade writing into experiments/results/, so the same files went to experiments/runs/ (contents as declared; not ingested by the tracker either way). MaCFP matl-db pinned at 0e4ad8b (MIT; DOI 10.18434/mds2-2586; per-lab READMEs; TUBS doi 10.24355/dbbs.084-202604271326-0).

Status: **ok**

## How to read this

- One material. R2 mostly measures the flux response plus lab bias; the per-lab and per-flux tables are the real result. Quote R2 only with them.
- Lab TTI definitions and time zeros are not harmonised (Aalto: from start of DAQ; FSRI: operator button). Sampling is 5 s at Aalto and IMT and 1 s elsewhere, which can lower the raw pHRR max there. Specimens were conditioned at about 23 C / 50% RH (UQ measured 10.2% moisture); some FZJ 60 kW/m2 specimens were desiccated.
- UQ is kept everywhere and flagged (README: 60 kW/m2 mass readings fluctuated).
- The models are the records exactly as fitted on FSRI: exp074 (`models/v2/phrr_hgb_v4.joblib`), exp046 (Poisson GLM on 5 physics features, refit on all 681 FSRI rows) and exp117 (Avg of HGB, ExtraTrees and Ridge, refit on all 567 FSRI rows). No MaCFP row was used to fit anything.
- The PMMA cross-lab 0.588 (exp054) used exp043's model, not exp074; it is a different material and model and is never pooled with anything here.

## Exclusions applied (all a priori, from the config)

- BUWfd_Wood_Cone_50kW_hor_R1: non-physical mass (MaCFP's own script excludes it). The other 5 BUWfd tests have no cone metadata and appear only in the end-of-test residue table.
- UCantabria: 4 README-only rows with no CSV and no specimen mass; excluded everywhere.
- UMET (ISO 13927 mass-loss cone, thermopile HRR): a separate stratum, never pooled with the oxygen-consumption labs.
- Aalto `*_Gasification_*` and TUBS `*_Gasification_*` files are N2 tests and are never used.
- MaCFP's outcome-based averaging exclusion (IMT 25 kW/m2 R2/R5, chosen by ignition time) is NOT applied.
- Char residue at flameout: a test enters only if it has a README t_flameout inside its finite mass record (53 tests). FZJ and TUBS report no flameout; FPL R2 lost its DAQ at 1400 s; 4 Aalto and 3 UMET tests fall outside.
- FZJ 60 kW/m2 perpendicular R3 and 75 kW/m2 parallel R1 have all-NaN mass: they are in the pHRR/TTI pools but not in the residue tables.
- Edinburgh PMMA (12 README rows) is a TTI-only stratum. It probably overlaps FSRI Black_PMMA, which is in exp046's training rows: known material (probable), new lab.

## Step 1: records reproduced on FSRI before any MaCFP prediction

| record | rows/groups | GKF 5x10 mean | CI | record value | reproduced (within 0.002) |
|---|---|---|---|---|---|
| exp074 | 191/64 | 0.6113 | [0.5155, 0.8376] | 0.6113 [0.5155, 0.8376] | True |
| exp046 | 681/76 | 0.3133 | [0.0715, 0.4958] | 0.3133 [0.0715, 0.4958] | True |
| exp117 | 567/61 | 0.6504 | [0.4389, 0.7893] | 0.6504 [0.4389, 0.7893] | True |

exp074 refit vs saved joblib on the 191 FSRI rows: max |diff| 0 kW/m2 (limit 1e-6).

## Step 2: PS MCC gate (label-free)

Reference = median of the 5 lab medians: HRC 1119.8 J/gK, h_c 40.58 kJ/g. Pass: 0.90 <= lab/reference <= 1.10 for both. No PS-derived factor is applied to any input. HRC and h_c are treated as rate-independent instrument checks (FZJ's 60 K/min PS runs vouch for its 30 K/min wood runs).

| lab | runs | HRC J/gK | HRC ratio | h_c kJ/g | h_c ratio | gate |
|---|---|---|---|---|---|---|
| FZJ | 6 | 1175.8 | 1.050 | 40.00 | 0.986 | PASS |
| Lund | 1 | 1059.8 | 0.946 | 42.75 | 1.053 | PASS |
| NIST | 1 | 1156.9 | 1.033 | 40.45 | 0.997 | PASS |
| TUBS | 1 | 1109.1 | 0.990 | 40.58 | 1.000 | PASS |
| UDRI | 3 | 1119.8 | 1.000 | 40.65 | 1.002 | PASS |

Context only, never part of the gate: TUBS self-report 1099 J/gK and 38.8 kJ/g (long-run 1108 +- 15 J/gK, 38.7 +- 0.2 kJ/g, 30 K/min); UDRI self-report 1099-1166 W/g and 41.2-41.8 kJ/g (60 K/min).

## Step 3: frozen inputs (label-free)

Material-level: MCC peak 66.58 W/g (FZJ raw 30K R1-R2 + TUBS 30K R1-R3), T_peak 378.80 C, total HR (bc) 11.973 kJ/g, onset (bc) 243.95 C (mean = median with 2 runs: 243.95), MCC char yield 14.73% (FZJ README masses), STA residues 3/10/30 K/min 15.90 / 19.10 / 21.89% (FSRI MaCFP STA; at 750 C, report only: 16.15 / 19.34 / 22.14%), log_kpc primary NaN, S1 10.9505 (TPS). G2 medians: L 25.60 mm, density 378.7 kg/m3, mass/area 10.043 kg/m2. CR flags primary {structural, exterior finish}; S3 all 0. features.json sha256 `8f2831f2a009eeb3466f43f00836910419dec66f264f6d8cbee7694117956b3e`.

Sanity gates (all must pass before any prediction):

| gate | value | range | ok |
|---|---|---|---|
| mcc_peak_hrr_Wg | 66.580 | [40, 120] | True |
| mcc_total_hr_kJg | 11.973 | [8, 16] | True |
| mcc_t_onset_C (mean) | 243.950 | [200, 400] | True |
| mcc_t_onset_C (median) | 243.950 | [200, 400] | True |
| mcc_t_peak_C | 378.800 | [330, 420] | True |
| mcc_char_yield_pct | 14.733 | [5, 30] | True |
| sta_res_3K | 15.900 | [5, 40] | True |
| sta_res_10K | 19.105 | [5, 40] | True |
| sta_res_30K | 21.894 | [5, 40] | True |
| per-test density_kgm3 min (n=86) | 314.431 | [250, 600] | True |
| per-test density_kgm3 max (n=86) | 495.137 | [250, 600] | True |
| per-test cone_thickness_mm min (n=86) | 25.000 | [20, 30] | True |
| per-test cone_thickness_mm max (n=86) | 26.500 | [20, 30] | True |
| per-test cone_mass_per_area_kgm2 min (n=86) | 8.270 | [6, 16] | True |
| per-test cone_mass_per_area_kgm2 max (n=86) | 13.552 | [6, 16] | True |

MaCFP pine inputs next to the FSRI training rows of the two solid-softwood products (label-free; MaCFP geometry shown as G2 medians):

| feature | MaCFP pine | Pine_Siding | Wood_Stud |
|---|---|---|---|
| mcc_peak_hrr_Wg | 66.580 | 78.093 | 73.979 |
| mcc_char_yield_pct | 14.733 | 10.450 | 12.887 |
| mcc_total_hr_kJg | 11.973 | 12.758 | 11.219 |
| mcc_t_peak_C | 378.800 | 376.533 | 370.667 |
| mcc_t_onset_C | 243.950 | 239.833 | 261.367 |
| log_kpc | NaN (S1 10.951) | 10.483 | 10.510 |
| cone_thickness_mm | 25.600 | 18.020 | 38.060 |
| density_kgm3 | 378.681 | 310.000 | 355.000 |
| cone_mass_per_area_kgm2 | 10.043 | 6.771 | 17.507 |
| log_dT (exp046) | 5.411 | 5.355 | 5.513 |
| log_L (exp046) | -3.665 | -4.016 | -3.269 |
| log_rho (exp046) | 5.937 | 5.737 | 5.872 |
| sta_res_3K (exp117) | 15.900 | 17.607 | 13.815 |
| sta_res_10K (exp117) | 19.105 | 20.500 | 18.147 |
| sta_res_30K (exp117) | 21.894 | 20.445 | 17.646 |
| cat_structural (exp117) | 1.000 | 1.000 | 1.000 |
| cat_exterior_finish (exp117) | 1.000 | 1.000 | 1.000 |
| cat_interior_finish (exp117) | 0.000 | 0.000 | 1.000 |

Every Wood MCC file in the pin (label-free; only FZJ raw 30K R1-R2 and TUBS 30K R1-R3 peak are inputs):

| lab | file | kind | atmosphere | rate_K_min | purpose | peak_Wg | hrc_JgK | t_peak_C | total_hr_bc_kJg | t_onset_bc_C | rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FZJ | FZJ_Wood_MCC-raw_N2_30K_R1.csv | raw | N2 | 30 | main measurement | 73.614 | 147.23 | 379.8 | 11.889 | 249.4 | 2127 |
| FZJ | FZJ_Wood_MCC-raw_N2_30K_R2.csv | raw | N2 | 30 | main measurement | 70.936 | 141.87 | 377.8 | 12.057 | 238.5 | 2127 |
| FZJ | FZJ_Wood_MCC-raw_N2_45K_R1.csv | raw | N2 | 45 | main measurement | 112.197 | 149.6 | 387.3 | 12.334 | 257.1 | 1426 |
| FZJ | FZJ_Wood_MCC-raw_N2_45K_R2.csv | raw | N2 | 45 | main measurement | 112.385 | 149.85 | 385.2 | 12.205 | 266.7 | 1426 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R1.csv | raw | N2 | 60 | mass down-scaling | 138.72 | 138.72 | 395.3 | 6.675 | 341.9 | 1077 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R10.csv | raw | N2 | 60 | main measurement | 141.811 | 141.81 | 392.9 | 12.393 | 261.4 | 1177 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R2.csv | raw | N2 | 60 | mass down-scaling | 164.37 | 164.37 | 396.4 | 13.807 | 304.4 | 1052 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R3.csv | raw | N2 | 60 | mass down-scaling | 153.2 | 153.2 | 393.3 | 10.363 | 317.8 | 1077 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R4.csv | raw | N2 | 60 | mass down-scaling | 153.761 | 153.76 | 397.4 | 9.651 | 327.9 | 1078 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R5.csv | raw | N2 | 60 | mass down-scaling | 156.89 | 156.89 | 393.3 | 12.508 | 304.3 | 1078 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R6.csv | raw | N2 | 60 | mass down-scaling | 143.556 | 143.56 | 392.1 | 12.559 | 255.5 | 1277 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R7.csv | raw | N2 | 60 | mass down-scaling | 142.911 | 142.91 | 389.4 | 12.695 | 260.3 | 1277 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R8.csv | raw | N2 | 60 | main measurement | 146.945 | 146.94 | 391.6 | 12.4 | 261.0 | 1277 |
| FZJ | FZJ_Wood_MCC-raw_N2_60K_R9.csv | raw | N2 | 60 | main measurement | 143.809 | 143.81 | 392.4 | 12.477 | 267.2 | 1277 |
| FZJ | FZJ_Wood_MCC_N2_30K_R1.csv | processed | N2 | 30 | main measurement | 74.89 | 149.78 | 379.8 | 12.01 | 246.3 | 1055 |
| FZJ | FZJ_Wood_MCC_N2_30K_R2.csv | processed | N2 | 30 | main measurement | 71.97 | 143.94 | 378.1 | 12.215 | 233.1 | 1054 |
| FZJ | FZJ_Wood_MCC_N2_45K_R1.csv | processed | N2 | 45 | main measurement | 113.16 | 150.88 | 387.3 | 12.514 | 251.8 | 1055 |
| FZJ | FZJ_Wood_MCC_N2_45K_R2.csv | processed | N2 | 45 | main measurement | 113.14 | 150.85 | 385.2 | 12.409 | 260.7 | 1056 |
| FZJ | FZJ_Wood_MCC_N2_60K_R1.csv | processed | N2 | 60 | mass down-scaling | 144.87 | 144.87 | 395.4 | 7.426 | 333.9 | 1058 |
| FZJ | FZJ_Wood_MCC_N2_60K_R10.csv | processed | N2 | 60 | main measurement | 141.29 | 141.29 | 392.8 | 12.678 | 252.8 | 1158 |
| FZJ | FZJ_Wood_MCC_N2_60K_R2.csv | processed | N2 | 60 | mass down-scaling | 155.77 | 155.77 | 396.8 | 14.293 | 292.8 | 1031 |
| FZJ | FZJ_Wood_MCC_N2_60K_R3.csv | processed | N2 | 60 | mass down-scaling | 154.87 | 154.87 | 393.4 | 10.698 | 312.4 | 1058 |
| FZJ | FZJ_Wood_MCC_N2_60K_R4.csv | processed | N2 | 60 | mass down-scaling | 155.83 | 155.83 | 397.5 | 10.675 | 312.0 | 1058 |
| FZJ | FZJ_Wood_MCC_N2_60K_R5.csv | processed | N2 | 60 | mass down-scaling | 154.02 | 154.02 | 393.3 | 12.828 | 296.8 | 1058 |
| FZJ | FZJ_Wood_MCC_N2_60K_R6.csv | processed | N2 | 60 | mass down-scaling | 143.5 | 143.5 | 391.8 | 12.777 | 249.3 | 1258 |
| FZJ | FZJ_Wood_MCC_N2_60K_R7.csv | processed | N2 | 60 | mass down-scaling | 142.7 | 142.7 | 389.6 | 13.039 | 249.1 | 1258 |
| FZJ | FZJ_Wood_MCC_N2_60K_R8.csv | processed | N2 | 60 | main measurement | 146.82 | 146.82 | 391.8 | 12.705 | 252.8 | 1258 |
| FZJ | FZJ_Wood_MCC_N2_60K_R9.csv | processed | N2 | 60 | main measurement | 142.99 | 142.99 | 392.4 | 12.834 | 257.9 | 1258 |
| IMT | IMT_Wood_MCC-raw_N2_60K_R1.csv | raw | N2 | 60 |  | 129.151 | 129.15 | 382.6 | 15.044 | 193.4 | 1376 |
| IMT | IMT_Wood_MCC-raw_N2_60K_R2.csv | raw | N2 | 60 |  | 126.593 | 126.59 | 389.4 | 15.538 | 184.1 | 1383 |
| IMT | IMT_Wood_MCC_N2_60K_R1.csv | processed | N2 | 60 |  | 121.939 | 121.94 | 382.6 | 13.204 | 224.9 | 1216 |
| IMT | IMT_Wood_MCC_N2_60K_R2.csv | processed | N2 | 60 |  | 114.266 | 114.27 | 389.4 | 12.814 | 227.1 | 1173 |
| IMT | IMT_Wood_MCC_O2-10_60K_R1.csv | processed | O2-10 | 60 |  | 139.35 | 139.35 | 353.0 | 17.131 | 242.3 | 1141 |
| IMT | IMT_Wood_MCC_O2-10_60K_R2.csv | processed | O2-10 | 60 |  | 139.91 | 139.91 | 354.1 | 18.982 | 222.3 | 1205 |
| IMT | IMT_Wood_MCC_O2-20_60K_R1.csv | processed | O2-20 | 60 |  | 168.684 | 168.68 | 349.2 | 19.246 | 227.1 | 1150 |
| IMT | IMT_Wood_MCC_O2-20_60K_R2.csv | processed | O2-20 | 60 |  | 236.938 | 236.94 | 467.0 | 20.107 | 237.7 | 1171 |
| IMT | IMT_Wood_MCC_O2-2_60K_R1.csv | processed | O2-2 | 60 |  | 127.028 | 127.03 | 369.8 | 20.128 | 221.2 | 1213 |
| IMT | IMT_Wood_MCC_O2-2_60K_R2.csv | processed | O2-2 | 60 |  | 139.305 | 139.31 | 365.0 | 22.873 | 219.1 | 1219 |
| IMT | IMT_Wood_MCC_O2-5_60K_R1.csv | processed | O2-5 | 60 |  | 128.382 | 128.38 | 364.3 | 19.112 | 218.0 | 1250 |
| IMT | IMT_Wood_MCC_O2-5_60K_R2.csv | processed | O2-5 | 60 |  | 136.077 | 136.08 | 359.5 | 18.646 | 244.7 | 1148 |
| Lund | Lund_Wood-char_MCC_O2-20_60K_R1.csv | processed | O2-20 | 60 |  | 385.468 | 385.47 | 371.4 | 33.222 | 295.9 | 720 |
| Lund | Lund_Wood-char_MCC_O2-20_60K_R2.csv | processed | O2-20 | 60 |  | 476.671 | 476.67 | 399.4 | 36.376 | 336.4 | 885 |
| Lund | Lund_Wood-char_MCC_O2-20_60K_R3.csv | processed | O2-20 | 60 |  | 368.552 | 368.55 | 396.4 | 36.242 | 331.9 | 856 |
| Lund | Lund_Wood_MCC-raw_N2_60K_R1.csv | raw | N2 | 60 |  | 137.005 | 137.0 | 375.2 | 12.118 | 265.8 | 994 |
| Lund | Lund_Wood_MCC-raw_N2_60K_R2.csv | raw | N2 | 60 |  | 139.457 | 139.46 | 373.2 | 12.54 | 257.0 | 1197 |
| Lund | Lund_Wood_MCC-raw_N2_60K_R3.csv | raw | N2 | 60 |  | 137.619 | 137.62 | 374.3 | 12.462 | 263.6 | 1176 |
| Lund | Lund_Wood_MCC_N2_60K_R1.csv | processed | N2 | 60 |  | 133.051 | 133.05 | 375.4 | 11.99 | 265.9 | 764 |
| Lund | Lund_Wood_MCC_N2_60K_R2.csv | processed | N2 | 60 |  | 137.392 | 137.39 | 372.9 | 12.22 | 260.9 | 940 |
| Lund | Lund_Wood_MCC_N2_60K_R3.csv | processed | N2 | 60 |  | 136.072 | 136.07 | 374.4 | 12.382 | 262.9 | 1022 |
| Lund | Lund_Wood_MCC_O2-20_60K_R1.csv | processed | O2-20 | 60 |  | 286.785 | 286.79 | 330.4 | 18.798 | 265.9 | 696 |
| Lund | Lund_Wood_MCC_O2-20_60K_R2.csv | processed | O2-20 | 60 |  | 289.576 | 289.58 | 331.4 | 19.025 | 263.9 | 756 |
| Lund | Lund_Wood_MCC_O2-20_60K_R3.csv | processed | O2-20 | 60 |  | 287.527 | 287.53 | 330.9 | 19.211 | 263.4 | 759 |
| NIST | NIST_Wood-char_MCC_O2-20_60K_R1.csv | processed | O2-20 | 60 |  | 428.882 | 428.88 | 397.9 | 34.938 | 359.4 | 1612 |
| NIST | NIST_Wood-char_MCC_O2-20_60K_R2.csv | processed | O2-20 | 60 |  | 469.571 | 469.57 | 393.4 | 34.052 | 367.4 | 997 |
| NIST | NIST_Wood-char_MCC_O2-20_60K_R3.csv | processed | O2-20 | 60 |  | 402.554 | 402.55 | 406.4 | 34.14 | 368.9 | 1020 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R1.csv | raw | N2 | 60 |  | 142.008 | 142.01 | 395.4 | 13.361 | 259.6 | 1978 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R10.csv | raw | N2 | 60 |  | 147.849 | 147.85 | 394.2 | 12.26 | 271.9 | 1278 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R11.csv | raw | N2 | 60 |  | 148.457 | 148.46 | 393.5 | 12.296 | 272.1 | 1279 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R12.csv | raw | N2 | 60 |  | 150.835 | 150.84 | 394.2 | 12.339 | 273.8 | 1279 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R2.csv | raw | N2 | 60 |  | 143.826 | 143.83 | 395.8 | 12.247 | 270.0 | 1279 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R3.csv | raw | N2 | 60 |  | 140.747 | 140.75 | 394.6 | 12.35 | 266.3 | 1278 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R4.csv | raw | N2 | 60 |  | 144.401 | 144.4 | 395.8 | 12.217 | 271.0 | 1278 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R5.csv | raw | N2 | 60 |  | 141.291 | 141.29 | 394.4 | 12.367 | 265.1 | 1278 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R6.csv | raw | N2 | 60 |  | 140.646 | 140.65 | 394.4 | 12.068 | 266.0 | 1279 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R7.csv | raw | N2 | 60 |  | 139.151 | 139.15 | 394.4 | 12.091 | 265.0 | 1278 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R8.csv | raw | N2 | 60 |  | 139.677 | 139.68 | 394.1 | 12.096 | 265.2 | 1277 |
| NIST | NIST_Wood_MCC-raw_N2_60K_R9.csv | raw | N2 | 60 |  | 141.767 | 141.77 | 395.2 | 12.221 | 269.4 | 1278 |
| NIST | NIST_Wood_MCC_N2_60K_R1.csv | processed | N2 | 60 |  | 139.682 | 139.68 | 395.4 | 12.213 | 258.9 | 1674 |
| NIST | NIST_Wood_MCC_N2_60K_R10.csv | processed | N2 | 60 |  | 148.043 | 148.04 | 394.4 | 12.744 | 246.4 | 1090 |
| NIST | NIST_Wood_MCC_N2_60K_R11.csv | processed | N2 | 60 |  | 145.419 | 145.42 | 393.9 | 12.507 | 251.4 | 1089 |
| NIST | NIST_Wood_MCC_N2_60K_R12.csv | processed | N2 | 60 |  | 148.278 | 148.28 | 394.4 | 12.716 | 250.4 | 1087 |
| NIST | NIST_Wood_MCC_N2_60K_R2.csv | processed | N2 | 60 |  | 139.9 | 139.9 | 395.4 | 11.938 | 263.4 | 1053 |
| NIST | NIST_Wood_MCC_N2_60K_R3.csv | processed | N2 | 60 |  | 137.952 | 137.95 | 394.9 | 11.919 | 261.4 | 1056 |
| NIST | NIST_Wood_MCC_N2_60K_R4.csv | processed | N2 | 60 |  | 141.317 | 141.32 | 395.9 | 11.946 | 263.4 | 1054 |
| NIST | NIST_Wood_MCC_N2_60K_R5.csv | processed | N2 | 60 |  | 137.916 | 137.92 | 394.4 | 11.928 | 260.9 | 1055 |
| NIST | NIST_Wood_MCC_N2_60K_R6.csv | processed | N2 | 60 |  | 137.619 | 137.62 | 394.4 | 11.811 | 258.9 | 1055 |
| NIST | NIST_Wood_MCC_N2_60K_R7.csv | processed | N2 | 60 |  | 135.364 | 135.36 | 394.4 | 11.464 | 263.9 | 1057 |
| NIST | NIST_Wood_MCC_N2_60K_R8.csv | processed | N2 | 60 |  | 136.175 | 136.18 | 394.4 | 11.55 | 263.4 | 1057 |
| NIST | NIST_Wood_MCC_N2_60K_R9.csv | processed | N2 | 60 |  | 138.991 | 138.99 | 395.4 | 12.096 | 260.9 | 1053 |
| NIST | NIST_Wood_MCC_O2-20_60K_R1.csv | processed | O2-20 | 60 |  | 206.423 | 206.42 | 356.4 | 18.028 | 269.9 | 1294 |
| NIST | NIST_Wood_MCC_O2-20_60K_R2.csv | processed | O2-20 | 60 |  | 205.508 | 205.51 | 355.4 | 18.349 | 267.9 | 1282 |
| NIST | NIST_Wood_MCC_O2-20_60K_R3.csv | processed | O2-20 | 60 |  | 207.994 | 207.99 | 354.9 | 18.048 | 268.9 | 1288 |
| TUBS | TUBS_Wood_MCC_N2_30K_R1.csv | processed | N2 | 30 |  | 59.52 | 119.04 | 379.4 | 7.635 | 279.4 | 797 |
| TUBS | TUBS_Wood_MCC_N2_30K_R2.csv | processed | N2 | 30 |  | 62.92 | 125.84 | 377.9 | 10.85 | 266.4 | 799 |
| TUBS | TUBS_Wood_MCC_N2_30K_R3.csv | processed | N2 | 30 |  | 65.91 | 131.82 | 376.9 | 10.301 | 274.4 | 797 |
| UDRI | UDRI_Wood_MCC_N2_60K_R1.csv | processed | N2 | 60 |  | 125.41 | 125.41 | 394.4 | 11.263 | 244.4 | 1148 |
| UDRI | UDRI_Wood_MCC_N2_60K_R2.csv | processed | N2 | 60 |  | 126.23 | 126.23 | 394.9 | 11.205 | 243.9 | 1184 |
| UDRI | UDRI_Wood_MCC_N2_60K_R3.csv | processed | N2 | 60 |  | 128.31 | 128.31 | 395.4 | 11.101 | 244.9 | 1170 |

## pHRR (exp074), primary

| pool | n | R2 (kW/m2) | lab-bootstrap 95% CI | MAE | bias (pred-obs) |
|---|---|---|---|---|---|
| primary_80 | 80 | 0.2156 | [-1.8485, 0.6019] | 44.3 | 39.1 |
| FSRI-lab_18 | 18 | 0.6375 | n/a (single lab) | 34.9 | 33.2 |
| other-labs_62 | 62 | 0.0496 | [-2.6566, 0.6209] | 47.1 | 40.8 |

Per lab:

| lab | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) |
|---|---|---|---|---|---|---|---|
| Aalto | 9 | 206.6 | 256.9 | 50.3 | 50.3 | 0.80 | 5 |
| FPL | 6 | 205.0 | 288.1 | 83.1 | 83.1 | 0.71 | 1 |
| FSRI | 18 | 242.4 | 275.6 | 34.9 | 33.2 | 0.88 | 1 |
| FZJ | 22 | 270.4 | 278.7 | 23.1 | 8.3 | 0.97 | 1 |
| IMT | 12 | 197.1 | 269.8 | 74.9 | 72.7 | 0.73 | 5 |
| TUBS | 3 | 158.3 | 193.2 | 35.0 | 35.0 | 0.82 | 1 |
| UDRI | 7 | 197.5 | 250.8 | 58.3 | 53.3 | 0.79 | 1 |
| UQ (flag: README mass fluctuations) | 3 | 261.4 | 276.0 | 15.8 | 14.6 | 0.95 | 1 |

Per nominal flux:

| flux | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) |
|---|---|---|---|---|---|---|---|
| 25 | 11 | 162.9 | 196.6 | 36.1 | 33.7 | 0.83 | 1/5 |
| 30 | 15 | 172.5 | 196.8 | 32.8 | 24.3 | 0.88 | 1/5 |
| 45 | 3 | 205.9 | 288.4 | 82.4 | 82.4 | 0.71 | 5 |
| 50 | 16 | 217.0 | 284.3 | 68.8 | 67.3 | 0.76 | 1/5 |
| 60 (incl. UQ, flagged) | 19 | 261.2 | 281.6 | 30.8 | 20.4 | 0.93 | 1/5 |
| 75 | 16 | 311.0 | 353.4 | 45.3 | 42.4 | 0.88 | 1/5 |

Per orientation:

| orientation | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) |
|---|---|---|---|---|---|---|---|
| parallel (incl. UQ, flagged) | 34 | 230.3 | 262.7 | 38.2 | 32.4 | 0.88 | 1/5 |
| perpendicular | 21 | 266.7 | 280.0 | 21.2 | 13.3 | 0.95 | 1 |
| unknown | 25 | 199.1 | 268.9 | 72.2 | 69.8 | 0.74 | 1/5 |

Same-flux perpendicular minus parallel (kW/m2):

| lab | flux | n perp/par | obs perp-par | pred perp-par |
|---|---|---|---|---|
| FSRI | 25 | 3/3 | -9.8 | 0.2 |
| FSRI | 50 | 3/3 | 2.7 | 4.1 |
| FSRI | 75 | 3/3 | 45.6 | 4.2 |
| FZJ | 30 | 3/3 | -33.1 | 4.9 |
| FZJ | 60 | 5/4 | 22.4 | 0.1 |
| FZJ | 75 | 4/3 | 49.3 | 0.0 |

QA: time of the pHRR max minus reported t_ign (s), per lab:

| lab | t(pHRR)-t_ign min s | median s | max s |
|---|---|---|---|
| Aalto | 13.0 | 19.0 | 32.0 |
| FPL | 8.0 | 9.5 | 15.0 |
| FSRI | 21.2 | 28.8 | 37.0 |
| FZJ | 4.0 | 10.0 | 20.0 |
| IMT | 5.0 | 13.0 | 17.0 |
| TUBS | 16.0 | 16.0 | 17.0 |
| UDRI | 12.2 | 23.2 | 26.5 |
| UQ | 5.0 | 8.0 | 11.0 |

UMET stratum (mass-loss cone, never pooled): n 6, MAE 97.9, bias +97.9 kW/m2.

| file | flux | obs_phrr_kWm2 | pred_phrr_primary | pred-obs |
|---|---|---|---|---|
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R1.csv | 30 | 119.0 | 200.5 | 81.5 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R2.csv | 30 | 113.4 | 193.0 | 79.6 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R3.csv | 30 | 144.6 | 200.8 | 56.2 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R1.csv | 60 | 137.8 | 275.8 | 138.0 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R2.csv | 60 | 173.0 | 288.1 | 115.1 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R3.csv | 60 | 158.5 | 275.8 | 117.3 |

Secondary variants (each changes one thing; never primary):

| variant | pool | n | R2 | CI | MAE | bias |
|---|---|---|---|---|---|---|
| S1 | primary_80 | 80 | -1.0926 | [-5.5146, -0.3017] | 81.4 | 80.8 |
| S1 | FSRI-lab_18 | 18 | -0.4036 | n/a | 76.0 | 76.0 |
| S1 | other-labs_62 | 62 | -1.3767 | [-7.2403, -0.2262] | 83.0 | 82.2 |
| S2 | primary_80 | 80 | 0.3583 | [-1.1801, 0.6383] | 40.0 | 34.3 |
| S2 | FSRI-lab_18 | 18 | 0.6564 | n/a | 33.2 | 31.5 |
| S2 | other-labs_62 | 62 | 0.2396 | [-1.8002, 0.6595] | 42.0 | 35.1 |
| S1 | UMET 6 | 6 |  |  | 139.8 | 139.8 |
| S2 | UMET 6 | 6 |  |  | 93.6 | 93.6 |

S1: log_kpc = TPS value; S2: G2 (median) geometry for every row.

## TTI (exp046), primary

| pool | n | R2 (s) | lab-bootstrap 95% CI | log R2 | MAE s | bias s |
|---|---|---|---|---|---|---|
| primary_80 | 80 | 0.5242 | [-0.5670, 0.7851] | 0.7283 | 15.1 | -0.9 |
| FSRI-lab_18 | 18 | 0.8513 | n/a | 0.8508 | 9.8 | -2.5 |
| other-labs_62 | 62 | 0.4644 | [-1.9213, 0.4972] | 0.6774 | 16.7 | -0.4 |
| tti_25 | 11 | -0.3593 | n/a | -0.3529 | 52.5 | -43.7 |

Per lab:

| lab | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) | log R2 | median abs(pred-obs)/obs (%) |
|---|---|---|---|---|---|---|---|---|---|
| Aalto | 9 | 26.6 | 33.4 | 10.5 | 6.8 | 0.80 | 5 | 0.5490 | 31.7 |
| FPL | 6 | 12.2 | 17.6 | 5.4 | 5.4 | 0.69 | 1 | -8.5893 | 46.9 |
| FSRI | 18 | 42.1 | 39.6 | 9.8 | -2.5 | 1.06 | 1 | 0.8508 | 28.9 |
| FZJ | 22 | 18.7 | 23.6 | 7.8 | 4.9 | 0.79 | 1 | 0.6731 | 28.0 |
| IMT | 12 | 83.1 | 46.7 | 42.4 | -36.4 | 1.78 | 5 | 0.8240 | 56.5 |
| TUBS | 3 | 26.0 | 55.9 | 29.9 | 29.9 | 0.46 | 1 | -594.5809 | 115.6 |
| UDRI | 7 | 17.0 | 35.4 | 18.4 | 18.4 | 0.48 | 1 | -0.1244 | 168.2 |
| UQ (flag: README mass fluctuations) | 3 | 13.0 | 10.4 | 2.8 | -2.6 | 1.25 | 1 | -1.3852 | 24.3 |

Per nominal flux:

| flux | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) | log R2 | median abs(pred-obs)/obs (%) |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 11 | 138.1 | 94.5 | 52.5 | -43.7 | 1.46 | 1/5 | -0.3529 | 16.9 |
| 30 | 15 | 37.4 | 62.7 | 25.3 | 25.3 | 0.60 | 1/5 | -2.0831 | 70.1 |
| 45 | 3 | 21.7 | 24.2 | 9.7 | 2.6 | 0.89 | 5 | -0.3762 | 37.3 |
| 50 | 16 | 15.7 | 17.6 | 6.8 | 1.9 | 0.89 | 1/5 | -0.2879 | 43.7 |
| 60 (incl. UQ, flagged) | 19 | 10.8 | 11.1 | 3.4 | 0.3 | 0.97 | 1/5 | -0.2702 | 24.3 |
| 75 | 16 | 7.1 | 6.3 | 3.2 | -0.7 | 1.12 | 1/5 | -0.0196 | 35.8 |

Per orientation:

| orientation | n | mean obs | mean pred | MAE | bias (pred-obs) | mean obs / mean pred | sampling (s) | log R2 | median abs(pred-obs)/obs (%) |
|---|---|---|---|---|---|---|---|---|---|
| parallel (incl. UQ, flagged) | 34 | 23.9 | 32.1 | 11.0 | 8.2 | 0.74 | 1/5 | 0.6969 | 31.7 |
| perpendicular | 21 | 33.9 | 30.5 | 7.9 | -3.4 | 1.11 | 1 | 0.8057 | 28.3 |
| unknown | 25 | 47.6 | 36.5 | 26.8 | -11.0 | 1.30 | 1/5 | 0.7119 | 59.0 |

log R2 is natural log of seconds (as experiments/configs/tti_secondary_metrics.json). Within a lab tested at one flux (FPL, TUBS, UQ) or within one flux, log R2 is computed on a narrow spread of observed times and is not informative (e.g. TUBS).

Same-flux perpendicular minus parallel (s):

| lab | flux | n perp/par | obs perp-par | pred perp-par |
|---|---|---|---|---|
| FSRI | 25 | 3/3 | 38.0 | 3.8 |
| FSRI | 50 | 3/3 | 2.7 | 0.5 |
| FSRI | 75 | 3/3 | 6.7 | 0.0 |
| FZJ | 30 | 3/3 | 17.3 | 7.5 |
| FZJ | 60 | 5/4 | 5.8 | -0.5 |
| FZJ | 75 | 4/3 | 5.2 | 0.0 |

UMET stratum: n 6, MAE 23.7 s, bias +23.7 s.

| file | flux | obs_tti_s | pred_tti_primary | pred-obs |
|---|---|---|---|---|
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R1.csv | 30 | 25.0 | 63.0 | 38.0 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R2.csv | 30 | 21.0 | 55.4 | 34.4 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R3.csv | 30 | 15.0 | 66.9 | 51.9 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R1.csv | 60 | 3.0 | 10.0 | 7.0 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R2.csv | 60 | 5.0 | 10.9 | 5.9 |
| Wood/Calibration_Data/UMET/UMET_Wood_Cone_60kW_hor_R3.csv | 60 | 5.0 | 10.1 | 5.1 |

tti_86_secondary (primary_80 + UMET 6; aggregate only, to match the plan's '86 rows' wording): R2 0.5061, MAE 15.7 s.

Secondary variants:

| variant | pool | n | R2 (s) | CI | MAE s | bias s |
|---|---|---|---|---|---|---|
| S1 | primary_80 | 80 | 0.3704 | [-6.2744, 0.5957] | 26.4 | 18.2 |
| S1 | FSRI-lab_18 | 18 | 0.2948 | n/a | 22.4 | 20.4 |
| S1 | other-labs_62 | 62 | 0.3780 | [-11.2461, 0.6096] | 27.6 | 17.6 |
| S1 | tti_25 | 11 | -0.0869 | n/a | 66.3 | 10.8 |
| S2 | primary_80 | 80 | 0.5478 | [-0.5064, 0.8014] | 14.6 | -1.1 |
| S2 | FSRI-lab_18 | 18 | 0.8562 | n/a | 10.4 | -2.2 |
| S2 | other-labs_62 | 62 | 0.4914 | [-1.7660, 0.5273] | 15.8 | -0.8 |
| S2 | tti_25 | 11 | -0.2961 | n/a | 52.2 | -42.1 |
| S1 | UMET 6 | 6 |  |  | 44.5 | 44.5 |
| S2 | UMET 6 | 6 |  |  | 23.7 | 23.7 |

## Edinburgh black cast PMMA TTI (separate stratum; known material (probable), new lab)

n 12: R2 (s) 0.9424, log R2 0.8988, MAE 7.3 s, bias -7.0 s. Inputs: FSRI Black_PMMA material features from exp046's training frame, 6 mm thickness. Only two flux levels (25 and 65 kW/m2), so this R2 mostly reflects the 25-vs-65 difference; the per-flux rows are the result.

| flux | n | mean obs s | pred s | MAE s | bias s |
|---|---|---|---|---|---|
| 25 | 6 | 108.8 | 99.8 | 9.7 | -9.1 |
| 65 | 6 | 14.3 | 9.4 | 5.0 | -5.0 |

Context row, no model: FSRI Black_PMMA replicate-mean TTI at 25 kW/m2 (exp046 training rows, n 6) 110.0 s vs Edinburgh 25 kW/m2 mean 108.8 s.

## Char Residue (exp117), diagnostic only (absolute error; no R2)

Observed = 100 * m(t_flameout) / m0 (README m0), the mass channel interpolated at the README flameout. This is NOT the FSRI end-of-test CR definition, which is unchanged; residue at flameout is a diagnostic comparison, never a CR target.

| variant | pool | n | MAE % | bias % |
|---|---|---|---|---|
| primary | 50 (declared primary: cr_flameout minus UMET) | 50 | 5.8 | 1.6 |
| primary | 53 (all) | 53 | 6.1 | 2.1 |
| primary | 47 (minus UMET and UQ) | 47 | 4.4 | 3.5 |
| S2 | 50 (declared primary: cr_flameout minus UMET) | 50 | 5.8 | 1.7 |
| S2 | 53 (all) | 53 | 6.0 | 2.2 |
| S2 | 47 (minus UMET and UQ) | 47 | 4.4 | 3.5 |
| S3 | 50 (declared primary: cr_flameout minus UMET) | 50 | 4.3 | -1.3 |
| S3 | 53 (all) | 53 | 4.4 | -0.9 |
| S3 | 47 (minus UMET and UQ) | 47 | 2.7 | 0.4 |

S2: G2 density; S3: all five category flags 0.

Per lab (primary):

| lab | n | mean obs (flameout, README m0) % | mean obs (channel m0) % [report] | mean pred % | MAE | bias | mean end-of-test residue % [extra] | MAE vs end-of-test [extra] |
|---|---|---|---|---|---|---|---|---|
| Aalto | 5 | 21.3 | 20.3 | 21.4 | 4.1 | 0.1 | 21.1 | 4.1 |
| FPL | 5 | 15.7 | 15.7 | 18.6 | 2.9 | 2.9 | 14.3 | 4.3 |
| FSRI | 18 | 17.3 | 17.3 | 20.3 | 4.2 | 3.0 | 16.0 | 4.7 |
| IMT | 12 | 16.5 | 16.5 | 21.5 | 5.2 | 5.0 | 15.5 | 6.4 |
| UDRI | 7 | 16.9 | 16.9 | 21.5 | 5.0 | 4.6 | 16.2 | 5.4 |
| UMET (separate stratum) | 3 | 15.4 | 15.4 | 25.7 | 10.3 | 10.3 | 11.5 | 14.2 |
| UQ (flag: README mass fluctuations) | 3 | 44.4 | 44.9 | 17.1 | 27.4 | -27.4 | 44.4 | 27.4 |

FSRI-lab per flux:

| FSRI-lab flux | n | mean obs % | mean pred % | bias |
|---|---|---|---|---|
| 25 | 6 | 17.7 | 26.8 | 9.1 |
| 50 | 6 | 17.4 | 18.1 | 0.7 |
| 75 | 6 | 16.9 | 16.2 | -0.7 |

Per test:

| lab | file | flux | orientation | t_flameout_reported_s | cr_flameout_readme_m0_pct | cr_flameout_channel_m0_pct | pred_cr_primary | eot_residue_pct |
|---|---|---|---|---|---|---|---|---|
| Aalto | Wood/Calibration_Data/Aalto/Aalto_Wood_Cone_30kW_hor_parallel_R1.csv | 30 | parallel | 1854.0 | 22.9 | 22.5 | 25.4 | 22.8 |
| Aalto | Wood/Calibration_Data/Aalto/Aalto_Wood_Cone_30kW_hor_parallel_R3.csv | 30 | parallel | 2156.0 | 18.7 | 18.2 | 25.4 | 18.7 |
| Aalto | Wood/Calibration_Data/Aalto/Aalto_Wood_Cone_45kW_hor_parallel_R1.csv | 45 | parallel | 1500.0 | 23.4 | 22.1 | 20.4 | 23.1 |
| Aalto | Wood/Calibration_Data/Aalto/Aalto_Wood_Cone_60kW_hor_parallel_R1.csv | 60 | parallel | 1492.0 | 16.3 | 15.6 | 17.9 | 16.2 |
| Aalto | Wood/Calibration_Data/Aalto/Aalto_Wood_Cone_60kW_hor_parallel_R2.csv | 60 | parallel | 1090.0 | 24.9 | 23.2 | 17.9 | 24.9 |
| FPL | Wood/Calibration_Data/FPL/FPL_Wood_Cone_50kW_hor_R1.csv | 50 | unknown | 1661.0 | 15.7 | 15.7 | 18.6 | 14.4 |
| FPL | Wood/Calibration_Data/FPL/FPL_Wood_Cone_50kW_hor_R3.csv | 50 | unknown | 1635.0 | 18.3 | 18.3 | 18.6 | 16.9 |
| FPL | Wood/Calibration_Data/FPL/FPL_Wood_Cone_50kW_hor_R4.csv | 50 | unknown | 1563.0 | 10.5 | 10.6 | 18.6 | 9.1 |
| FPL | Wood/Calibration_Data/FPL/FPL_Wood_Cone_50kW_hor_R5.csv | 50 | unknown | 1694.0 | 16.9 | 17.0 | 18.6 | 15.4 |
| FPL | Wood/Calibration_Data/FPL/FPL_Wood_Cone_50kW_hor_R6.csv | 50 | unknown | 1501.0 | 17.1 | 17.1 | 18.6 | 15.7 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Parallel_R1.csv | 25 | parallel | 2279.8 | 12.7 | 12.6 | 26.8 | 11.6 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Parallel_R2.csv | 25 | parallel | 2131.8 | 15.6 | 15.6 | 26.8 | 14.2 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Parallel_R3.csv | 25 | parallel | 2050.2 | 15.7 | 15.7 | 26.7 | 14.7 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Perpendicular_R1.csv | 25 | perpendicular | 1957.8 | 20.3 | 20.3 | 26.7 | 19.2 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Perpendicular_R2.csv | 25 | perpendicular | 2001.5 | 21.1 | 21.0 | 26.8 | 19.9 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_25kW_hor_Perpendicular_R3.csv | 25 | perpendicular | 1821.0 | 20.6 | 20.6 | 26.8 | 19.4 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Parallel_R1.csv | 50 | parallel | 1460.5 | 13.1 | 13.4 | 17.8 | 11.9 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Parallel_R2.csv | 50 | parallel | 1405.0 | 16.2 | 16.2 | 17.8 | 14.8 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Parallel_R3.csv | 50 | parallel | 1449.0 | 15.4 | 15.3 | 17.8 | 13.9 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Perpendicular_R1.csv | 50 | perpendicular | 1234.2 | 19.9 | 20.0 | 18.6 | 18.6 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Perpendicular_R2.csv | 50 | perpendicular | 1278.0 | 19.1 | 19.1 | 18.6 | 17.9 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_50kW_hor_Perpendicular_R3.csv | 50 | perpendicular | 1213.0 | 20.6 | 20.6 | 17.8 | 19.2 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Parallel_R1.csv | 75 | parallel | 1220.0 | 14.9 | 14.9 | 15.8 | 13.7 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Parallel_R2.csv | 75 | parallel | 1199.5 | 15.3 | 15.3 | 16.6 | 13.9 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Parallel_R3.csv | 75 | parallel | 1258.0 | 17.6 | 17.6 | 16.6 | 16.4 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Perpendicular_R1.csv | 75 | perpendicular | 902.8 | 17.5 | 17.5 | 15.8 | 16.0 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Perpendicular_R2.csv | 75 | perpendicular | 969.5 | 17.2 | 17.2 | 16.6 | 15.8 |
| FSRI | Wood/Calibration_Data/FSRI/FSRI_Wood_Cone_75kW_hor_Perpendicular_R3.csv | 75 | perpendicular | 890.5 | 18.7 | 18.7 | 15.8 | 17.4 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_25kW_hor_R1.csv | 25 | unknown | 2301.0 | 15.5 | 15.5 | 26.7 | 14.3 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_25kW_hor_R2.csv | 25 | unknown | 2178.0 | 15.1 | 15.2 | 26.8 | 13.8 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_25kW_hor_R3.csv | 25 | unknown | 2273.0 | 18.0 | 18.0 | 26.8 | 16.2 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_25kW_hor_R4.csv | 25 | unknown | 1788.0 | 21.2 | 21.1 | 26.8 | 20.0 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_25kW_hor_R5.csv | 25 | unknown | 2155.0 | 18.4 | 18.4 | 26.8 | 17.1 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_50kW_hor_R1.csv | 50 | unknown | 1505.0 | 16.5 | 16.5 | 19.1 | 15.4 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_50kW_hor_R2.csv | 50 | unknown | 1613.0 | 13.5 | 13.7 | 18.6 | 12.5 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_50kW_hor_R3.csv | 50 | unknown | 1673.0 | 15.0 | 15.1 | 18.8 | 13.9 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_50kW_hor_R4.csv | 50 | unknown | 1374.0 | 15.3 | 15.3 | 18.6 | 14.0 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_75kW_hor_R1.csv | 75 | unknown | 1206.0 | 16.7 | 16.8 | 16.6 | 19.0 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_75kW_hor_R2.csv | 75 | unknown | 1159.0 | 16.0 | 16.0 | 17.0 | 14.9 |
| IMT | Wood/Calibration_Data/IMT/IMT_Wood_Cone_75kW_hor_R3.csv | 75 | unknown | 1111.0 | 16.8 | 16.8 | 15.8 | 15.4 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_30kW_hor_R1.csv | 30 | unknown | 2250.0 | 16.9 | 16.9 | 25.4 | 15.9 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_30kW_hor_R2.csv | 30 | unknown | 2010.0 | 18.4 | 18.4 | 25.7 | 18.2 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_30kW_hor_R3.csv | 30 | unknown | 1920.0 | 20.3 | 20.3 | 25.3 | 19.4 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_60kW_hor_R1.csv | 60 | unknown | 1440.0 | 19.9 | 19.9 | 18.4 | 18.8 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_60kW_hor_R2.csv | 60 | unknown | 1500.0 | 16.1 | 16.1 | 18.4 | 15.6 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_60kW_hor_R3.csv | 60 | unknown | 1380.0 | 14.3 | 14.3 | 18.4 | 13.7 |
| UDRI | Wood/Calibration_Data/UDRI/UDRI_Wood_Cone_60kW_hor_R4.csv | 60 | unknown | 1380.0 | 12.3 | 12.3 | 18.4 | 11.6 |
| UMET | Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R1.csv | 30 | unknown | 2001.0 | 13.5 | 13.5 | 25.8 | 10.5 |
| UMET | Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R2.csv | 30 | unknown | 2105.0 | 17.7 | 17.7 | 25.8 | 14.2 |
| UMET | Wood/Calibration_Data/UMET/UMET_Wood_Cone_30kW_hor_R3.csv | 30 | unknown | 2020.0 | 14.9 | 14.9 | 25.4 | 9.8 |
| UQ | Wood/Calibration_Data/UQ/UQ_Wood_Cone_60kW_hor_Parallel_R1.CSV | 60 | parallel | 908.0 | 36.7 | 37.5 | 17.0 | 36.7 |
| UQ | Wood/Calibration_Data/UQ/UQ_Wood_Cone_60kW_hor_Parallel_R2.CSV | 60 | parallel | 920.0 | 46.4 | 46.5 | 17.1 | 46.4 |
| UQ | Wood/Calibration_Data/UQ/UQ_Wood_Cone_60kW_hor_Parallel_R3.CSV | 60 | parallel | 918.0 | 50.2 | 50.8 | 17.1 | 50.2 |

## Stop-time table (end-of-test residue; a warning about the FSRI end-of-test CR definition for charring materials, never grounds to change it)

| lab | n | residue mean % | residue min % | residue max % | duration mean s | duration min s | duration max s | stop rule (README) |
|---|---|---|---|---|---|---|---|---|
| Aalto | 9 | 21.9 | 16.2 | 28.4 | 1534.4 | 1095.0 | 2160.0 | not stated for cone-in-air tests; t_flameout reported (from start of DAQ); 'Test duration' filled only for N2 gasification |
| BUWfd | 5 | 11.2 | 8.4 | 14.3 | 2100.0 | 2100.0 | 2100.0 | README has no cone metadata |
| FPL | 6 | 15.3 | 9.1 | 20.2 | 1676.0 | 1400.0 | 1825.0 | not stated (ASTM E1354-25); t_flameout reported; R2 DAQ lost at 1400 s, flameout recorded by hand |
| FSRI | 18 | 16.0 | 11.6 | 19.9 | 1603.8 | 1010.0 | 2399.0 | not stated (ASTM E1354, general accordance); flameout by operator button |
| FZJ | 20 | 0.8 | -0.3 | 7.2 | 5021.9 | 3124.0 | 10770.0 | not stated (ISO 5660); no flameout column |
| IMT | 12 | 15.5 | 12.5 | 20.0 | 1829.2 | 1215.0 | 2440.0 | not stated (ISO 5660); t_flameout reported |
| TUBS | 3 | 8.3 | 7.7 | 9.0 | 2700.0 | 2700.0 | 2700.0 | not stated (ISO 5660); t_flameout '-' |
| UDRI | 7 | 16.2 | 11.6 | 19.4 | 1763.0 | 1436.0 | 2359.0 | not stated (ASTM E1354-25); t_flameout reported |
| UMET | 6 | 9.1 | 3.4 | 14.2 | 2069.2 | 1840.0 | 2347.0 | not stated (ISO 13927 mass-loss cone); t_flameout reported |
| UQ (flag: README mass fluctuations) | 3 | 44.4 | 36.7 | 50.2 | 915.3 | 908.0 | 920.0 | not stated (ISO 5660); t_flameout reported; README: 60 kW/m2 mass readings fluctuated |

| FSRI DB material (training data, no model) | n | end_of_test_time_s range | residue_pct range |
|---|---|---|---|
| Pine_Siding | 9 | 684.8 - 1521.2 | 12.8 - 65.8 |
| Wood_Stud | 9 | 1917.0 - 3384.0 | 16.3 - 50.6 |

## Reported-only extras

- pHRR R2 on lab x flux x orientation cell means: 0.2753 (23 cells).
- TTI R2 (s) on lab x flux x orientation cell means: 0.6743 (23 cells).

Flux response, OLS slope on log(q):

| target | pool | n | fluxes | slope log(obs) | slope log(pred) |
|---|---|---|---|---|---|
| pHRR | primary_80 | 80 | 25/30/45/50/60/75 | 0.594 | 0.543 |
| pHRR | Aalto | 9 | 30/45/60 | 0.519 | 0.545 |
| pHRR | FSRI | 18 | 25/50/75 | 0.578 | 0.534 |
| pHRR | FZJ | 22 | 30/60/75 | 0.557 | 0.600 |
| pHRR | IMT | 12 | 25/50/75 | 0.456 | 0.556 |
| TTI | primary_80 | 80 | 25/30/45/50/60/75 | -2.372 | -2.473 |
| TTI | Aalto | 9 | 30/45/60 | -1.807 | -2.500 |
| TTI | FSRI | 18 | 25/50/75 | -2.287 | -2.461 |
| TTI | FZJ | 22 | 30/60/75 | -1.862 | -2.530 |
| TTI | IMT | 12 | 25/50/75 | -3.810 | -2.417 |

exp074 predictions by flux on primary_80 (FSRI trains on 15/25/35/50/75/100 kW/m2):

| flux | n | pred mean | pred min | pred max |
|---|---|---|---|---|
| 25 | 11 | 196.6 | 193.0 | 200.5 |
| 30 | 15 | 196.8 | 193.0 | 200.8 |
| 45 | 3 | 288.4 | 288.1 | 288.5 |
| 50 | 16 | 284.3 | 275.9 | 288.5 |
| 60 | 19 | 281.6 | 275.9 | 288.5 |
| 75 | 16 | 353.4 | 348.5 | 366.3 |

exp074 at G2 geometry and the MaCFP material inputs, swept over flux:

| flux | exp074 pred at G2 geometry (kW/m2) |
|---|---|
| 15 | 193.4 |
| 25 | 193.4 |
| 30 | 193.4 |
| 35 | 275.2 |
| 45 | 275.9 |
| 50 | 275.9 |
| 60 | 275.9 |
| 75 | 352.3 |
| 100 | 352.3 |

## Follow-up

- Step 6 of the config: a fresh-context critic pass before any number here is quoted in STATUS or a model card. Any leaderboard row is the orchestrator's decision after the critic.
