# TTI conformal intervals: 681 rows vs 675 rows (report-only)

Config: `experiments/configs/conformal_intervals.json`, re-run unchanged on the TTI rows as that config's caveat asks ("If D4 makes 675 rows canonical, re-run this config unchanged"). The only change is the row set. `tti_round3.json` D4b made S675 canonical: the 681-row population minus the 6 `data_qa_decision.flagged_ids` replicates (HDPE|HF25|R3, HDPE|HF25|R4, Nylon|HF25-Frame|R4, Nylon|HF75|R1, PET|HF25|R2, Overstuffed_Chair_Polyester_Fabric|HF25|R2). Five of those rows are at 25 kW/m2 and one at 75. The S675 run has 675 rows and 76 material groups. Raw flux counts are {15: 3, 25: 218, 35: 16, 50: 221, 75: 216, 100: 1}, and the strata are 25: 237, 50: 221 and 75: 217 rows (73/72/72 groups).

This is report-only. No R2, CI, leaderboard row or number of record changes, and nothing was written to `experiments/results/`.

## Step 1 (record reproduction)

| rows | record | reproduced GKF mean | CI | within 0.002 |
|---|---|---|---|---|
| 681 (existing run) | exp046 0.3133 [0.071, 0.496] | 0.3133 | [0.0715, 0.4958] | True |
| 675 (this run) | exp154 0.3179 [0.0917, 0.5067] (numbers_of_record.json) | 0.3179 | [0.0917, 0.5067] | True |

Nested-disjointness asserts passed on S675, and every row got exactly one interval per repeat.

## Reading rule (verbatim from the config)

> Applied to group_mean_coverage, for every variant, overall and per stratum: 'consistent with nominal' if its 95% CI contains 0.90; 'under-covers' if the CI upper bound is below 0.90; 'conservative' if the CI lower bound is above 0.90. For group_all, wherever groups have more than one row in the cell (overall for every target, since FSRI materials have rows at several fluxes; per stratum for TTI, CR, LOI and wherever 15/35/100 kW/m2 rows join an anchor), say it is expected to fall below 0.90, because the primary interval is per-row, not simultaneous; group_all gets no reading-rule label. No variant is ever called better or worse; the tables are only juxtaposed.

group_all is expected to fall below 0.90 in every cell below, because groups have more than one row in each cell and the interval is per-row, not simultaneous. It gets no reading-rule label.

## Headline (P, primary)

On S675, P's group_mean coverage is 0.922 [0.873, 0.963] overall, which is consistent with nominal. On 681 rows it was 0.921 [0.872, 0.961]. P's median raw-unit width (display-clipped, nested, mean over repeats) per stratum is:

| stratum | 681 rows | 675 rows |
|---|---|---|
| 25 kW/m2 | 1444 s | 1434 s |
| 50 kW/m2 | 196.6 s | 194.9 s |
| 75 kW/m2 | 57.76 s | 55.76 s |

Every group_mean cell for P, V1, V2 and V3 keeps the label "consistent with nominal" on both row sets. CQR is the only variant whose label differs: overall it goes from 0.934 [0.897, 0.966] "consistent with nominal" on 681 rows to 0.937 [0.901, 0.969] "conservative" on 675 rows, and at 50 kW/m2 from 0.934 [0.891, 0.971] to 0.942 [0.901, 0.976], also "conservative". In both cells the CI lower bound moved above 0.90, by 0.001. All coverage changes between 681 and 675 rows are at most 0.016 in group_all and at most 0.008 in group_mean. Both are well inside the binomial reference SD of about 0.034 at G = 76.

## Tables (each cell shows 681 / 675)

Widths are means over the 10 repeats. Raw widths are display-clipped at 0 s, and score widths are ln-scale and unclipped, except V3, whose widths are in seconds. Unbounded and empty intervals are 0 in every cell for both row sets.

### P: primary (deployed): group-weighted CV+, Mondrian by flux anchor (FSRI) or one stratum (LOI)

| cell | rows 681 / 675 | groups 681 / 675 | group_all 681 / 675 | group_mean [95% CI] reading, 681 | group_mean [95% CI] reading, 675 | row 681 / 675 | median raw width (s) 681 / 675 | mean score width 681 / 675 | unbounded 681 / 675 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 681 / 675 | 76 / 76 | 0.830 / 0.829 | 0.921 [0.872, 0.961] consistent with nominal | 0.922 [0.873, 0.963] consistent with nominal | 0.916 / 0.917 | 206.7 / 199.8 | 4.536 / 4.531 | 0 / 0 |
| 25 kW/m2 | 242 / 237 | 73 / 73 | 0.879 / 0.886 | 0.910 [0.856, 0.960] consistent with nominal | 0.913 [0.860, 0.963] consistent with nominal | 0.920 / 0.922 | 1444 / 1434 | 5.090 / 5.099 | 0 / 0 |
| 50 kW/m2 | 221 / 221 | 72 / 72 | 0.885 / 0.885 | 0.916 [0.863, 0.963] consistent with nominal | 0.916 [0.865, 0.963] consistent with nominal | 0.914 / 0.914 | 196.6 / 194.9 | 4.444 / 4.440 | 0 / 0 |
| 75 kW/m2 | 218 / 217 | 72 / 72 | 0.887 / 0.887 | 0.918 [0.869, 0.961] consistent with nominal | 0.919 [0.869, 0.961] consistent with nominal | 0.915 / 0.915 | 57.76 / 55.76 | 4.013 / 4.004 | 0 / 0 |

### V1: report-only: P with one stratum (no Mondrian)

| cell | rows 681 / 675 | groups 681 / 675 | group_all 681 / 675 | group_mean [95% CI] reading, 681 | group_mean [95% CI] reading, 675 | row 681 / 675 | median raw width (s) 681 / 675 | mean score width 681 / 675 | unbounded 681 / 675 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 681 / 675 | 76 / 76 | 0.828 / 0.828 | 0.916 [0.864, 0.958] consistent with nominal | 0.917 [0.865, 0.960] consistent with nominal | 0.911 / 0.912 | 212 / 208 | 4.500 / 4.490 | 0 / 0 |
| 25 kW/m2 | 242 / 237 | 73 / 73 | 0.838 / 0.848 | 0.871 [0.806, 0.933] consistent with nominal | 0.874 [0.809, 0.938] consistent with nominal | 0.884 / 0.885 | 1068 / 1065 | 4.503 / 4.494 | 0 / 0 |
| 50 kW/m2 | 221 / 221 | 72 / 72 | 0.892 / 0.890 | 0.923 [0.873, 0.968] consistent with nominal | 0.920 [0.869, 0.966] consistent with nominal | 0.920 / 0.918 | 200.1 / 197.7 | 4.495 / 4.484 | 0 / 0 |
| 75 kW/m2 | 218 / 217 | 72 / 72 | 0.911 / 0.915 | 0.936 [0.892, 0.976] consistent with nominal | 0.940 [0.896, 0.978] consistent with nominal | 0.933 / 0.936 | 75.06 / 73.55 | 4.501 / 4.491 | 0 / 0 |

### V2: report-only: P with all w_i = 1 (row-exchangeable textbook CV+)

| cell | rows 681 / 675 | groups 681 / 675 | group_all 681 / 675 | group_mean [95% CI] reading, 681 | group_mean [95% CI] reading, 675 | row 681 / 675 | median raw width (s) 681 / 675 | mean score width 681 / 675 | unbounded 681 / 675 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 681 / 675 | 76 / 76 | 0.808 / 0.811 | 0.909 [0.857, 0.953] consistent with nominal | 0.911 [0.859, 0.955] consistent with nominal | 0.904 / 0.905 | 190.4 / 186.1 | 4.327 / 4.333 | 0 / 0 |
| 25 kW/m2 | 242 / 237 | 73 / 73 | 0.852 / 0.868 | 0.883 [0.822, 0.942] consistent with nominal | 0.891 [0.833, 0.948] consistent with nominal | 0.896 / 0.901 | 1221 / 1221 | 4.750 / 4.776 | 0 / 0 |
| 50 kW/m2 | 221 / 221 | 72 / 72 | 0.879 / 0.875 | 0.909 [0.855, 0.958] consistent with nominal | 0.908 [0.855, 0.957] consistent with nominal | 0.906 / 0.906 | 185.3 / 180.6 | 4.298 / 4.294 | 0 / 0 |
| 75 kW/m2 | 218 / 217 | 72 / 72 | 0.882 / 0.879 | 0.915 [0.865, 0.960] consistent with nominal | 0.914 [0.864, 0.958] consistent with nominal | 0.911 / 0.910 | 54.41 / 52.88 | 3.887 / 3.889 | 0 / 0 |

### V3: report-only (TTI): Poisson-normalised score |y - mu| / sqrt(mu) in seconds, group-weighted, Mondrian

| cell | rows 681 / 675 | groups 681 / 675 | group_all 681 / 675 | group_mean [95% CI] reading, 681 | group_mean [95% CI] reading, 675 | row 681 / 675 | median raw width (s) 681 / 675 | mean score width 681 / 675 | unbounded 681 / 675 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 681 / 675 | 76 / 76 | 0.786 / 0.793 | 0.925 [0.888, 0.958] consistent with nominal | 0.927 [0.889, 0.960] consistent with nominal | 0.921 / 0.922 | 57.79 / 57.61 | 190.955 / 191.899 | 0 / 0 |
| 25 kW/m2 | 242 / 237 | 73 / 73 | 0.862 / 0.871 | 0.917 [0.868, 0.960] consistent with nominal | 0.920 [0.871, 0.964] consistent with nominal | 0.918 / 0.922 | 328.2 / 332.2 | 441.805 / 448.932 | 0 / 0 |
| 50 kW/m2 | 221 / 221 | 72 / 72 | 0.901 / 0.906 | 0.925 [0.876, 0.967] consistent with nominal | 0.928 [0.879, 0.969] consistent with nominal | 0.920 / 0.921 | 54.82 / 54.67 | 70.572 / 70.676 | 0 / 0 |
| 75 kW/m2 | 218 / 217 | 72 / 72 | 0.894 / 0.896 | 0.925 [0.874, 0.969] consistent with nominal | 0.924 [0.873, 0.968] consistent with nominal | 0.926 / 0.924 | 24.28 / 24.11 | 34.529 / 34.633 | 0 / 0 |

### CQR: report-only: CQR with HGB quantile 0.05 / 0.95 models on the score scale, group-weighted, Mondrian

| cell | rows 681 / 675 | groups 681 / 675 | group_all 681 / 675 | group_mean [95% CI] reading, 681 | group_mean [95% CI] reading, 675 | row 681 / 675 | median raw width (s) 681 / 675 | mean score width 681 / 675 | unbounded 681 / 675 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 681 / 675 | 76 / 76 | 0.808 / 0.813 | 0.934 [0.897, 0.966] consistent with nominal | 0.937 [0.901, 0.969] conservative | 0.930 / 0.934 | 92.79 / 92.27 | 3.745 / 3.781 | 0 / 0 |
| 25 kW/m2 | 242 / 237 | 73 / 73 | 0.863 / 0.870 | 0.919 [0.872, 0.962] consistent with nominal | 0.927 [0.886, 0.966] consistent with nominal | 0.931 / 0.938 | 647.7 / 677.4 | 4.600 / 4.664 | 0 / 0 |
| 50 kW/m2 | 221 / 221 | 72 / 72 | 0.906 / 0.914 | 0.934 [0.891, 0.971] consistent with nominal | 0.942 [0.901, 0.976] conservative | 0.932 / 0.939 | 81.51 / 81.71 | 3.334 / 3.382 | 0 / 0 |
| 75 kW/m2 | 218 / 217 | 72 / 72 | 0.900 / 0.893 | 0.931 [0.883, 0.970] consistent with nominal | 0.928 [0.880, 0.968] consistent with nominal | 0.928 / 0.924 | 37.61 / 38.29 | 3.212 / 3.222 | 0 / 0 |

### Diagnostic: P row coverage by raw heat flux (no CIs)

| raw flux (kW/m2) | n 681 / 675 | row coverage 681 / 675 |
|---|---|---|
| 15 | 3 / 3 | 1.000 / 1.000 |
| 25 | 223 / 218 | 0.914 / 0.916 |
| 35 | 16 / 16 | 0.988 / 0.988 |
| 50 | 221 / 221 | 0.914 / 0.914 |
| 75 | 217 / 216 | 0.914 / 0.914 |
| 100 | 1 / 1 | 1.000 / 1.000 |

## Notes

- The 681-row column is taken from the existing `experiments/runs/conformal/tti_summary.json` (Oct 7 run). The default `--tti-rows S681` path was re-run on this machine into a scratch directory as an identity check. Every coverage frequency matched exactly, and bounds and widths differed by less than 1e-9 relative (floating-point differences between machines).
- `experiments/runs/conformal_tti_s675.log` contains about 480 numpy `RuntimeWarning`s ("divide by zero / overflow / invalid value encountered in matmul") from sklearn's GLM and imputer predict. The 681-row identity re-run on this machine emits the same warnings and still matches the committed outputs, so they are a property of this interpreter's BLAS, not of S675. Step 1 matches exp154 exactly.
- The honesty caveats in the config apply unchanged. Exchangeability holds only between materials. A coverage estimate has an SD of about 3-4 points. Wide intervals at 25 kW/m2 are the honest form of a known TTI weakness.
