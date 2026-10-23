# Critic: LOI round 6 (c3, c4, c5 vs exp092), proxy-free report
Verdict: **SOUND WITH CAVEATS**. The round ran as declared, exp092 reproduces, nothing replaces exp092, and the proxy-free pick (c3) follows the declared rule. However, c3 is the model with the *known* typing errors. Report it together with c4 as a range, not as one clean number.

## Pre-declaration and implementation (Q1)
- Order: config and code were committed at 11:51:11 (bd467e0). The code mtime is 11:52:49, the log ends at 12:05:58 (775 s run), and results were committed at 12:06 (40f581e). OK.
- Prefix: a diff against train_loi_round5.py shows the code is verbatim up to round-5 FEAT_C1/FEAT_C2. Three things differ: the docstring, the round-5 c1/c2 fit block being removed, and run_cand being refactored with identical logic.
- exp089 reproduces to 0.3542 [0.1694, 0.4664]. exp092 reproduces to 0.3738 [0.2007, 0.4701], within 0.002.
- F6_v6:
  - R1 is the RX_PES pattern plus PET-glycolysis terms. R2 is RX_PET plus the full PPG/PEG names.
  - R3 (RX_BIO_FIRST) is tested before PES and PET, as declared.
  - R4 is `sum(has)==1 and not has[0]`, which falls back to the first-listed component.
  - "Stated amount" means after split_components' amount filter (>= 2 amounts or exactly 1 unit amount). That is consistent with F6_fixed.
  - The audit was printed before any fit: 8 strings, 29 benchmark rows, 7 papers. Every change matches the round-5 critic's list.
- Feature sets: c3 = FEAT_C1 + log_density (47 features, no std_*). c4 swaps only the pt_* columns. c5 = c4 + std5_* (52 features).
- 4589-3 tie: std5 relabels only rows already labelled ISO_4589 whose text contains "ISO 4589-3".
  - In the data no string names 4589-3 together with another standard. The only 4589-3 string is "PN-EN ISO 4589-3:2017-06" (14 rows), so the tie rule is never exercised.
  - Counts: 144 / 59 / 85 / 14 / 158.
- Selection: c3 is the max B among c3-c5. It gets 2/10 wins and ci_lo 0.183 < 0.2007, so NOT adopted. This is correct.
- Proxy-free report = max(c3, c4) by B mean, i.e. c3. This is correct per the config.
## Leakage / protocol (Q2)
- Same y and groups (asserted per candidate). Same rf() (500 trees, leaf 2, rs 0). Same ep.repeated_group_kfold seeds, A 80/20 x50, C by dom. No LOI-derived feature. No difference from exp092 except the declared columns.
## Is c3 proxy-free? (Q3)
- Yes for the test-standard metadata: no std_* in the features list of exp093.
- Residual provenance-like columns remain:
  - pt_unknown, which marks "polyol not stated / unparsed" (78 rows, 19 papers).
  - el_imputed_share and resolved_fr_wt_share, which measure extraction completeness.
- These columns are not doing the work. Dropping pt_unknown gives B 0.3594 (unchanged: the one-hot is redundant for an RF). Dropping el_imputed_share gives 0.3592.
- So "proxy-free" holds for the explicit lab/standard proxy. Paper/lab clustering across papers (same group, several papers) is not removed by paper-GKF, for any candidate.
- log_density is a monotone duplicate of density, so it is an RF no-op. c3 is therefore effectively round-5 c1 (exp091, 0.361 [0.197, 0.475]).
## Reporting c3 0.359 [0.183, 0.474] (Q4)
- It is acceptable as a supplementary number, because the rule was pre-declared and applied regardless of direction. But note three things:
  - (a) RF tie-break noise is about 0.005. Permuting c3's column order alone gives 0.353-0.356, and c3 vs exp091 differs by 0.002.
  - (b) It is a max of 2, and the winner is the version with typing errors the project already knows about.
  - (c) The proxy cost is exp092 - c3 = -0.0145 (2/10 wins). That is consistent with the round-5 estimate that about +0.013 came from std.
- Suggested wording: "proxy-free LOI B 0.36 [0.18, 0.47] (c3); 0.34 [0.15, 0.45] with corrected polyol typing (c4)".
## Why c4 < c3 (Q5): read-only rerun, all 10 repeats
- c4 - c3 = -0.0175, 1/10 wins, which is about 3x the RF tie-break noise. Ablation, applying one change group to c3 at a time:
  - PPG -> polyether (R2, 9 rows, all paper M079): -0.0124, 2/10.
  - bio (R3/R4, 14 rows): -0.0023.
  - PET (R1, 6 rows): -0.0028.
- Per-paper dSSE (+100 total):
  - M079 alone accounts for +38. Changed papers in total account for only 24%: M058 (-13) and N022 (-5) improve.
  - 76% of the loss is on unchanged papers (M046 +22, M067 +19, M015 +18, M110 +14). Relabelling training rows reshapes the shared trees.
- Most plausible explanation: the "unknown" level was a cluster label, not chemistry.
  - M079 (a top-SSE paper in round 5 too) behaves more like the other "unknown" papers than like the polyether mass (256 rows).
  - Putting it among the polyethers pulls its predictions and those of neighbouring papers.
- So yes, the unknown level was acting partly as a paper/reporting-style proxy. The gap rests mainly on one paper (M079), so do not generalise to "correct typing hurts".
- New typing error: the N069 string "PET glycolysis (r-PUF) / commercial polyol (c-PUF)" labels all 4 rows polyester_aromatic. The c-PUF control rows (commercial polyol) are probably mislabelled. The effect is negligible (dSSE -0.65).
## Caveats to carry
1. exp092 stays the record: 0.374 [0.201, 0.470]. The proxy-free number is 0.359 [0.183, 0.474] (c3). Always give c4's 0.342 alongside it.
2. Most of c3 > c4 comes from one paper (M079) and spillover onto unchanged papers. It is evidence that "unknown" encodes reporting style, not that the old typing is better.
3. Proxy-free removes the test-standard metadata only. Lab clustering across papers and extraction-completeness columns remain, though removing them changes nothing measurable.
