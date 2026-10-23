# Critic: litcone_gt_round1 (FINAL, --v3, primary + variant_C)
Verdict: **SOUND WITH CAVEATS**. The primary model is clean. **variant_C is not quotable as-is**, because its control feature leaks the target into the control rows. Its P2/P3 headline numbers are inflated by those rows predicting themselves.

## Leakage (Q1): variant_C control feature
- `control_feature` builds `ctrl` from all `is_control == 1` rows, and every row is then scored, the control rows included. As a result, a control row's `ctrl_log` is computed from its own target.
- Self-leak: in 582/755 TTI, 706/879 pHRR, 565/706 THR and 133/172 avg-HRR control rows, `ctrl_log == log1p(y)` exactly. In the remaining control rows it is a mean that includes the row's own target. Control rows are about 22% of each set.
- The config said "the control row itself is scored like any other row", so this was declared, but it is still label leakage. A row cannot be "given" its own label.
- Non-control rows in P3: the feature uses the test paper's own control result, which is a measured target value from the test fold. This counts as legitimate 'given-control' use only if the claim is framed as "predict an FR formulation given the paper's measured neat-polymer result". It must never be reported as a prediction from composition alone.
- Scratch recompute (this critic, scratchpad copy, OMP_NUM_THREADS=6), variant_C, rep 0. Columns:
  - P2/P3 all = R2 over all rows.
  - excl. controls = R2 with the `is_control` rows removed from scoring.
  - control rows only = R2 on the control rows alone.

| target | P2 all | P2 excl. controls | P3 all | P3 excl. controls | P3 control rows only |
|---|---|---|---|---|---|
| TTI | 0.678 | 0.607 | -0.571 | -1.009 | 0.267 |
| pHRR | 0.854 | 0.763 | 0.717 | **0.505** | 0.927 |
| THR | 0.887 | 0.862 | 0.718 | **0.637** | 0.923 |
| avg_HRR | 0.892 | 0.843 | 0.748 | **0.562** | 0.932 |

  The P3 "all" values match the log's rep-0 numbers closely (the log reports 0.711/0.724/0.746 as 10-repeat means). Of the pHRR P3 gain over primary (0.29 -> 0.72), about half comes from control rows that are given their own label.
- Fix, under a new amendment:
  - Set `ctrl_log` = NaN on the control rows, or drop the control rows from variant_C's scored set.
  - Report the remaining figure as "given the paper's measured control".
- Naive baseline (predict = the control's value, non-control rows): R2 TTI 0.54, pHRR -1.64, THR 0.49, avg-HRR -0.88. The model does add over simply copying the control.

## Imputer and scaler (Q2)
- The imputer, scaler and Ridge sit inside each pipeline. `fit_pred` clones the model per fold, and so does `Avg.fit`. Clipping uses the training-fold `ytr` range only. P3 asserts that groups are disjoint. OK.
- Matrix categories (>= 3 papers), the top-100 additive vocabulary and the form categories are computed on the whole pool. They use names and counts only, no labels, so this is transductive but label-free. Minor.

## Dedupe (Q3)
- Paper-level dedupe (amendment 2) and the row key are implemented as declared. The snapshot has 4,529 rows / 806 papers; only 5 groups lack a DOI.
- Identical (pHRR, TTI, THR, flux) tuples appear under 9 different paper pairs (19 tuples). Examples:
  - 10.3390/molecules25214963 and 10.3390/polym11010180 share 7 tuples.
  - 10.1021/acsomega.1c03393 and 10.3390/polym13132167 share 4 tuples.
  - One EBSCO no-DOI group matches a DOI paper.
- These are likely re-published data or the same lab's reused controls. They put the same values on both sides of a P3 split. The volume is small (fewer than 1% of rows), but these pairs should be merged into one group.

## Wording (Q4)
- Every P1 print carries "THEIR PROTOCOL: computed on training data; not held-out evidence". No "beats" or "SOTA" appears in the log or script. The competitor's values are labelled as a training-data fit. OK.
- Under the published (training-data) protocol, no target reaches the competitor's values. Best P1: variant_C avg-HRR 0.946 vs 0.96 and THR 0.927 vs 0.97. "Matches/exceeds" therefore must not be used.
- Cosmetic: the final print says it wrote `litcone_gt_round1.json`, but the file actually written is `litcone_gt_round1_v3.json`.

## Quotable now
- Primary, P3 paper-grouped: TTI -0.25, pHRR 0.29, THR 0.24, avg-HRR 0.23. P2 random: 0.38 / 0.60 / 0.68 / 0.68.
- variant_C: only the control-excluded numbers above, labelled "given measured control".
