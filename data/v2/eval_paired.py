#!/usr/bin/env python3
"""Dravix V2 -- paired same-rows, same-folds ablation delta with a material-bootstrap CI (2026-09-29, sta_fix).

WHY: eval_protocol.report() gives each feature set its own CI, and those CIs
overlap heavily at ~50 materials even when one set is better on every repeat.
The question an ablation asks is about the DIFFERENCE, so this scores both
feature sets on identical folds (GroupKFold splits depend only on the groups)
and bootstraps materials on the paired delta. Candidate for folding into
eval_protocol.py (lead-owned). Run this file for a self-check.
"""
import numpy as np

from eval_protocol import repeated_group_kfold


def paired_delta(model, X_base, X_new, y, groups, log_target=False, n_boot=2000, seed=0):
    """Returns (mean per-repeat delta R2, #repeats improved, 95% CI of that mean delta).

    CI: resample materials with replacement; on each resample recompute every repeat's
    pooled-OOF R2 for both feature sets and take the mean paired delta, so the CI is
    centred on the same statistic as the point estimate (material + fold randomness).
    """
    ra, oa = repeated_group_kfold(model, X_base, y, groups, log_target=log_target)
    rb, ob = repeated_group_kfold(model, X_new, y, groups, log_target=log_target)
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    boots = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))])
        yb = y[rows]
        sst = ((yb - yb.mean()) ** 2).sum()
        r2a = 1 - ((oa[:, rows] - yb) ** 2).sum(axis=1) / sst
        r2b = 1 - ((ob[:, rows] - yb) ** 2).sum(axis=1) / sst
        boots.append((r2b - r2a).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (rb - ra).mean(), int((rb > ra).sum()), (lo, hi)


def print_paired(name, *args, **kw):
    d, wins, (lo, hi) = paired_delta(*args, **kw)
    print(f"  PAIRED {name}: delta pooled-OOF R2 {d:+.4f} (improved in {wins}/10 repeats); "
          f"material-bootstrap 95% CI of delta [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 5)
    x = rng.normal(size=40)[groups]
    noise = rng.normal(size=200)
    y = 3 * x + 10
    rf = RandomForestRegressor(n_estimators=30, random_state=0)
    d, wins, (lo, hi) = paired_delta(rf, noise[:, None], np.c_[noise, x], y, groups)
    assert d > 0.5 and wins == 10 and lo > 0 and lo <= d <= hi, (d, wins, lo, hi)  # informative feature wins clearly
    print("eval_paired self-check OK")
