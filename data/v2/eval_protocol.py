#!/usr/bin/env python3
"""Dravix V2 -- shared held-out evaluation protocol (added 2026-09-29).

Every estimator here splits by material group, so a material never spans
train and test.

- grouped_shuffle_r2: the protocol every leaderboard row since exp001 uses
  (GroupShuffleSplit test_size=0.25, seeds 0-9, one R2 per split). Kept so
  new rows stay comparable with old ones.
- repeated_group_kfold: 5-fold GroupKFold repeated with different shuffles.
  Every material is predicted out-of-fold exactly once per repeat, and each
  repeat's R2 is pooled over the whole dataset. That makes it much less
  sensitive to which few materials happen to land in a 25% test split. Use it,
  with bootstrap_ci, for any "beats SOTA" claim.
- bootstrap_ci: 95% CI for R2 from resampling materials (not rows) with
  replacement, so the CI reflects how few materials we have.

Run this file directly for a self-check.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def _fit_predict(model, X_train, y_train, X_test, log_target):
    m = clone(model)
    m.fit(X_train, np.log1p(y_train) if log_target else y_train)
    pred = m.predict(X_test)
    return np.expm1(pred) if log_target else pred


def grouped_shuffle_r2(model, X, y, groups, seeds=range(10), test_size=0.25, log_target=False):
    """Per-seed held-out R2 (in the target's raw units even when log_target=True)."""
    scores = []
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        assert set(groups[tr]).isdisjoint(groups[te])
        scores.append(r2_score(y[te], _fit_predict(model, X[tr], y[tr], X[te], log_target)))
    return np.array(scores)


def repeated_group_kfold(model, X, y, groups, n_splits=5, repeats=10, log_target=False):
    """Returns (pooled out-of-fold R2 per repeat, OOF predictions shaped (repeats, n_rows))."""
    oof = np.empty((repeats, len(y)))
    for r in range(repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for tr, te in gkf.split(X, y, groups):
            assert set(groups[tr]).isdisjoint(groups[te])
            oof[r, te] = _fit_predict(model, X[tr], y[tr], X[te], log_target)
    return np.array([r2_score(y, p) for p in oof]), oof


def bootstrap_ci(y, pred, groups, n_boot=2000, seed=0):
    """95% CI of R2 over material-level bootstrap resamples of one set of OOF predictions."""
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))])
        stats.append(r2_score(y[rows], pred[rows]))
    return np.percentile(stats, [2.5, 97.5])


def report(name, model, X, y, groups, log_target=False, sota=None):
    """Prints both protocols plus the bootstrap CI; returns a dict of the headline numbers."""
    gss = grouped_shuffle_r2(model, X, y, groups, log_target=log_target)
    rkf, oof = repeated_group_kfold(model, X, y, groups, log_target=log_target)
    lo, hi = bootstrap_ci(y, oof[0], groups)
    out = dict(gss_mean=gss.mean(), gss_std=gss.std(), gss_min=gss.min(), gss_max=gss.max(),
               rkf_mean=rkf.mean(), rkf_std=rkf.std(), ci_lo=lo, ci_hi=hi,
               n_rows=len(y), n_materials=len(np.unique(groups)))
    print(f"{name}: {out['n_rows']} rows / {out['n_materials']} materials")
    print(f"  GroupShuffleSplit x10 (leaderboard protocol): mean R2={gss.mean():.4f} std={gss.std():.4f} "
          f"min={gss.min():.4f} max={gss.max():.4f}")
    print(f"  Repeated GroupKFold 5x10 (pooled OOF):        mean R2={rkf.mean():.4f} std={rkf.std():.4f}; "
          f"material-bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    if sota is not None:
        print(f"  vs SOTA {sota}: delta (GKF mean) {rkf.mean() - sota:+.4f}; CI excludes SOTA from above: {lo > sota}")
    return out


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 5)
    x = rng.normal(size=40)[groups] + 0.05 * rng.normal(size=200)
    X, y = x[:, None], 3 * x + 10
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    assert grouped_shuffle_r2(rf, X, y, groups).mean() > 0.9
    rkf, oof = repeated_group_kfold(rf, X, y, groups, repeats=2)
    assert rkf.min() > 0.9 and np.isfinite(oof).all()
    lo, hi = bootstrap_ci(y, oof[0], groups, n_boot=200)
    assert lo <= r2_score(y, oof[0]) <= hi
    print("eval_protocol self-check OK")
