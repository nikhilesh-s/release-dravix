#!/usr/bin/env python3
"""Group-weighted Mondrian CV+ prediction intervals and their coverage metrics (pure numpy, no data access).

Follows experiments/configs/conformal_intervals.json ("method", "metrics", "variants"). Used by
data/v2/conformal_intervals.py, data/v2/conformal_loi_snapshot.py and demo/predict.py.

Weighted CV+ (Barber et al. 2021, with group weights after Dunn, Wasserman & Ramdas 2023): for a test point and
calibration rows i with weights w_i (W = sum w_i), candidate upper values u_i and lower values l_i,
  tau   = (1 - alpha) * (W + 1)
  upper = the smallest u_i whose {j : u_j <= u_i} carries summed weight >= tau   (+inf if W < tau)
  lower = the largest  l_i whose {j : l_j >= l_i} carries summed weight >= tau   (-inf if W < tau)
with a 1e-12 tolerance on the weight sums. With all w_i = 1 this is the textbook CV+ (the ceil((1-alpha)(n+1))-th
smallest u_i and the floor(alpha(n+1))-th smallest l_i); the self-check at the bottom asserts that.

Run this file directly for the self-check.
"""
import numpy as np

ALPHA = 0.10          # fixed by the config; there is no alpha flag anywhere
ANCHORS = (25.0, 50.0, 75.0)
TOL = 1e-12


# ---------------------------------------------------------------- strata and weights
def stratum_of(flux, anchors=ANCHORS):
    """Nearest anchor in kW/m2, ties to the lower anchor (15, 35 -> 25; 100 -> 75)."""
    a = np.asarray(anchors, float)
    f = np.atleast_1d(np.asarray(flux, float))
    idx = np.argmin(np.abs(f[:, None] - a[None, :]), axis=1)  # argmin takes the first (= lower) anchor on a tie
    return a[idx]


def group_weights(groups, strata=None):
    """w_i = 1 / (number of rows of group g(i) in row i's stratum), so each group carries total weight 1 per stratum."""
    groups = np.asarray(groups)
    strata = np.zeros(len(groups)) if strata is None else np.asarray(strata)
    keys = np.array([f"{s!r}\x00{g}" for s, g in zip(strata, groups)])
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


# ---------------------------------------------------------------- weighted CV+ bounds
def weighted_upper(U, w, alpha=ALPHA):
    """U: (n_test, n_cal) candidate upper values; w: (n_cal,) weights. Returns (n_test,) upper bounds."""
    U = np.atleast_2d(np.asarray(U, float))
    w = np.asarray(w, float)
    tau = (1 - alpha) * (w.sum() + 1)
    order = np.argsort(U, axis=1, kind="stable")
    Us = np.take_along_axis(U, order, axis=1)
    cw = np.cumsum(w[order], axis=1)
    ok = cw >= tau - TOL
    pos = np.argmax(ok, axis=1)
    out = Us[np.arange(len(U)), pos]
    return np.where(ok[:, -1], out, np.inf)


def weighted_lower(L, w, alpha=ALPHA):
    """L: (n_test, n_cal) candidate lower values; w: (n_cal,) weights. Returns (n_test,) lower bounds."""
    L = np.atleast_2d(np.asarray(L, float))
    w = np.asarray(w, float)
    tau = (1 - alpha) * (w.sum() + 1)
    order = np.argsort(-L, axis=1, kind="stable")
    Ls = np.take_along_axis(L, order, axis=1)
    cw = np.cumsum(w[order], axis=1)
    ok = cw >= tau - TOL
    pos = np.argmax(ok, axis=1)
    out = Ls[np.arange(len(L)), pos]
    return np.where(ok[:, -1], out, -np.inf)


def cvplus(center_test, cal_fold, R, w, scale_test=None, alpha=ALPHA):
    """P / V1 / V2 form (and V3 with scale_test): center_test (n_folds, n_test) = mu_k(x) for each fold model k;
    cal_fold (n_cal,) = fold k(i) of calibration row i; R (n_cal,) = its score; w (n_cal,) = its weight.
    u_i = mu_{k(i)}(x) + R_i * scale_{k(i)}(x), l_i = mu_{k(i)}(x) - R_i * scale_{k(i)}(x) (scale = 1 unless V3)."""
    C = np.asarray(center_test, float)[np.asarray(cal_fold)].T          # (n_test, n_cal)
    S = 1.0 if scale_test is None else np.asarray(scale_test, float)[np.asarray(cal_fold)].T
    R = np.asarray(R, float)[None, :]
    return weighted_lower(C - R * S, w, alpha), weighted_upper(C + R * S, w, alpha)


def cqr(qlo_test, qhi_test, cal_fold, E, w, alpha=ALPHA):
    """CQR form: u_i = qhi_{k(i)}(x) + E_i, l_i = qlo_{k(i)}(x) - E_i; qlo/qhi_test are (n_folds, n_test)."""
    f = np.asarray(cal_fold)
    E = np.asarray(E, float)[None, :]
    return (weighted_lower(np.asarray(qlo_test, float)[f].T - E, w, alpha),
            weighted_upper(np.asarray(qhi_test, float)[f].T + E, w, alpha))


def heldout_halfwidth(R, w, alpha=ALPHA):
    """Demo in-data path: the smallest R_i with summed weight of {j : R_j <= R_i} >= (1 - alpha) * (W + 1); inf if none."""
    return float(weighted_upper(np.asarray(R, float)[None, :], w, alpha)[0])


# ---------------------------------------------------------------- metrics
def _per_group(values, groups, fn):
    """fn over each group's values; groups returned sorted (np.unique order)."""
    ug, inv = np.unique(groups, return_inverse=True)
    return ug, np.array([fn(values[inv == j]) for j in range(len(ug))])


def coverage_metrics(cov, groups):
    """cov: (n,) 0/1 indicator for one repeat on one cell's rows. Returns group_all, group_mean, row."""
    cov = np.asarray(cov, float)
    _, gall = _per_group(cov, groups, np.min)
    _, gmean = _per_group(cov, groups, np.mean)
    return dict(group_all=float(gall.mean()), group_mean=float(gmean.mean()), row=float(cov.mean()))


def width_metrics(lo, hi, back, clip, log_scale):
    """Widths for one repeat on one cell's rows. lo/hi on the score scale (unclipped). Bounded intervals only;
    an empty interval (lo > hi, CQR only) has width 0. Raw widths = clip(back(hi)) - clip(back(lo))."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    bounded = np.isfinite(lo) & np.isfinite(hi)
    empty = bounded & (lo > hi)
    out = dict(n_unbounded=int((~bounded).sum()), n_empty=int(empty.sum()), n_bounded=int(bounded.sum()))
    if not bounded.any():
        out.update(score_width_mean=None, score_width_median=None, raw_width_mean=None, raw_width_median=None,
                   ratio_median=None)
        return out
    sw = np.where(empty, 0.0, hi - lo)[bounded]
    rl = np.clip(back(lo[bounded]), clip[0], clip[1])
    rh = np.clip(back(hi[bounded]), clip[0], clip[1])
    rw = np.where(empty[bounded], 0.0, rh - rl)
    out.update(score_width_mean=float(sw.mean()), score_width_median=float(np.median(sw)),
               raw_width_mean=float(rw.mean()), raw_width_median=float(np.median(rw)),
               ratio_median=float(np.median(np.exp(sw))) if log_scale else None)
    return out


def bootstrap_ci_groups(group_values, n_boot=2000, seed=0):
    """95% CI of the mean of per-group values, resampling groups as eval_protocol.bootstrap_ci does
    (fresh np.random.default_rng(seed), rng.choice(unique_groups, G))."""
    ug = np.array(sorted(group_values))
    v = np.array([group_values[g] for g in ug], float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.searchsorted(ug, rng.choice(ug, len(ug)))
        stats[b] = v[pick].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_rows(row_values, groups, n_boot=2000, seed=0):
    """95% CI of the pooled row mean, resampling groups (all of a drawn group's rows enter, as bootstrap_ci)."""
    ug, inv = np.unique(groups, return_inverse=True)
    S = np.bincount(inv, weights=np.asarray(row_values, float), minlength=len(ug))
    N = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.searchsorted(ug, rng.choice(ug, len(ug)))
        stats[b] = S[pick].sum() / N[pick].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def reading_rule(ci, nominal=1 - ALPHA):
    """Fixed report-only reading of group_mean_coverage's 95% CI against the nominal 0.90."""
    lo, hi = ci
    if hi < nominal:
        return "under-covers"
    if lo > nominal:
        return "conservative"
    return "consistent with nominal"


def cell_summary(cov, lo, hi, groups, back, clip, log_scale):
    """All metrics for one cell (overall or one stratum). cov, lo, hi: (repeats, n_cell). Per-repeat metrics,
    mean / min / max over repeats, group-bootstrap CIs on repeat-averaged per-group values, reading rule."""
    cov = np.asarray(cov, float)
    groups = np.asarray(groups)
    reps = [dict(**coverage_metrics(cov[r], groups), **width_metrics(lo[r], hi[r], back, clip, log_scale))
            for r in range(cov.shape[0])]
    out = {}
    for k in ("group_all", "group_mean", "row", "score_width_mean", "score_width_median", "raw_width_mean",
              "raw_width_median", "ratio_median"):
        vals = [d[k] for d in reps if d[k] is not None]
        out[k] = (dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
                  if vals else None)
    for k in ("n_unbounded", "n_empty", "n_bounded"):
        out[k] = dict(total_over_repeats=int(sum(d[k] for d in reps)), per_repeat_max=int(max(d[k] for d in reps)))
    # per-group values averaged over repeats
    ug, inv = np.unique(groups, return_inverse=True)
    gmean_r = np.array([[cov[r, inv == j].mean() for j in range(len(ug))] for r in range(cov.shape[0])])
    gall_r = np.array([[cov[r, inv == j].min() for j in range(len(ug))] for r in range(cov.shape[0])])
    out["group_mean"]["ci"] = bootstrap_ci_groups(dict(zip(ug, gmean_r.mean(axis=0))))
    out["group_all"]["ci"] = bootstrap_ci_groups(dict(zip(ug, gall_r.mean(axis=0))))
    out["row"]["ci"] = bootstrap_ci_rows(cov.mean(axis=0), groups)
    out["group_mean"]["reading"] = reading_rule(out["group_mean"]["ci"])
    multi = bool((np.bincount(inv) > 1).any())
    out["group_all"]["note"] = ("expected to fall below 0.90: the interval is per-row, not simultaneous, and groups "
                                "have more than one row in this cell; no reading-rule label"
                                if multi else "every group has one row in this cell, so group_all equals group_mean")
    out["n_rows"] = int(len(groups))
    out["n_groups"] = int(len(ug))
    out["binomial_sd"] = float(np.sqrt((1 - ALPHA) * ALPHA / len(ug)))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # 1) unit weights reproduce textbook CV+ order statistics
    for n in (9, 10, 19, 37, 100):
        U = rng.normal(size=(4, n))
        L = rng.normal(size=(4, n))
        k_hi = int(np.ceil((1 - ALPHA) * (n + 1) - 1e-9))
        k_lo = int(np.floor(ALPHA * (n + 1) + 1e-9))
        up = weighted_upper(U, np.ones(n))
        lo = weighted_lower(L, np.ones(n))
        exp_up = np.sort(U, axis=1)[:, k_hi - 1] if k_hi <= n else np.full(4, np.inf)
        exp_lo = np.sort(L, axis=1)[:, k_lo - 1] if k_lo >= 1 else np.full(4, -np.inf)
        assert np.allclose(up, exp_up) and np.allclose(lo, exp_lo), n
    assert np.isinf(weighted_upper(np.zeros((1, 8)), np.ones(8))).all()      # 8 < 0.9 * 9 -> unbounded
    assert np.isfinite(weighted_upper(np.zeros((1, 9)), np.ones(9))).all()   # 9 >= 0.9 * 10
    # 2) group weights: each group carries total weight 1 per stratum
    g = np.array(list("aabbbc"))
    s = np.array([25, 50, 25, 25, 25, 75])
    w = group_weights(g, s)
    assert np.allclose(w, [1, 1, 1 / 3, 1 / 3, 1 / 3, 1]) and np.isclose(w.sum(), 4)  # 2 groups at 25, 1 at 50, 1 at 75
    # 3) strata: nearest anchor, ties low
    assert list(stratum_of([15, 25, 35, 37.5, 50, 62.5, 75, 100])) == [25, 25, 25, 25, 50, 50, 75, 75]
    # 4) exchangeable groups -> group-weighted CV+ covers about 90% of a new group's rows
    G, cov = 60, []
    for rep in range(1000):
        off = rng.normal(size=G + 1)
        rows = rng.integers(1, 6, size=G + 1)
        gg = np.repeat(np.arange(G + 1), rows)
        yy = off[gg] + 0.3 * rng.normal(size=len(gg))
        cal, te = gg < G, gg == G
        lo_, hi_ = cvplus(np.zeros((1, te.sum())), np.zeros(cal.sum(), int), np.abs(yy[cal]), group_weights(gg[cal]))
        cov.append(((lo_ <= yy[te]) & (yy[te] <= hi_)).mean())
    assert 0.87 < np.mean(cov) < 0.94, np.mean(cov)
    print("conformal self-check OK (mean simulated group coverage %.3f)" % np.mean(cov))
