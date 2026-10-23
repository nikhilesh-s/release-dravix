#!/usr/bin/env python3
"""Round 10 (physics-anchored candidates) -- experiments/configs/round10_physics.json (committed 284ed43), followed literally.

  python data/v2/train_round10_physics.py --target SELFTEST          # label-free self-tests only
  python data/v2/train_round10_physics.py --target pHRR|FIGRA|CR|LOI

Per target: label-free self-tests, numbers_of_record check, step 1 = incumbent reproduced within 0.001 of its record
(mean, ci_lo, ci_hi; exit 2 = BLOCKED), then every candidate in config order on identical folds, result JSONs
(experiments/results/<id>.json), experiments/runs/round10_<t>_oof.npz and adoption gate v2 (data/v2/adoption_gate.py)
-> experiments/runs/round10_<t>_gate_v2.json. Mirrors data/v2/train_round9_tabpfn.py.
LOI: snapshot 3 is built in a scratch copy exactly as rescore_loi_snapshot.py does (wave4 + wave5 staging, its
patch_build and patch_round6), and train_loi_rescore_r8.py lines 9-637 (data prep, exp089/exp092 checks, round-8 model
definitions) run in-process from that copy; exp115 = Avg(rf, et, ridge) on DATA['c2'][0].
"""
import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_k] = "3"  # worker limit; n_jobs=-1 forests resolve to 3 threads (forest output does not depend on n_jobs)

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS, RESULTS = ROOT / "experiments" / "runs", ROOT / "experiments" / "results"
sys.path.insert(0, str(HERE))
import adoption_gate  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402
from train_blend_round1 import Avg, data_cr, data_figra, data_phrr, et, ridge  # noqa: E402

CONFIG = "experiments/configs/round10_physics.json"
CODE = "data/v2/train_round10_physics.py"
FLAGS = RUNS / "round10_flags.md"
TOL = 0.001
# Apple Accelerate raises spurious FP flags inside matmul (finite outputs; every OOF is checked with isfinite below)
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
GP_WARN = collections.Counter()


# ------------------------------------------------------------------ physics features ----------------------------------
def fgc_K(Q_Jg, T1, T2, T0=298.0):
    """Fire growth capacity, FAA DOT/FAA/TC-20/35 Eq. 9: Q/(T2-T1) * (T2-T0)/(T1-T0) [J/g-K]; NaN unless T2 > T1."""
    Q, T1, T2 = (np.asarray(v, float) for v in (Q_Jg, T1, T2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = Q / (T2 - T1) * (T2 - T0) / (T1 - T0)
    return np.where(T2 > T1, out, np.nan)  # NaN comparisons are False -> NaN when T1 or T2 is missing


def mcc_physics():
    """Per-material means over mcc_extras_fsri.csv rows (the aggregation exp074 uses for mcc_t_onset_C under --bc)."""
    x = pd.read_csv(HERE / "mcc_extras_fsri.csv")
    a = x.groupby("material").agg(t_onset_bc=("t_onset_bc_C", "mean"), mcc_t_end_C=("t_end_bc_C", "mean"),
                                  q_bc=("total_hr_bc_kJg", "mean"))
    a["mcc_fgc_JgK"] = fgc_K(np.maximum(a.q_bc, 0) * 1000, a.t_onset_bc + 273.15, a.mcc_t_end_C + 273.15)
    return a


# ------------------------------------------------------------------ candidate estimators ------------------------------
class LnOffset(BaseEstimator, RegressorMixin):
    """o1: fit `est` on r = ln(y) - ln(X[:, col]); predict exp(m(x) + ln(X[:, col])). Coefficient fixed at 1."""

    def __init__(self, est, col):
        self.est, self.col = est, col

    def fit(self, X, y):
        self.m_ = clone(self.est).fit(X, np.log(y) - np.log(X[:, self.col]))
        return self

    def predict(self, X):
        return np.exp(self.m_.predict(X) + np.log(X[:, self.col]))


class GPOffset(BaseEstimator, RegressorMixin):
    """g1: physics-mean GP. t = ln(y) if log else y; base b = ln(X[:, cols[0]]) if log else row-wise nanmean(X[:, cols]),
    all-NaN base -> in-fold training median of y. GP on r = t - b, inputs SimpleImputer(median, indicators, keep_empty)
    -> StandardScaler; kernel C(1)*RBF(ones(d)) + White(1), d = imputer output width (fold-dependent: one indicator per
    feature with a missing value in the training fold); normalize_y, n_restarts 0, random_state 0. predict exp/id(b + mu)."""

    def __init__(self, cols, log):
        self.cols, self.log = cols, log

    def _b(self, X):
        if self.log:
            return np.log(X[:, self.cols[0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, handled by the fallback
            b = np.nanmean(X[:, list(self.cols)], axis=1)
        return np.where(np.isnan(b), self.med_, b)

    def fit(self, X, y):
        t = np.log(y) if self.log else np.asarray(y, float)
        self.med_ = float(np.median(y))
        self.pre_ = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                  StandardScaler()).fit(X)
        Z = self.pre_.transform(X)
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Z.shape[1])) + WhiteKernel(1.0)
        self.gp_ = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.gp_.fit(Z, t - self._b(X))
        for m in w:
            GP_WARN[m.category.__name__] += 1
        return self

    def t_predict(self, X):
        return self._b(X) + self.gp_.predict(self.pre_.transform(X))

    def predict(self, X):
        return np.exp(self.t_predict(X)) if self.log else self.t_predict(X)


class LnAvg(BaseEstimator, RegressorMixin):
    """g2: mean on the ln scale of exp074's prediction (ln of clip(expm1(hgb on log1p y), 1e-6)) and g1's ln prediction."""

    def __init__(self, hgb, gp):
        self.hgb, self.gp = hgb, gp

    def fit(self, X, y):
        self.h_ = clone(self.hgb).fit(X, np.log1p(y))  # identical to exp074's fold model
        self.g_ = clone(self.gp).fit(X, y)
        return self

    def predict(self, X):
        a = np.log(np.clip(np.expm1(self.h_.predict(X)), 1e-6, None))
        return np.exp(0.5 * (a + self.g_.t_predict(X)))


class MonoImp(BaseEstimator, RegressorMixin):
    """LOI m1 member: make_pipeline(SimpleImputer(median, indicators, keep_empty), est + monotonic_cst c), c = +1 on the
    `up` columns, 0 on every other column incl. indicator columns. c is sized at fit time (indicator count is per fold)."""

    def __init__(self, est, up):
        self.est, self.up = est, up

    def fit(self, X, y):
        self.imp_ = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(X)
        Z = self.imp_.transform(X)
        c = np.zeros(Z.shape[1], dtype=int)
        c[list(self.up)] = 1
        self.m_ = clone(self.est).set_params(monotonic_cst=c).fit(Z, y)
        return self

    def predict(self, X):
        return self.m_.predict(self.imp_.transform(X))


class StrEnc(BaseEstimator, RegressorMixin):
    """LOI s1: X's last column is the FR-name string; skrub StringEncoder(n_components=8, random_state=0) is fit on the
    training rows only and its 8 columns are appended to the numeric columns; `est` fits on the result."""

    def __init__(self, est):
        self.est = est

    def _num(self, X):
        return X[:, :-1].astype(float)

    def fit(self, X, y):
        from skrub import StringEncoder
        self.se_ = StringEncoder(n_components=8, random_state=0)
        E = self.se_.fit_transform(pd.Series(X[:, -1].astype(str), name="fr_names")).to_numpy(float)
        self.m_ = clone(self.est).fit(np.column_stack([self._num(X), E]), y)
        return self

    def predict(self, X):
        E = self.se_.transform(pd.Series(X[:, -1].astype(str), name="fr_names")).to_numpy(float)
        return self.m_.predict(np.column_stack([self._num(X), E]))


# ------------------------------------------------------------------ self-tests (label-free) ---------------------------
def self_test():
    v = float(fgc_K(29.3e3, 695.0, 837.0))
    assert abs(v - 280) <= 2, v
    assert np.isnan(fgc_K(1e4, 700.0, 690.0)) and np.isnan(fgc_K(1e4, 700.0, np.nan))
    rng = np.random.default_rng(0)
    n, d = 120, 6
    X = rng.normal(size=(n, d))
    X[:, 1] = np.exp(rng.normal(size=n))  # positive "peak"
    X[rng.random((n, d)) < 0.1] = np.nan
    X[:, 1] = np.where(np.isnan(X[:, 1]), 1.0, X[:, 1])
    X[:5, 3:6] = np.nan  # all-NaN base rows exercise the median fallback
    y = np.exp(1 + 0.5 * np.nan_to_num(X[:, 0]) + 0.2 * rng.normal(size=n))
    tr, te = np.arange(90), np.arange(90, n)
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=50, random_state=0)
    for name, m in [("LnOffset", LnOffset(hgb, 1)), ("GPOffset-ln", GPOffset((1,), True)),
                    ("GPOffset-raw", GPOffset((3, 4, 5), False)), ("LnAvg", LnAvg(hgb, GPOffset((1,), True)))]:
        p = clone(m).fit(X[tr], y[tr]).predict(X[te])
        assert p.shape == (len(te),) and np.isfinite(p).all(), name
    m = LnOffset(hgb, 1).fit(X[tr], y[tr])
    assert np.allclose(m.predict(X[te]), np.exp(m.m_.predict(X[te]) + np.log(X[te, 1])))
    g = GPOffset((3, 4, 5), False).fit(X[:20], y[:20])
    assert np.allclose(g._b(X[:5]), np.median(y[:20]))
    # MonoImp with no constrained column == make_pipeline(imputer, est); constrained version predicts finite values
    for est in (RandomForestRegressor(n_estimators=30, min_samples_leaf=2, random_state=0),
                ExtraTreesRegressor(n_estimators=30, min_samples_leaf=2, random_state=0)):
        a = MonoImp(est, ()).fit(X[tr], y[tr]).predict(X[te])
        b = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          clone(est)).fit(X[tr], y[tr]).predict(X[te])
        assert np.array_equal(a, b), type(est).__name__
        mm = MonoImp(est, (0,)).fit(X[tr], y[tr])
        assert mm.m_.monotonic_cst.shape[0] == mm.imp_.transform(X[:1]).shape[1] and np.isfinite(mm.predict(X[te])).all()
        grid = np.tile(np.nanmedian(X, axis=0), (25, 1))
        grid[:, 0] = np.linspace(-2, 2, 25)
        assert np.all(np.diff(mm.predict(grid)) >= -1e-12), "monotone +1 violated"
    Xs = np.column_stack([X.astype(object), np.array([f"fr name {i % 7} phosphate" for i in range(n)], dtype=object)])
    p = StrEnc(ExtraTreesRegressor(n_estimators=30, random_state=0)).fit(Xs[tr], y[tr]).predict(Xs[te])
    assert p.shape == (len(te),) and np.isfinite(p).all()
    print(f"round10 self-test OK (FGC PC/ABS example {v:.1f} J/g-K; wrappers: shapes, finite, offset identity, "
          f"median fallback, MonoImp == make_pipeline when unconstrained, monotone +1 holds, StrEnc in-fold)")


# ------------------------------------------------------------------ helpers -------------------------------------------
def topk(y, g, oof_inc, oof_c):
    """reported_only top-k removal; same construction as train_round9_tabpfn.topk. Diagnostic only."""
    row = ((y - oof_inc) ** 2 - (y - oof_c) ** 2).mean(axis=0)
    con = {str(m): float(row[g == m].sum()) for m in np.unique(g)}
    ranked = sorted(con, key=lambda m: (-con[m], m))
    tot = sum(con.values())
    out = dict(top5=[[m, con[m]] for m in ranked[:5]], top5_share_of_net=sum(con[m] for m in ranked[:5]) / tot if tot else None)
    for k in (1, 3, 5):
        keep = ~np.isin(np.asarray(g).astype(str), ranked[:k])
        d = np.array([r2_score(y[keep], oof_c[r, keep]) - r2_score(y[keep], oof_inc[r, keep]) for r in range(10)])
        out[f"drop_top{k}"] = dict(mean_gain=float(d.mean()), wins=int((d > 0).sum()))
    out["not_established"] = bool(out["drop_top5"]["mean_gain"] <= 0 or out["drop_top5"]["wins"] < 8)
    return out


def score(model, X, y, g, logt, pooled=None, extra_groups=None):
    t = time.time()
    GP_WARN.clear()
    rk, oof = repeated_group_kfold(model, X, y, g, log_target=logt)
    if not np.isfinite(oof).all():
        raise FloatingPointError(f"non-finite OOF predictions ({int((~np.isfinite(oof)).sum())})")
    lo, hi = bootstrap_ci(y, oof[0], g)
    plo, phi = (pooled or (lambda yy, o, gg: bootstrap_ci(yy, o.mean(axis=0), gg)))(y, oof, g)
    r = dict(rk=rk, oof=oof, lo=float(lo), hi=float(hi), plo=float(plo), phi=float(phi),
             r2_0=float(r2_score(y, oof[0])), mae_0=float(mean_absolute_error(y, oof[0])),
             rmse_0=float(np.sqrt(mean_squared_error(y, oof[0]))))
    if extra_groups is not None:  # pHRR category-GKF / LOI protocol C (report-only)
        ek, eoof = repeated_group_kfold(model, X, y, extra_groups, log_target=logt)
        assert np.isfinite(eoof).all()
        elo, ehi = bootstrap_ci(y, eoof[0], g)
        r.update(ex=float(ek.mean()), ex_std=float(ek.std()), ex_lo=float(elo), ex_hi=float(ehi), ex_rk=[float(v) for v in ek])
    r["gp_warnings"] = dict(GP_WARN)
    r["wall"] = time.time() - t
    return r


def line(cid, r, inc=None, exname=""):
    s = (f"{cid:14s} GKF {r['rk'].mean():.4f} +- {r['rk'].std():.4f} CI [{r['lo']:.4f}, {r['hi']:.4f}] "
         f"pooled [{r['plo']:.4f}, {r['phi']:.4f}]  seed0 R2 {r['r2_0']:.4f} MAE {r['mae_0']:.4g} RMSE {r['rmse_0']:.4g}")
    if inc is not None:
        d = r["rk"] - inc["rk"]
        s += f"  d {d.mean():+.4f} wins {(d > 0).sum()}/10"
    if "ex" in r:
        s += f"  {exname} {r['ex']:.4f} [{r['ex_lo']:.3f}, {r['ex_hi']:.3f}]"
    if r.get("gp_warnings"):
        s += f"  gp_warnings {r['gp_warnings']}"
    return s + f"  ({r['wall']:.0f}s)"


def flag(text):
    new = not FLAGS.exists()
    with open(FLAGS, "a") as f:
        if new:
            f.write("# Round 10 (round10_physics) -- worker flags\n\nItems marked FLAG skipped a candidate or need an "
                    "orchestrator decision; NOTE items record how a definition was implemented, for the critic.\n\n")
        f.write(f"- {text}\n")
    print(text)


def check_record(name, exp):
    nor = json.load(open(ROOT / "experiments" / "numbers_of_record.json"))["targets"][name]["exp_id"]
    print(f"numbers_of_record.json {name}: {nor}; config incumbent {exp}")
    if nor != exp:
        print(f"STOP: incumbent changed to {nor}; the config's rule says score against the new record -- not implemented here")
        sys.exit(2)


# ------------------------------------------------------------------ per-target specs ----------------------------------
def spec_phrr():
    import train_phrr_v3_robust as T
    D = data_phrr()  # applies --bc EXTRAS exactly as train_phrr_promote_c2.py
    full = T.load(replicate_target=True)
    df = full[full["has_mcc"]].reset_index(drop=True)
    F10 = D["feats"]
    assert np.array_equal(df[F10].to_numpy(float), D["X"], equal_nan=True) and np.array_equal(df[T.TARGET].to_numpy(float), D["y"])
    P = mcc_physics()
    df = df.join(P, on="material")
    assert np.allclose(df["t_onset_bc"], df["mcc_t_onset_C"], equal_nan=True), "T1 aggregation differs from exp074's"
    F12 = F10 + ["mcc_t_end_C", "mcc_fgc_JgK"]
    X10, X12, y, g = D["X"], df[F12].to_numpy(float), D["y"], D["g"]
    ip = F10.index("mcc_peak_hrr_Wg")
    assert np.isfinite(X10[:, ip]).all() and (X10[:, ip] > 0).all(), "missing / non-positive mcc_peak_hrr_Wg"
    assert X10.shape == (191, 10) and X12.shape == (191, 12)
    print(f"new features (label-free): mcc_t_end_C NaN {int(np.isnan(X12[:, 10]).sum())}/191, range "
          f"[{np.nanmin(X12[:, 10]):.1f}, {np.nanmax(X12[:, 10]):.1f}] C; mcc_fgc_JgK NaN {int(np.isnan(X12[:, 11]).sum())}/191, "
          f"range [{np.nanmin(X12[:, 11]):.1f}, {np.nanmax(X12[:, 11]):.1f}] J/g-K; peak min {X10[:, ip].min():.2f} W/g")
    hgb = T.hgb()
    hp = "hgb(): learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, random_state=0"
    gpd = ("GaussianProcessRegressor(C(1.0)*RBF(ones(d))+WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=0, "
           "random_state=0) on r = ln(y) - ln(mcc_peak_hrr_Wg); inputs SimpleImputer(median, add_indicator, keep_empty) -> "
           "StandardScaler; d = imputer output width")
    cands = [
        ("r10_phrr_o1", True, LnOffset(clone(hgb), ip), X10, False, F10,
         "HRC-proportional offset: exp074 HGB on r = ln(y) - ln(mcc_peak_hrr_Wg); predict exp(m + b); coefficient 1", hp),
        ("r10_phrr_g2", True, LnAvg(clone(hgb), GPOffset((ip,), True)), X10, False, F10,
         "mean on ln scale of exp074 (ln of clip(expm1, 1e-6)) and g1 (b + mu_GP), then exp", hp + "; " + gpd),
        ("r10_phrr_g1", False, GPOffset((ip,), True), X10, False, F10, "physics-mean GP alone: exp(ln(peak) + mu_GP)", gpd),
        ("r10_phrr_f1", True, clone(hgb), X12, True, F12,
         "exp074 recipe (HGB, log1p) + mcc_t_end_C (mean t_end_bc_C) + mcc_fgc_JgK (FAA TC-20/35 Eq. 9)", hp),
    ]
    return dict(name="pHRR", slug="phrr", exp="exp074", rec=(0.6113, 0.5155, 0.8376), X=X10, y=y, g=g, logt=True,
                inc=hgb, cands=cands, extra=("category-GKF", df["category"].to_numpy()), target=D["target"],
                dataset=D["dataset"].rsplit(", as ", 1)[0], sota=D["sota"], sota_src=D["sota_src"], protocol="gkf_group5x10",
                rec_extra=0.3917)


def spec_figra():
    D = data_figra()
    # rebuild exp119's frame (same code as train_blend_round1.data_figra) to append the physics columns by material
    reps = pd.read_csv(HERE / "figra_fsri.csv")
    reps = reps[reps.spark_ignitor & reps.ignition_recorded]
    df = (reps.groupby(["material", "heat_flux_kWm2"]).agg(figra_kWm2s=("figra_kWm2s", "mean")).reset_index())
    mcc = pd.read_csv(HERE / "mcc_hrr.csv").groupby("material").agg(mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"))
    d = df.join(mcc, on="material")
    d = d[d.mcc_peak_hrr_Wg.notna()].reset_index(drop=True).join(mcc_physics(), on="material")
    assert np.array_equal(d.figra_kWm2s.to_numpy(float), D["y"]) and np.array_equal(d.material.to_numpy(), D["g"])
    d = d.rename(columns={"t_onset_bc": "mcc_t_onset_C"})
    F9 = D["feats"]
    assert "mcc_t_onset_C" not in F9
    new = ["mcc_t_onset_C", "mcc_t_end_C", "mcc_fgc_JgK"]
    X12 = np.column_stack([D["X"], d[new].to_numpy(float)])
    assert D["X"].shape[1] == 9 and X12.shape == (len(D["y"]), 12)
    print("new features (label-free): " + "; ".join(f"{c} NaN {int(np.isnan(X12[:, 9 + i]).sum())}/{len(X12)} range "
                                                    f"[{np.nanmin(X12[:, 9 + i]):.1f}, {np.nanmax(X12[:, 9 + i]):.1f}]"
                                                    for i, c in enumerate(new)))
    inc = Avg([D["inc"], et(), ridge()])
    hp = ("HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=0), ExtraTrees(500, "
          "min_samples_leaf=2), Ridge(alpha=10) with impute+indicators+scaler; equal average in the log1p wrapper (exp119)")
    cands = [("r10_figra_g1", True, clone(inc), X12, True, F9 + new,
              "exp119 recipe on exp119's 9 features + mcc_t_onset_C (mean t_onset_bc_C), mcc_t_end_C, mcc_fgc_JgK", hp)]
    return dict(name="FIGRA", slug="figra", exp="exp119", rec=(0.4975, 0.3464, 0.6475), X=D["X"], y=D["y"], g=D["g"],
                logt=True, inc=inc, cands=cands, extra=None, target=D["target"], dataset=D["dataset"].rsplit(", as ", 1)[0],
                sota=D["sota"], sota_src=D["sota_src"], protocol="gkf_group5x10")


def spec_cr():
    D = data_cr()
    F = D["feats"]
    sta = [F.index(c) for c in ("sta_res_3K", "sta_res_10K", "sta_res_30K")]
    assert sta == [3, 4, 5] and D["X"].shape[1] == 11
    print(f"CR rows with all three STA residues NaN (median fallback): {int(np.isnan(D['X'][:, sta]).all(axis=1).sum())}")
    hgb = D["inc"]
    cst = [1 if c in ("sta_res_3K", "sta_res_10K", "sta_res_30K", "mcc_char_yield_pct") else 0 for c in F]
    assert sum(cst) == 4
    hp = ("HGB(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0), ExtraTrees(500, "
          "min_samples_leaf=2), Ridge(alpha=10) with impute+indicators+scaler (exp117), raw %")
    gpd = ("GaussianProcessRegressor(C(1.0)*RBF(ones(d))+WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=0, "
           "random_state=0) on r = y - nanmean(sta_res_3K, 10K, 30K) (all-NaN -> in-fold median y); inputs "
           "SimpleImputer(median, add_indicator, keep_empty) -> StandardScaler; d = imputer output width; no clipping")
    cands = [
        ("r10_cr_g2", True, Avg([clone(hgb), et(), ridge(), GPOffset(tuple(sta), False)]), D["X"], False, F,
         "equal average of exp117's HGB, ET, Ridge and g1 (4 x 1/4), no clipping", hp + "; " + gpd),
        ("r10_cr_g1", False, GPOffset(tuple(sta), False), D["X"], False, F, "physics-mean GP alone: base + mu_GP", gpd),
        ("r10_cr_m1", True, Avg([clone(hgb).set_params(monotonic_cst=cst), et(), ridge()]), D["X"], False, F,
         "exp117 blend with HGB monotonic_cst +1 on sta_res_3K/10K/30K and mcc_char_yield_pct", hp + f"; HGB monotonic_cst={cst}"),
    ]
    return dict(name="Char Residue", slug="cr", exp="exp117", rec=(0.6504, 0.4389, 0.7893), X=D["X"], y=D["y"], g=D["g"],
                logt=False, inc=Avg([hgb, et(), ridge()]), cands=cands, extra=None, target=D["target"],
                dataset=D["dataset"].rsplit(", as ", 1)[0], sota=D["sota"], sota_src=D["sota_src"], protocol="gkf_group5x10")


def spec_loi(work):
    """Build snapshot 3 in `work` as rescore_loi_snapshot.main does, then run train_loi_rescore_r8.py lines 9-637 there."""
    import rescore_loi_snapshot as RS
    RS.copy_tree(work)
    for st in ("data/v2/loi_rpuf_parts/wave4_staging", "data/v2/loi_rpuf_parts/wave5_staging"):
        RS.apply_stage(work, (ROOT / st).resolve())
    RS.patch_build(work)
    subprocess.run([sys.executable, "data/v2/build_loi_rpuf.py"], cwd=work, check=True)
    d = pd.read_csv(work / "data/v2/loi_rpuf_literature.csv")
    bm = d.loi_pct.notna() & (d.extraction != "figure")
    n_rows, n_papers = int(bm.sum()), int(d[bm].ref_id.nunique())
    print(f"snapshot: {n_rows} benchmark rows / {n_papers} papers")
    src = (HERE / "train_loi_rescore_r8.py").read_text().splitlines(keepends=True)
    assert src[634].startswith("                         StandardScaler(), Ridge(alpha=10.0))") and src[8].startswith("import glob")
    assert not "".join(src[635:637]).strip() and src[637].startswith("res = {}")
    pre = work / "data/v2/r10_loi_prefix.py"
    pre.write_text("".join(src[8:637]))  # lines 9-637 verbatim
    RS.patch_round6(work, n_rows, n_papers, script="data/v2/r10_loi_prefix.py")
    assert "build_loi_rpuf" not in sys.modules
    ns = runpy.run_path(str(pre), run_name="r10_loi_prefix")
    assert Path(sys.modules["build_loi_rpuf"].__file__).resolve().is_relative_to(work.resolve()), "unpatched build_loi_rpuf"
    X2, y, g, dom, F = ns["DATA"]["c2"][0], ns["y"], ns["g"], ns["dom"], list(ns["FEAT_C2"])
    assert (len(y), len(np.unique(g)), X2.shape[1]) == (606, 107, 51)
    df, bmask = ns["df"], ns["bm"]
    assert np.array_equal(df.loc[bmask, "loi_pct"].to_numpy(float), y) and np.array_equal(df.loc[bmask, "ref_id"].to_numpy(), g)
    frn = df.loc[bmask, "fr_names"].fillna("none").astype(str).to_numpy(dtype=object)
    print(f"fr_names: {int((frn == 'none').sum())} rows 'none', {len(set(frn))} distinct strings")
    rf, et_l, ridge_l, AvgL = ns["rf"], ns["et"], ns["ridge"], ns["Avg"]
    up = (F.index("total_fr_wtpct"), F.index("el_P"))
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)  # noqa: E731
    inc = AvgL([rf(), et_l(), ridge_l()])
    hp = ("RF(500, min_samples_leaf=2, random_state=0), ExtraTrees(500, min_samples_leaf=2, random_state=0), Ridge(alpha=10) "
          "with impute+indicators+scaler; equal average (exp115)")
    Xs = np.column_stack([X2.astype(object), frn])
    cands = [
        ("r10_loi_m1", True, AvgL([MonoImp(rf(), up), MonoImp(et_l(), up), ridge_l()]), X2, False, F,
         "Avg(RF_m, ET_m, Ridge): RF/ET behind SimpleImputer(median, indicators, keep_empty) with monotonic_cst +1 on "
         "total_fr_wtpct and el_P, 0 elsewhere incl. indicators", hp + f"; monotone columns {up}"),
        ("r10_loi_m0", False, AvgL([make_pipeline(imp(), rf()), make_pipeline(imp(), et_l()), ridge_l()]), X2, False, F,
         "m1 without monotonic_cst (RF/ET behind the same imputer)", hp),
        ("r10_loi_s1", True, StrEnc(AvgL([rf(), et_l(), ridge_l()])), Xs, False, F + [f"fr_names_se{i}" for i in range(8)],
         "exp115 recipe + 8 skrub StringEncoder(n_components=8, random_state=0) columns of fr_names (NaN -> 'none'), "
         "fit inside each training fold", hp + "; skrub 0.10.1 StringEncoder defaults otherwise"),
    ]
    return dict(name="LOI", slug="loi", exp="exp115", rec=(0.3875, 0.3124, 0.5289), X=X2, y=y, g=g, logt=False, inc=inc,
                cands=cands, extra=("protocol C", dom), target="LOI (%) RPUF literature",
                dataset=f"LOI snapshot 3 ({n_rows} rows / {n_papers} papers) built as rescore_loi_snapshot.py (wave4 + wave5 staging); exp115 rows",
                sota=0.84, sota_src="sun_fu_2026", protocol="gkf_paper5x10", pooled=ns["pooled_ci"], rec_extra=0.2475)


# ------------------------------------------------------------------ run ----------------------------------------------
def run(S):
    t0 = time.time()
    X, y, g, logt = S["X"], S["y"], S["g"], S["logt"]
    exname, exg = S["extra"] if S["extra"] else ("", None)
    print(f"{S['name']}: {len(y)} rows / {len(np.unique(g))} groups / {X.shape[1]} features; log1p: {logt}; incumbent "
          f"{S['exp']}; sklearn {sklearn.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}; "
          f"OMP {os.environ['OMP_NUM_THREADS']}, LOKY_MAX_CPU_COUNT {os.environ['LOKY_MAX_CPU_COUNT']}", flush=True)

    # ---- step 1 ----
    inc = score(S["inc"], X, y, g, logt, S.get("pooled"), exg)
    print(line("inc " + S["exp"], inc, exname=exname), flush=True)
    m0, l0, h0 = S["rec"]
    dev = (inc["rk"].mean() - m0, inc["lo"] - l0, inc["hi"] - h0)
    if not all(abs(v) <= TOL for v in dev):
        print(f"BLOCKED: {S['exp']} not reproduced within {TOL} (mean, ci_lo, ci_hi deviations {dev}); stopping.")
        sys.exit(2)
    print(f"REPRO {S['exp']} ok within {TOL}: mean {inc['rk'].mean():.6f} ci [{inc['lo']:.6f}, {inc['hi']:.6f}] "
          f"(record {S['rec']})", flush=True)
    if "rec_extra" in S:
        print(f"  {exname} {inc['ex']:.4f} vs record {S['rec_extra']} (report only)")

    # ---- candidates, config order ----
    res, failed = {}, {}
    for cid, sel, model, Xc, lg, feats, desc, hp in S["cands"]:
        try:
            res[cid] = score(model, Xc, y, g, lg, S.get("pooled"), exg)
        except Exception as e:  # exit trigger: GP fit raises / returns NaN -> non-selectable, still in K, reported
            failed[cid] = repr(e)
            print(f"{cid}: FAILED {e!r}", flush=True)
            continue
        print(line(cid, res[cid], inc, exname), flush=True)

    order = [c[0] for c in S["cands"]]
    sel = [c[0] for c in S["cands"] if c[1]]
    paths = {}
    for cid, s, model, Xc, lg, feats, desc, hp in S["cands"]:
        if cid not in res:
            continue
        r = res[cid]
        d = r["rk"] - inc["rk"]
        tk = topk(y, g, inc["oof"], r["oof"])
        role = "selectable" if s else "report-only (in K, never selectable)"
        note = f"round 10 {S['name']} {cid} ({role}); vs {S['exp']} on identical folds"
        if "ex" in r:
            note += f"; {exname} {r['ex']:.3f} (incumbent {inc['ex']:.3f})"
        if tk["not_established"]:
            note += "; gain not established without the top 5 groups"
        out = dict(slug=cid, target=S["target"], round_config=CONFIG, dataset=f"{S['dataset']}; incumbent {S['exp']} rows, order and groups",
                   features=feats, model=desc, hyperparameters=hp + "; fixed by the config, nothing tuned", code=CODE,
                   n_rows=int(len(y)), n_groups=int(len(np.unique(g))), protocol=S["protocol"],
                   rkf_mean=round(float(r["rk"].mean()), 4), rkf_std=round(float(r["rk"].std()), 4),
                   ci_lo=round(r["lo"], 4), ci_hi=round(r["hi"], 4), pooled_ci_lo=round(r["plo"], 4), pooled_ci_hi=round(r["phi"], 4),
                   paired_vs=S["exp"], paired_delta=round(float(d.mean()), 4), paired_wins=f"{int((d > 0).sum())}/10",
                   sota=S["sota"], sota_source=S["sota_src"], status="ok", blocked_reason="", note=note, role=role,
                   rkf_mean_full=float(r["rk"].mean()), ci_lo_full=r["lo"], ci_hi_full=r["hi"],
                   per_repeat_r2=[float(v) for v in r["rk"]], seed0_r2=r["r2_0"], seed0_mae=r["mae_0"], seed0_rmse=r["rmse_0"],
                   top_k_removal=tk, gp_warnings=r["gp_warnings"], wall_time_s=round(r["wall"], 1))
        if "ex" in r:
            key = "category_grouped" if S["slug"] == "phrr" else "c"
            out.update({f"{key}_mean": round(r["ex"], 4), f"{key}_std": round(r["ex_std"], 4), f"{key}_ci_lo": round(r["ex_lo"], 4),
                        f"{key}_ci_hi": round(r["ex_hi"], 4), f"{key}_per_repeat": r["ex_rk"]})
        p = RESULTS / f"{cid}.json"
        p.write_text(json.dumps(out, indent=2))
        paths[cid] = p
        print(f"wrote {p.relative_to(ROOT)}  top-k: {json.dumps({k: tk[k] for k in ('drop_top1', 'drop_top3', 'drop_top5')})}")

    incd = dict(rkf_mean_full=float(inc["rk"].mean()), per_repeat_r2=[float(v) for v in inc["rk"]], ci=[inc["lo"], inc["hi"]],
                seed0_r2=inc["r2_0"], seed0_mae=inc["mae_0"], seed0_rmse=inc["rmse_0"])
    if "ex" in inc:
        incd[exname] = dict(mean=inc["ex"], std=inc["ex_std"], ci=[inc["ex_lo"], inc["ex_hi"]])
    print("incumbent (reproduced):", json.dumps(incd))
    if failed:
        flag(f"FLAG ({S['name']}): candidate(s) failed {failed}; they stay in K but have no finite OOF, so adoption gate v2 "
             f"cannot be computed as written for this target. No gate run; no adoption decision.")
        print(f"runtime {time.time() - t0:.0f}s")
        return
    npz = RUNS / f"round10_{S['slug']}_oof.npz"
    np.savez(npz, y=y, groups=np.asarray(g).astype(str), row_id=np.arange(len(y)), oof_incumbent=inc["oof"],
             inc_per_repeat_r2=inc["rk"], inc_ci=np.array([inc["lo"], inc["hi"]]), order=np.array(order),
             selectable=np.array(sel), **{f"oof_{c}": res[c]["oof"] for c in order})
    print(f"wrote {npz.relative_to(ROOT)}")
    gate = adoption_gate.run_npz(str(npz), {c: str(p) for c, p in paths.items()}, str(RUNS / f"round10_{S['slug']}_gate_v2.json"))
    for c, p in paths.items():  # as train_round9_tabpfn.annotate (reported only)
        o = json.loads(p.read_text())
        o["gate_v2"] = dict(gate["per_candidate"][c], K=gate["K"], q=gate["q"], selected=(c == gate["best"]))
        if c == gate["best"]:
            o["note"] += (f"; gate v2 selected (K={gate['K']}, q={gate['q']:.3f}, LB={gate['per_candidate'][c]['lb_maxT']:+.4f}): "
                          f"G1 {gate['G1']}, G2 {gate['G2']}, G3 {gate['G3']}")
        p.write_text(json.dumps(o, indent=2))
    print(f"GATE {S['name']}: best {gate['best']}; G1 {gate['G1']} G2 {gate['G2']} G3 {gate['G3']} -> "
          + ("ADOPT pending fresh-context critic" if gate["adopt_if_critic_passes"] else f"NOT adopted; {S['exp']} stays"))
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["SELFTEST", "pHRR", "FIGRA", "CR", "LOI"])
    a = ap.parse_args()
    print(f"=== round 10 --target {a.target} ({time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ===")
    self_test()
    if a.target == "SELFTEST":
        sys.exit(0)
    nor_name, exp = {"pHRR": ("pHRR", "exp074"), "FIGRA": ("FIGRA", "exp119"), "CR": ("Char Residue", "exp117"),
                     "LOI": ("LOI", "exp115")}[a.target]
    check_record(nor_name, exp)
    if a.target == "LOI":
        work = Path(tempfile.mkdtemp(prefix="dravix_r10_loi_"))
        try:
            run(spec_loi(work))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    else:
        run({"pHRR": spec_phrr, "FIGRA": spec_figra, "CR": spec_cr}[a.target]())
