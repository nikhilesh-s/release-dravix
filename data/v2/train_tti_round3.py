#!/usr/bin/env python3
"""TTI round 3 (the final TTI model round) -- follows experiments/configs/tti_round3.json literally, together with the
tti_arm (p1, p2) of experiments/configs/round9_tabpfn_smearing.json (D4d). Amendments: D4a yes, D4b S675 canonical,
D4c keep a2, D4d family order a0, a1, o1, o2, a2, p1, p2 (a0 control); D1 yes -> adoption_gate_v2 (rule A).

Canonical scoring rows S675 (681 ignited spark rows minus the 6 operator-flagged replicates). S681 is reported only.
exp046 re-scored on S675 is the incumbent. p1/p2 run on S675 only.

Modes (one script, nothing else in the repo changes):
  --tabpfn-setup   label-free, synthetic only: post-install checks, V2 weight fetch through the package's own
                   downloader (HF_HUB_OFFLINE=0 only here), sha256/size check, fit+predict twice in-process and via
                   clone; saves predictions for the fresh-process check. Writes experiments/runs/tti_round3_tabpfn_env.json.
  --step0          step 0 only: self-tests T1-T4 (synthetic), every data/QA/censored-row assert, o1.check, the
                   amendment checks, gate self-test, TabPFN fresh-process determinism + sha256. No model is fit on labels.
  --record-only    numbers-of-record repro of exp154 only (run_numbers_of_record.sh): build_data() with every
                   data/QA assert (the 6 flagged ids from tti_round3.json), then steps 1-2 (steps12(), the same
                   function the default run calls). No self-tests, TabPFN, candidates or output files.
  (default)        step 0, then steps 1-6.

Executor notes (for the critic):
- No edit to train_tti_v5_physics.py. Its diagnostics-only repeated_group_kfold(..., repeats=1) call without
  log_target (line ~265) is in run(), which is never called here; every model in this script is scored on the raw
  seconds scale through eval_protocol.repeated_group_kfold or rgkf() below, which has no log_target path at all.
- Regexes, flagged ids and censored ids are read from the config JSON, not retyped.
- TABPFN_MODEL_CACHE_DIR is the darwin default for this user (~/Library/Caches/tabpfn); the config names
  ~/... as "the darwin default, made explicit" (that home directory does not exist here).
- data/v2/adoption_gate.py (written by another worker per adoption_gate_v2 pseudo_code) is used unchanged after its
  self-test and a check that every pseudo_code line occurs in it in order.

Run: OMP_NUM_THREADS=1 .venv312/bin/python data/v2/train_tti_round3.py | tee experiments/runs/train_tti_round3.log
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
assert MODE in ("run", "--step0", "--tabpfn-setup", "--record-only"), MODE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TABPFN_MODEL_CACHE_DIR"] = os.path.expanduser("~/Library/Caches/tabpfn")
os.environ["TABPFN_EXCLUDE_DEVICES"] = "cuda,mps"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" if MODE == "--tabpfn-setup" else "1"

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import check_grad, minimize  # noqa: E402
from scipy.special import expit, log_ndtr  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import PoissonRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data" / "v2"
sys.path.insert(0, str(HERE))
_argv = sys.argv
sys.argv = [_argv[0], "--onset-bc", "--replicate-table"]
import train_tti_v5_physics as v5  # noqa: E402
sys.argv = _argv
import adoption_gate  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402
from train_blend_round1 import Avg  # noqa: E402

T_START = time.time()
# numpy 2.2 + macOS Accelerate sets spurious FP flags in matmul ("divide by zero/overflow/invalid value encountered in
# matmul") on finite inputs; ignored everywhere. Real overflow shows up as non-finite OOF (checked) or exp warnings.
MATMUL_FPE = r".*encountered in matmul"
warnings.filterwarnings("ignore", message=MATMUL_FPE)
RUNS, RESULTS = ROOT / "experiments/runs", ROOT / "experiments/results"
CFG_PATH, R9_PATH = "experiments/configs/tti_round3.json", "experiments/configs/round9_tabpfn_smearing.json"
CFG = json.loads((ROOT / CFG_PATH).read_text())
R9 = json.loads((ROOT / R9_PATH).read_text())
GATE_CFG = json.loads((ROOT / "experiments/configs/adoption_gate_v2.json").read_text())
SOTA = 0.58
PHYS = v5.PHYS
REPEATS = 10
CANON, NONCANON = "S675", "S681"
FAMILY = ["a0", "a1", "o1", "o2", "a2", "p1", "p2"]   # D4d order; a0 control
B4 = ["a0", "a1", "o1", "o2", "a2"]
TABPFN_IDS = ["p1", "p2"]
SELECTABLE0 = ["a1", "o1", "o2", "a2", "p1", "p2"]
AFT_IDS = ["a0", "a1", "a2"]
REG, QCAP, Q0 = 0.01, 13.5, 6.75
OPT = dict(maxiter=20000, maxfun=200000, ftol=1e-13, gtol=1e-9)
CLIP = 1e-6
TABPFN_W = next(v for k, v in R9.items() if k.startswith("tabpfn_model"))["weights"]
SHA256 = TABPFN_W["sha256_exact"].split()[0]
CKPT = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / "tabpfn-v2-regressor.ckpt"
SETUP_NPZ = RUNS / "tti_round3_tabpfn_setup_preds.npz"
SETUP_JSON = RUNS / "tti_round3_tabpfn_env.json"


def log(*a):
    print(*a, flush=True)


# =============================== models ===============================
def glm():
    return v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000))


def prep():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())


def qk(X):
    """q in kW/m2 from X[:, 0] = log(1000 q); rounding undoes the float round-trip (config o1.wrapper)."""
    return np.round(np.exp(X[:, 0]) / 1000, 6)


def mills(s):
    return np.exp(norm.logpdf(s) - log_ndtr(s))


def aft_obj(w, Xi, li, Xc, kc, alpha):
    """a0/a1 objective and gradient (Olsen parameterisation). w = (g0, g, th)."""
    n = len(li)
    g0, g, th = w[0], w[1:-1], w[-1]
    r = th * li - g0 - Xi @ g
    F = np.sum(-np.log(th) + 0.5 * r ** 2)
    G0, Gg, Gth = -r.sum(), -(Xi.T @ r), np.sum(-1.0 / th + r * li)
    if len(kc):
        s = g0 + Xc @ g - th * kc
        lam = mills(s)
        F += -log_ndtr(s).sum()
        G0, Gg, Gth = G0 - lam.sum(), Gg - Xc.T @ lam, Gth + (lam * kc).sum()
    F = F / n + 0.5 * alpha * (g @ g)
    return F, np.concatenate([[G0 / n], Gg / n + alpha * g, [Gth / n]])


def aft_start(Xi, li):
    A = np.column_stack([np.ones(len(li)), Xi])
    beta = np.linalg.lstsq(A, li, rcond=None)[0]
    s0 = np.sqrt(np.mean((li - A @ beta) ** 2))
    return np.concatenate([beta / s0, [1.0 / s0]])


def qcr_parts(w, Z, q, pz):
    a, b, c, d = w[0], w[1:1 + pz], w[1 + pz], w[2 + pz:2 + 2 * pz]
    p = expit(c + Z @ d)
    e = q - QCAP * p
    return a + Z @ b - 2 * np.log(e), e, QCAP * p * (1 - p), QCAP * p


def qcr_obj(w, Zi, qi, li, Zc, qc, kc, alpha):
    """a2 objective and gradient. w = (a, b, c, d, eta), sigma = exp(eta)."""
    n, pz = len(li), Zi.shape[1]
    b, d, eta = w[1:1 + pz], w[2 + pz:2 + 2 * pz], w[-1]
    sig = np.exp(eta)
    m, e, dq, _ = qcr_parts(w, Zi, qi, pz)
    z = (li - m) / sig
    F = np.sum(eta + 0.5 * z ** 2)
    dm = -z / sig
    Geta = np.sum(1 - z ** 2)
    t = dm * (2 / e) * dq
    Ga, Gb, Gc, Gd = dm.sum(), Zi.T @ dm, t.sum(), Zi.T @ t
    if len(kc):
        mc, ec, dqc, _ = qcr_parts(w, Zc, qc, pz)
        s = (mc - kc) / sig
        lam = mills(s)
        F += -log_ndtr(s).sum()
        dmc = -lam / sig
        Geta += np.sum(lam * s)
        tc = dmc * (2 / ec) * dqc
        Ga, Gb, Gc, Gd = Ga + dmc.sum(), Gb + Zc.T @ dmc, Gc + tc.sum(), Gd + Zc.T @ tc
    F = F / n + 0.5 * alpha * (b @ b + d @ d)
    return F, np.concatenate([[Ga / n], Gb / n + alpha * b, [Gc / n], Gd / n + alpha * d, [Geta / n]])


def qcr_start(Zi, qi, li):
    t = li + 2 * np.log(qi - Q0)
    A = np.column_stack([np.ones(len(li)), Zi])
    beta = np.linalg.lstsq(A, t, rcond=None)[0]
    eta0 = np.log(np.sqrt(np.mean((t - A @ beta) ** 2)))
    return np.concatenate([beta, [0.0], np.zeros(Zi.shape[1]), [eta0]])


def lbfgs(fun, w0, bounds=None):
    res = minimize(fun, w0, jac=True, method="L-BFGS-B", bounds=bounds, options=OPT)
    gmax = float(np.abs(fun(res.x)[1]).max())
    return res, bool(res.success or gmax <= 1e-5), gmax


class AFT(BaseEstimator, RegressorMixin):
    """a0 (censored=False) / a1 (censored=True): lognormal AFT on PHYS, global MLE sigma, capped lognormal mean."""

    def __init__(self, censored=False, alpha=REG):
        self.censored = censored
        self.alpha = alpha

    def fit(self, X, y, X_cens=None, c_cens=None):
        self.pp_ = prep().fit(X)
        Xi, li = self.pp_.transform(X), np.log(y)
        if X_cens is not None and len(X_cens):
            Xc, kc = self.pp_.transform(X_cens), np.log(c_cens)
        else:
            Xc, kc = np.empty((0, Xi.shape[1])), np.empty(0)
        w0 = aft_start(Xi, li)
        res, conv, gmax = lbfgs(lambda w: aft_obj(w, Xi, li, Xc, kc, self.alpha), w0,
                                [(None, None)] * (len(w0) - 1) + [(1e-8, None)])
        self.w_, self.cap_ = res.x, float(np.max(y))
        self.info_ = dict(converged=conv, success=bool(res.success), n_iter=int(res.nit), gmax=gmax,
                          sigma=float(1 / res.x[-1]), n_C=int(len(kc)), message=str(res.message))
        return self

    def mean_uncapped(self, X):
        g0, g, th = self.w_[0], self.w_[1:-1], self.w_[-1]
        return np.exp((g0 + self.pp_.transform(X) @ g) / th + 0.5 / th ** 2)

    def predict(self, X):
        return np.minimum(self.mean_uncapped(X), self.cap_)


class QcrAFT(BaseEstimator, RegressorMixin):
    """a2: t = A(x) / (q - q_cr(x))^2, q_cr = QCAP * expit(c + Zs.d), lognormal AFT with censored rows."""
    censored = True

    def __init__(self, alpha=REG):
        self.alpha = alpha

    def fit(self, X, y, X_cens=None, c_cens=None):
        qi, Z = qk(X), X[:, 1:5]
        self.pp_ = prep().fit(Z)
        Zi, li = self.pp_.transform(Z), np.log(y)
        if X_cens is not None and len(X_cens):
            Zc, qc, kc = self.pp_.transform(X_cens[:, 1:5]), qk(X_cens), np.log(c_cens)
        else:
            Zc, qc, kc = np.empty((0, Zi.shape[1])), np.empty(0), np.empty(0)
        w0 = qcr_start(Zi, qi, li)
        self.fun_ = lambda w: qcr_obj(w, Zi, qi, li, Zc, qc, kc, self.alpha)
        res, conv, gmax = lbfgs(self.fun_, w0)
        self.w_, self.cap_, self.pz_ = res.x, float(np.max(y)), Zi.shape[1]
        self.info_ = dict(converged=conv, success=bool(res.success), n_iter=int(res.nit), gmax=gmax,
                          sigma=float(np.exp(res.x[-1])), n_C=int(len(kc)), message=str(res.message), F=float(res.fun))
        return self

    def qcr(self, X):
        return qcr_parts(self.w_, self.pp_.transform(X[:, 1:5]), qk(X), self.pz_)[3]

    def mean_uncapped(self, X):
        m = qcr_parts(self.w_, self.pp_.transform(X[:, 1:5]), qk(X), self.pz_)[0]
        return np.exp(m + 0.5 * np.exp(self.w_[-1]) ** 2)

    def predict(self, X):
        return np.minimum(self.mean_uncapped(X), self.cap_)


class OffsetPoissonHGB(BaseEstimator, RegressorMixin):
    """o1: HGB Poisson with fixed thermally-thick offset log o = -2 log(q/50)."""

    def fit(self, X, y):
        q = qk(X)
        self.hgb_ = HistGradientBoostingRegressor(loss="poisson", **v5.HGB).fit(X, y * (q / 50) ** 2,
                                                                               sample_weight=(50 / q) ** 2)
        return self

    def predict(self, X):
        return self.hgb_.predict(X) * (50 / qk(X)) ** 2


def tabpfn():
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


# =============================== CV loop ===============================
def rgkf(model, X, y, g, XC=None, cC=None, gC=None, cells=None, ridC=None, benchC=None):
    """rgkf_censored: splitting copied from eval_protocol.repeated_group_kfold. Censored models get the fold's
    admissible C rows (group not among the fold's test groups). Returns (r2 per repeat, oof, per-fit info, warnings)."""
    oof = np.empty((REPEATS, len(y)))
    infos = []
    cens = bool(getattr(model, "censored", False))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=MATMUL_FPE)
        for r in range(REPEATS):
            gkf = GroupKFold(n_splits=5, shuffle=True, random_state=r)
            for k, (tr, te) in enumerate(gkf.split(X, y, g)):
                assert set(g[tr]).isdisjoint(g[te])
                m = clone(model)
                t0 = time.time()
                info = dict(r=r, k=k)
                if cens:
                    adm = ~np.isin(gC, g[te])
                    assert not (set(gC[adm]) & set(g[te])), "C-row leakage"
                    m.fit(X[tr], y[tr], XC[adm], cC[adm])
                else:
                    m.fit(X[tr], y[tr])
                oof[r, te] = m.predict(X[te])
                info["fit_s"] = round(time.time() - t0, 3)
                info.update(getattr(m, "info_", {}))
                if hasattr(m, "cap_"):
                    info["cap"] = m.cap_
                    info["n_at_cap"] = int((oof[r, te] == m.cap_).sum())
                if isinstance(m, QcrAFT):
                    qc = m.qcr(X[te])
                    info["qcr_test"] = [float(qc.min()), float(np.median(qc)), float(qc.max())]
                if cells is not None and r == 0 and cens:
                    for i in np.flatnonzero(np.isin(gC, g[te]) & benchC):
                        un = float(m.mean_uncapped(XC[i:i + 1])[0])
                        cells.append(dict(row_id=str(ridC[i]), censor_s=float(cC[i]), uncapped_mean_s=un,
                                          cap_fold_s=m.cap_, capped_pred_s=min(un, m.cap_)))
                infos.append(info)
    wcount = {}
    for w in wl:
        key = f"{w.category.__name__}: {str(w.message)[:60]}"
        wcount[key] = wcount.get(key, 0) + 1
    return np.array([r2_score(y, p) for p in oof]), oof, infos, wcount


# =============================== step 0 ===============================
def self_tests():
    out = {}
    # ---- T1 gradients ----
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    l = 5 + X @ np.array([0.5, -0.3, 0.2]) + 0.6 * rng.normal(size=500)
    q = rng.choice([15.0, 25.0, 50.0, 75.0], size=500)
    cens = rng.random(500) < 0.10
    X[rng.random((500, 3)) < 0.10] = np.nan
    pp = prep().fit(X[~cens])
    Xi, Xc = pp.transform(X[~cens]), pp.transform(X[cens])
    li, kc = l[~cens], l[cens]
    f1 = lambda w: aft_obj(w, Xi, li, Xc, kc, REG)  # noqa: E731
    f2 = lambda w: qcr_obj(w, Xi, q[~cens], li, Xc, q[cens], kc, REG)  # noqa: E731
    w1, w2 = aft_start(Xi, li), qcr_start(Xi, q[~cens], li)
    pts1 = [w1] + [w1 + 0.1 * rng.normal(size=w1.size) for _ in range(3)]
    pts2 = [w2] + [w2 + 0.1 * rng.normal(size=w2.size) for _ in range(3)]
    rel = []
    for f, pts in ((f1, pts1), (f2, pts2)):
        for w in pts:
            rel.append(check_grad(lambda v: f(v)[0], lambda v: f(v)[1], w) / np.linalg.norm(f(w)[1]))
    out["T1"] = dict(rel_err=[float(v) for v in rel], ok=bool(max(rel) < 1e-5), p_a1=int(w1.size), p_a2=int(w2.size))
    # ---- T2 AFT recovery ----
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 3))
    eps = rng.normal(size=20000)
    true = np.array([0.5, -0.3, 0.2])
    l = 5 + X @ true + 0.6 * eps
    c = l > 5.3
    m1 = AFT(censored=True, alpha=0.0).fit(X[~c], np.exp(l[~c]), X[c], np.full(c.sum(), np.exp(5.3)))
    th = m1.w_[-1]
    slope = m1.w_[1:-1] / (th * m1.pp_[-1].scale_)
    ok_a1 = bool(np.all(np.abs(slope - true) < 0.03) and abs(1 / th - 0.6) < 0.03 and m1.info_["converged"])
    m0 = AFT(censored=False, alpha=0.0).fit(X, np.exp(l))
    Xs = m0.pp_.transform(X)
    A = np.column_stack([np.ones(len(l)), Xs])
    beta = np.linalg.lstsq(A, l, rcond=None)[0]
    sd = np.sqrt(np.mean((l - A @ beta) ** 2))
    th0 = m0.w_[-1]
    dev = np.abs(np.concatenate([m0.w_[:-1] / th0 - beta, [1 / th0 - sd]]))
    out["T2"] = dict(cens_share=float(c.mean()), a1_raw_slopes=slope.tolist(), a1_sigma=float(1 / th),
                     a1_converged=m1.info_["converged"], a0_max_abs_dev=float(dev.max()),
                     ok=bool(ok_a1 and dev.max() < 1e-6))
    # ---- T3 a2 reaches truth ----
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(2000, 2))
    q = rng.choice([15.0, 25.0, 50.0, 75.0], size=2000)
    eps = rng.normal(size=2000)
    a, b, cc, d, sig = 10.0, np.array([0.5, -0.3]), 0.0, np.array([1.0, -0.5]), 0.4
    l = a + Z @ b - 2 * np.log(q - QCAP * expit(cc + Z @ d)) + sig * eps
    cen = l > np.log(600)
    Xf = np.column_stack([np.log(1000 * q), Z])
    m2 = QcrAFT().fit(Xf[~cen], np.exp(l[~cen]), Xf[cen], np.full(cen.sum(), 600.0))
    sc, mu = m2.pp_[-1].scale_, m2.pp_[-1].mean_
    w_true = np.concatenate([[a + b @ mu], b * sc, [cc + d @ mu], d * sc, [np.log(sig)]])
    F_true, F_fit = m2.fun_(w_true)[0], m2.fun_(m2.w_)[0]
    out["T3"] = dict(cens_share=float(cen.mean()), converged=m2.info_["converged"], F_fit=float(F_fit),
                     F_true=float(F_true), ok=bool(m2.info_["converged"] and F_fit <= F_true + 1e-6))
    # ---- T4 offset identity ----
    rng = np.random.default_rng(3)
    X = np.column_stack([np.full(500, np.log(50000.0)), rng.normal(size=(500, 4))])
    y = np.exp(3 + 0.5 * X[:, 1] + 0.3 * rng.normal(size=500))
    p_off = OffsetPoissonHGB().fit(X, y).predict(X)
    p_ref = HistGradientBoostingRegressor(loss="poisson", **v5.HGB).fit(X, y).predict(X)
    out["T4"] = dict(ok=bool(np.array_equal(p_off, p_ref)))
    for t, v in out.items():
        log(f"  self-test {t}: {'PASS' if v['ok'] else 'FAIL'}  {json.dumps({k: x for k, x in v.items() if k != 'ok'})}")
    return out


def cfg_get(d, prefix):
    return next(v for k, v in d.items() if k.startswith(prefix))


def check_amendments():
    a3, a9, ag = CFG["amendments"][0], R9["amendments"][0], GATE_CFG["amendments"][0]
    checks = {
        "tti_round3 status approved": CFG["status"].startswith("approved"),
        "D4a yes": "D4a yes" in a3,
        "D4b S675 canonical": "D4b: S675 is canonical" in a3,
        "D4c keep a2": "D4c: keep a2" in a3,
        "D4d family order (tti_round3)": "D4d: yes" in a3 and "a0, a1, o1, o2, a2, p1, p2" in a3,
        "D4d family order (round9)": "D4d yes" in a9 and "a0, a1, o1, o2, a2, p1, p2" in a9,
        "round9: tti_round3 output names govern": "tti_round3's output names govern" in a9,
        "round9: TTI arm uses S675": "canonical S675" in a9,
        "D1 yes (gate v2)": "D1 answered yes" in a3 and ag.startswith("D1 yes"),
        "human_override struck": "human_override clause is STRUCK" in ag,
    }
    for k, v in checks.items():
        log(f"  amendment check {k}: {'OK' if v else 'MISSING'}")
    return checks


def gate_provenance():
    """Every pseudo_code line occurs in data/v2/adoption_gate.py in order (whitespace-insensitive: line wrapping may
    differ), then its own label-free self-test."""
    src = re.sub(r"\s+", "", (HERE / "adoption_gate.py").read_text())
    pos = 0
    for line in GATE_CFG["pseudo_code"]:
        line = re.sub(r"\s+", "", line)
        if line.startswith("#data/v2/adoption_gate.py") or not line:
            continue
        i = src.find(line, pos)
        assert i >= 0, f"adoption_gate.py lacks pseudo_code line: {line!r}"
        pos = i + len(line)
    adoption_gate.self_test()
    return True


def build_data():
    rep = pd.read_csv(HERE / "cone_replicates_fsri.csv")
    assert len(rep) == 735 and rep["material"].nunique() == 81, (len(rep), rep["material"].nunique())
    rep["row_id"] = rep["material"] + "|" + rep["hf_tag"] + "|R" + rep["replicate"].astype(str)
    assert rep["row_id"].is_unique
    # ---- data_qa_decision rule (text only) ----
    rule = cfg_get(CFG["data_qa_decision"], "rule")
    POS, NEG = re.compile(rule["POS"], re.IGNORECASE), re.compile(rule["NEG"], re.IGNORECASE)
    cmt = rep["post_test_cmt"].fillna("")
    pos, neg = cmt.map(lambda s: bool(POS.search(s))), cmt.map(lambda s: bool(NEG.search(s)))
    flag = pos & ~neg
    flagged = cfg_get(CFG["data_qa_decision"], "flagged_ids")
    assert set(rep.loc[flag, "row_id"]) == set(flagged) and flag.sum() == 6 == len(flagged), sorted(rep.loc[flag, "row_id"])
    assert "PET|HF25|R1" not in set(rep.loc[flag, "row_id"])
    assert {"ABS|HF75|R1", "HIPS|HF25|R3"} <= set(rep.loc[neg, "row_id"])
    log(f"  QA regex on all 735 rows: POS {int(pos.sum())}, NEG {int(neg.sum())}, overlap {int((pos & neg).sum())}, "
        f"flagged {int(flag.sum())} == the 6 listed ids")
    # ---- U, joins, PHYS (v5 lines 107-131) ----
    U = rep[~rep["early_stop"] & rep["spark_ignitor"]].reset_index(drop=True)
    assert len(U) == 723 and U["material"].nunique() == 79
    assert not U["row_id"].str.startswith("PMMA|HF15-noSpark").any()
    U = U.join(v5.mcc, on="material").join(v5.mcc_x, on="material").join(v5.props, on="material").join(v5.thermal, on="material")
    assert len(U) == 723
    q, dT, L = U["heat_flux_kWm2"] * 1e3, U["t_onset_C"] - v5.T0_C, U["cone_thickness_mm"] / 1e3
    U["log_q"] = np.log(q)
    U["log_dT"] = v5.log_pos(dT)
    U["log_kpc"] = v5.log_pos(U["k_WmK"] * U["rhoc_Jm3K"])
    U["log_L"] = v5.log_pos(L)
    U["log_rho"] = v5.log_pos(U["density_kgm3"])
    U["family"] = U["categories"].fillna("unknown").str.split(";").str[0].str.strip()
    U["flag6"] = U["row_id"].isin(flagged)
    assert not {"tti_s", "end_of_test_time_s", "did_not_ignite", "early_stop", "test_status", "replicate",
                "test_date", "post_test_cmt"} & set(PHYS)
    # o1.check on the full U table
    assert np.array_equal(qk(U[PHYS].to_numpy(float)), U["heat_flux_kWm2"].to_numpy(float)), "o1.check"
    # ---- S681 ----
    S681 = U[~U["did_not_ignite"]].reset_index(drop=True)
    assert len(S681) == 681 and S681["material"].nunique() == 76
    vid = (v5.df["material"] + "|" + v5.df["hf_tag"] + "|R" + v5.df["replicate"].astype(str)).tolist()
    assert S681["row_id"].tolist() == vid, "S681 not index-aligned with v5.df"
    assert np.array_equal(S681["tti_s"].to_numpy(float), v5.df["time_to_ignition_s"].to_numpy(float))
    assert np.array_equal(S681[PHYS].to_numpy(float), v5.df[PHYS].to_numpy(float), equal_nan=True)
    assert np.array_equal(S681["family"].to_numpy(), v5.df["family"].to_numpy())
    fc = lambda d: {float(k): int(v) for k, v in d["heat_flux_kWm2"].value_counts().sort_index().items()}  # noqa: E731
    assert fc(S681) == {15.0: 3, 25.0: 223, 35.0: 16, 50.0: 221, 75.0: 217, 100.0: 1}, fc(S681)
    assert S681["flag6"].sum() == 6
    # ---- S675 ----
    S675 = S681[~S681["flag6"]].reset_index(drop=True)
    assert len(S675) == 675 and S675["material"].nunique() == 76
    assert fc(S675) == {15.0: 3, 25.0: 218, 35.0: 16, 50.0: 221, 75.0: 216, 100.0: 1}, fc(S675)
    for S in (S681, S675):
        assert (S["tti_s"] > 0).all()
    # ---- C (censored) ----
    C = U[U["did_not_ignite"]].reset_index(drop=True)
    assert len(C) == 42 and C["material"].nunique() == 8
    bench = cfg_get(CFG["censored_rows"], "ids_benchmark")
    inert = cfg_get(CFG["censored_rows"], "ids_never")
    assert set(C["row_id"]) == set(bench) | set(inert) and len(bench) == 15 and len(inert) == 27
    cc = C["end_of_test_time_s"].to_numpy(float)
    assert np.isfinite(cc).all() and (cc > 0).all()
    assert fc(C) == {25.0: 22, 50.0: 11, 75.0: 9}, fc(C)
    assert set(C.loc[C["row_id"].isin(bench), "material"]) <= set(S681["material"])
    assert not set(C.loc[C["row_id"].isin(inert), "material"]) & set(S681["material"])
    cov = C.drop_duplicates("material").set_index("material")[["log_dT", "log_kpc", "log_rho"]].isna()
    log("  C materials with missing PHYS inputs: " + "; ".join(
        f"{m} {[c for c in cov.columns if cov.loc[m, c]]}" for m in cov.index if cov.loc[m].any()))
    log(f"  U {len(U)} rows / {U['material'].nunique()} mats; S681 {len(S681)}; S675 {len(S675)}; C {len(C)} rows / "
        f"{C['material'].nunique()} mats (15 benchmark + 27 inert); flux S675 {fc(S675)}; C by flux {fc(C)}")

    def pack(S):
        return dict(X=S[PHYS].to_numpy(float), y=S["tti_s"].to_numpy(float), g=S["material"].to_numpy(),
                    fam=S["family"].to_numpy(), flux=S["heat_flux_kWm2"].to_numpy(float), rid=S["row_id"].to_numpy())
    D = {"S681": pack(S681), "S675": pack(S675)}
    CC = dict(X=C[PHYS].to_numpy(float), c=cc, g=C["material"].to_numpy(), fam=C["family"].to_numpy(),
              rid=C["row_id"].to_numpy(), bench=C["row_id"].isin(bench).to_numpy())
    return D, CC


def tabpfn_synthetic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([1, -2, 0.5, 0, 3]) + 0.1 * rng.normal(size=200)
    Xp = rng.normal(size=(50, 5))
    return X, y, Xp


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_ok():
    """S0.3: exact V2 checkpoint, size, and no 2.5+/3.x file in the cache."""
    if not CKPT.exists():
        return False, "checkpoint missing"
    sz, h = CKPT.stat().st_size, sha256(CKPT)
    bad = [p.name for p in CKPT.parent.iterdir() if re.search(r"2[._]5|2[._]6|tabpfn[-_]3|v3", p.name)]
    ok = sz == TABPFN_W["size_bytes"] and h == SHA256 and not bad
    return ok, f"size {sz}, sha256 {h}, non-V2 files {bad}"


def tabpfn_setup():
    import torch
    torch.set_num_threads(1)
    info = dict(python=sys.version, executable=sys.executable,
                env={k: os.environ[k] for k in ("TABPFN_MODEL_CACHE_DIR", "TABPFN_EXCLUDE_DEVICES",
                                                "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")})
    pins = R9["software_pins"]["already_installed_must_stay_unchanged"]
    got = {p: metadata.version(p) for p in pins}
    info["pins"] = {p: [pins[p], got[p]] for p in pins}
    info["pins_ok"] = all(pins[p] == got[p] for p in pins)
    info["installed"] = {p: metadata.version(p) for p in ("tabpfn", "einops", "lightgbm", "narwhals", "skrub",
                                                          "pydot", "cloudpickle")
                         if _has_dist(p)}
    pc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    named = [ln for ln in (pc.stdout + pc.stderr).splitlines()
             if re.search(r"\b(tabpfn|einops|lightgbm|narwhals|skrub|pydot|cloudpickle|mlx|mlx-metal)\b", ln, re.I)]
    info["pip_check_lines_naming_pinned"] = named
    info["pip_check_rc"] = pc.returncode
    import tabpfn as tp
    info["tabpfn_version"] = tp.__version__
    X, y, Xp = tabpfn_synthetic()
    t0 = time.time()
    m = tabpfn().fit(X, y)          # S0.2: first fit triggers the package's own download
    info["first_fit_s"] = round(time.time() - t0, 1)
    p1 = m.predict(Xp)
    t0 = time.time()
    p2 = tabpfn().fit(X, y).predict(Xp)
    info["fit_predict_s"] = round(time.time() - t0, 2)
    p3 = clone(m).fit(X, y).predict(Xp)
    info["determinism_in_process"] = bool(np.array_equal(p1, p2) and np.array_equal(p1, p3))
    info["clone_params_equal"] = repr(clone(m).get_params()) == repr(m.get_params())
    ok, msg = cache_ok()
    info["weights_ok"], info["weights"] = ok, msg
    info["fitted"] = {k: repr(getattr(m, k, None)) for k in ("n_estimators_", "devices_", "use_autocast_",
                                                            "forced_inference_dtype_")}
    info["model_path"] = str(m.get_params().get("model_path"))
    info["ok"] = bool(info["pins_ok"] and not named and info["tabpfn_version"] == "9.1.0" and ok
                      and info["determinism_in_process"] and info["clone_params_equal"])
    np.savez(SETUP_NPZ, pred=p1)
    SETUP_JSON.write_text(json.dumps(info, indent=2))
    log(json.dumps(info, indent=1))
    log(f"TABPFN SETUP {'OK' if info['ok'] else 'FAILED'}")
    return info["ok"]


def _has_dist(p):
    try:
        metadata.version(p)
        return True
    except metadata.PackageNotFoundError:
        return False


def tabpfn_fresh_check():
    """S0.4 fresh-process leg + S0.3 re-check; HF_HUB_OFFLINE=1 here."""
    if not SETUP_JSON.exists() or not SETUP_NPZ.exists():
        return False, "setup not run"
    setup = json.loads(SETUP_JSON.read_text())
    if not setup.get("ok"):
        return False, "setup failed"
    ok, msg = cache_ok()
    if not ok:
        return False, msg
    X, y, Xp = tabpfn_synthetic()
    p = tabpfn().fit(X, y).predict(Xp)
    same = bool(np.array_equal(p, np.load(SETUP_NPZ)["pred"]))
    return same, f"{msg}; fresh-process predictions bitwise equal: {same}"


def step0():
    log("=" * 100)
    log(f"STEP 0 (label-free self-tests and asserts) {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}")
    env = {p: metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pandas", "torch", "tabpfn")}
    log(f"  python {sys.version.split()[0]} ({sys.executable}); {env}")
    assert env["numpy"] == "2.2.6" and env["scipy"] == "1.16.1" and env["scikit-learn"] == "1.7.1" and env["pandas"] == "2.3.3"
    st = self_tests()
    assert all(v["ok"] for v in st.values()), "self-test failed -> FLAG"
    am = check_amendments()
    assert all(v for k, v in am.items() if not k.startswith("round9")) and am["D1 yes (gate v2)"], "amendments"
    tti_arm = all(am[k] for k in ("D4d family order (tti_round3)", "D4d family order (round9)",
                                  "round9: tti_round3 output names govern", "round9: TTI arm uses S675"))
    gate_provenance()
    log("  adoption_gate.py: pseudo_code lines present in order; self-test passed")
    D, CC = build_data()
    tp_ok, tp_msg = (False, "tti_arm not approved") if not tti_arm else tabpfn_fresh_check()
    log(f"  TabPFN arm: approved {tti_arm}; checks {'OK' if tp_ok else 'FAILED'} ({tp_msg})")
    log("STEP 0 PASSED" + ("" if tp_ok else " (TabPFN arm BLOCKED)"))
    return D, CC, st, am, tp_ok, tp_msg


# =============================== metrics ===============================
def stats(rk, oof, y, g, rk_inc=None):
    lo, hi = bootstrap_ci(y, oof[0], g)
    out = dict(per_repeat_r2=[float(v) for v in rk], rkf_mean=float(rk.mean()), rkf_std=float(rk.std()),
               ci=[float(lo), float(hi)])
    if rk_inc is not None:
        out.update(paired_delta=float((rk - rk_inc).mean()), paired_wins=int((rk > rk_inc).sum()))
    return out


SECONDARY = ["S25", "S50", "S75", "S50_75", "LOG_ALL", "LOG25", "LOG50", "LOG75",
             "SPR25", "SPR50", "SPR75", "SPM25", "SPM50", "SPM75"]
CI_METRICS = ["S25", "S50", "S75", "S50_75", "LOG_ALL"]


def secondary(d, oofs, inc="exp046"):
    """tti_secondary_metrics.json 'metrics', 'aggregation', 'confidence_intervals', 'paired_report' (definitions
    copied from data/v2/report_tti_secondary_metrics.py). Report-only, defined post hoc."""
    y, g, flux = d["y"], d["g"], d["flux"]
    masks = {"F25": flux == 25.0, "F50": flux == 50.0, "F75": flux == 75.0, "F50_75": np.isin(flux, [50.0, 75.0]),
             "ALL": np.ones(len(y), bool)}

    def metric(mname, p):
        if mname in ("S25", "S50", "S75", "S50_75"):
            m = masks["F" + mname[1:]]
            return r2_score(y[m], p[m])
        if mname.startswith("LOG"):
            m = masks["ALL" if mname == "LOG_ALL" else "F" + mname[3:]]
            return r2_score(np.log(y[m]), np.log(np.clip(p[m], CLIP, None)))
        m = masks["F" + mname[3:]]
        if mname.startswith("SPR"):
            return spearmanr(y[m], p[m]).statistic
        med = pd.DataFrame({"g": g[m], "y": y[m], "p": p[m]}).groupby("g")[["y", "p"]].median()
        return spearmanr(med["y"], med["p"]).statistic

    def ci0(mname, p0):
        if mname == "LOG_ALL":
            return bootstrap_ci(np.log(y), np.log(np.clip(p0, CLIP, None)), g, n_boot=2000, seed=0)
        m = masks["F" + mname[1:]]
        return bootstrap_ci(y[m], p0[m], g[m], n_boot=2000, seed=0)

    vals, out = {}, {"masks_rows": {k: int(v.sum()) for k, v in masks.items()},
                     "label": "report-only, defined post hoc (experiments/configs/tti_secondary_metrics.json); "
                              "never promotes; not comparable to the 0.58 bar", "models": {}}
    for mid, oof in oofs.items():
        vals[mid] = {k: np.array([metric(k, oof[r]) for r in range(REPEATS)], float) for k in SECONDARY}
        mo = {}
        for k in SECONDARY:
            pr = vals[mid][k]
            ci = [float(v) for v in ci0(k, oof[0])] if k in CI_METRICS else None
            mo[k] = dict(per_repeat=pr.tolist(), mean=float(np.mean(pr)), std=float(np.std(pr)), rep0=float(pr[0]),
                         ci_rep0=ci, wins_vs_exp046=None)
        mo["n_clipped"] = int((oof <= CLIP).sum())
        out["models"][mid] = mo
    for mid in oofs:
        if mid == inc:
            continue
        for k in SECONDARY:
            a, b = vals[mid][k], vals[inc][k]
            out["models"][mid][k]["wins_vs_exp046"] = int(((a > b) & np.isfinite(a) & np.isfinite(b)).sum())
    return out


def legacy(d, oof):
    f25 = d["flux"] == 25.0
    return dict(r2_25_rep0=float(r2_score(d["y"][f25], oof[0][f25])),
                r2_non25_pooled_rep0=float(r2_score(d["y"][~f25], oof[0][~f25])),
                label="repeat-0 seconds R2 on the 25 kW/m2 rows and on the non-25 rows pooled (15/35/50/75/100)")


def topk(d, oof_inc, oof_c):
    y, g = d["y"], d["g"]
    row = (((y - oof_inc) ** 2) - ((y - oof_c) ** 2)).mean(axis=0)
    contrib = pd.Series(row).groupby(g).sum()
    ranked = sorted(contrib.index, key=lambda m: (-contrib[m], m))
    out = {}
    for k in (1, 3, 5):
        keep = ~np.isin(g, ranked[:k])
        dd = np.array([r2_score(y[keep], oof_c[r, keep]) - r2_score(y[keep], oof_inc[r, keep]) for r in range(REPEATS)])
        out[f"top{k}_removed"] = dict(mean_gain=float(dd.mean()), wins=int((dd > 0).sum()))
    out["top5"] = [[m, float(contrib[m])] for m in ranked[:5]]
    out["net_contribution_sum"] = float(contrib.sum())
    out["top5_share_of_net"] = float(contrib[ranked[:5]].sum() / contrib.sum())
    t5 = out["top5_removed"]
    out["not_established"] = bool(t5["mean_gain"] <= 0 or t5["wins"] < 8)
    return out


def sse_share(d, oof):
    se = pd.Series((d["y"] - oof[0]) ** 2).groupby(d["g"]).sum().sort_values(ascending=False)
    return dict(top5=[[m, float(v / se.sum())] for m, v in se.head(5).items()], top5_share=float(se.head(5).sum() / se.sum()))


def fit_summary(infos):
    out = {}
    if infos and "converged" in infos[0]:
        sig = [i["sigma"] for i in infos]
        out.update(converged=int(sum(i["converged"] for i in infos)), n_fits=len(infos),
                   success=int(sum(i["success"] for i in infos)), n_iter=[i["n_iter"] for i in infos],
                   gmax_max=float(max(i["gmax"] for i in infos)),
                   sigma_median=float(np.median(sig)), sigma_range=[float(min(sig)), float(max(sig))],
                   n_at_cap=int(sum(i["n_at_cap"] for i in infos)), n_C_per_fold=[i["n_C"] for i in infos],
                   messages=sorted({i["message"] for i in infos}))
    if infos and "qcr_test" in infos[0]:
        out["qcr_test_min_median_max_per_fit"] = [i["qcr_test"] for i in infos]
    out["fit_s_total"] = round(sum(i["fit_s"] for i in infos), 1)
    return out


# =============================== steps 1-2 ===============================
def steps12(D):
    """exp046 reproduced on S681 (else exit 2) and re-scored on S675 (= exp154), each with the folds identity check."""
    log("=" * 100)
    inc = {}
    for rs in (NONCANON, CANON):
        d = D[rs]
        t0 = time.time()
        rk, oof = repeated_group_kfold(glm(), d["X"], d["y"], d["g"])
        lo, hi = bootstrap_ci(d["y"], oof[0], d["g"])
        log(f"STEP {'1' if rs == 'S681' else '2'}: exp046 on {rs}: rkf_mean {rk.mean():.4f} +- {rk.std():.4f}; "
            f"repeat-0 CI [{lo:.4f}, {hi:.4f}] ({time.time() - t0:.0f}s)")
        if rs == "S681":
            ok = abs(rk.mean() - 0.3133) <= 0.002 and abs(lo - 0.0715) <= 0.002 and abs(hi - 0.4958) <= 0.002
            if not ok:
                log("BLOCKED: exp046 not reproduced on S681 within 0.002; stopping.")
                sys.exit(2)
            log("  REPRO exp046 (S681) ok")
        rk2, oof2, infos, wc = rgkf(glm(), d["X"], d["y"], d["g"])
        if not np.array_equal(oof, oof2):
            log(f"BLOCKED: folds.identity_check failed on {rs}; stopping.")
            sys.exit(2)
        log(f"  identity_check {rs}: rgkf OOF == eval_protocol OOF (np.array_equal); warnings {wc}")
        inc[rs] = dict(rk=rk, oof=oof, ci=(lo, hi), warn=wc)
    return inc


# =============================== main ===============================
def main():
    if MODE == "--tabpfn-setup":
        sys.exit(0 if tabpfn_setup() else 2)
    if MODE == "--record-only":
        steps12(build_data()[0])
        log(f"record-only: runtime {time.time() - T_START:.0f}s")
        return
    D, CC, selftests, amend, tp_ok, tp_msg = step0()
    if MODE == "--step0":
        return
    flags = []
    if not tp_ok:
        flags.append(f"TabPFN arm BLOCKED ({tp_msg}); B4 candidates run; TTI adoption decision parked (family incomplete)")
    import torch
    torch.set_num_threads(1)

    # ---------------- step 1 & 2: exp046 reproduction / re-baseline + identity checks ----------------
    inc = steps12(D)

    # ---------------- step 3 & 4: candidates (primary GKF) and family-GKF ----------------
    models = {"exp046": glm(), "a0": AFT(censored=False), "a1": AFT(censored=True), "o1": OffsetPoissonHGB(),
              "o2": Avg([glm(), OffsetPoissonHGB()]), "a2": QcrAFT()}
    if tp_ok:
        models.update(p1=tabpfn(), p2=Avg([glm(), tabpfn()]))
    run = {rs: {"material": {}, "family": {}} for rs in D}
    cells = {"a1": [], "a2": []}
    for rs in (CANON, NONCANON):
        d = D[rs]
        ids = ["exp046"] + B4 + (TABPFN_IDS if (tp_ok and rs == CANON) else [])
        for grouping in ("material", "family"):
            g, gC = (d["g"], CC["g"]) if grouping == "material" else (d["fam"], CC["fam"])
            for cid in ids:
                t0 = time.time()
                cl = cells[cid] if (cid in cells and rs == CANON and grouping == "material") else None
                rk, oof, infos, wc = rgkf(models[cid], d["X"], d["y"], g, CC["X"], CC["c"], gC, cl, CC["rid"], CC["bench"])
                if cid == "exp046" and grouping == "material":
                    assert np.array_equal(oof, inc[rs]["oof"])
                run[rs][grouping][cid] = dict(rk=rk, oof=oof, infos=infos, warn=wc, wall_s=time.time() - t0)
                fs = fit_summary(infos)
                conv = f" converged {fs['converged']}/{fs['n_fits']}" if "converged" in fs else ""
                log(f"STEP {'3' if grouping == 'material' else '4'} [{rs} {grouping}-GKF] {cid:6s} rkf_mean "
                    f"{rk.mean():.4f} +- {rk.std():.4f}; max OOF {oof.max():.0f}s;{conv} warnings {wc} "
                    f"({time.time() - t0:.0f}s)")
                if not np.isfinite(oof).all():
                    log(f"  NON-FINITE OOF: {rs} {grouping} {cid}")

    # ---------------- per-row-set reporting ----------------
    report = {}
    for rs in (CANON, NONCANON):
        d = D[rs]
        R = run[rs]["material"]
        F = run[rs]["family"]
        rk_inc = R["exp046"]["rk"]
        oofs = {cid: v["oof"] for cid, v in R.items()}
        rep = {"rows": int(len(d["y"])), "groups": int(len(np.unique(d["g"]))), "models": {}}
        for cid, v in R.items():
            s = stats(v["rk"], v["oof"], d["y"], d["g"], None if cid == "exp046" else rk_inc)
            s.update(family_gkf_mean=float(F[cid]["rk"].mean()), family_gkf_per_repeat=F[cid]["rk"].tolist(),
                     legacy=legacy(d, v["oof"]), max_oof=float(v["oof"].max()), max_oof_rep0=float(v["oof"][0].max()),
                     sse_share=sse_share(d, v["oof"]), warnings=v["warn"], fit=fit_summary(v["infos"]),
                     family_fit=fit_summary(F[cid]["infos"]), wall_s=round(v["wall_s"] + F[cid]["wall_s"], 1))
            if cid != "exp046":
                s["top_k_removal"] = topk(d, R["exp046"]["oof"], v["oof"])
            rep["models"][cid] = s
        rep["secondary_metrics"] = secondary(d, oofs)
        report[rs] = rep
        log(f"\n----- {rs} ({'CANONICAL' if rs == CANON else 'non-canonical, report only'}): "
            f"{rep['rows']} rows / {rep['groups']} materials -----")
        log(f"{'model':7s} {'rkf_mean':>9s} {'std':>7s} {'CI rep-0':>18s} {'dGKF':>8s} {'wins':>5s} {'famGKF':>7s} "
            f"{'R2@25':>7s} {'non25':>7s} {'maxOOF':>7s} {'conv':>6s}")
        for cid, s in rep["models"].items():
            conv = f"{s['fit']['converged']}/50" if "converged" in s["fit"] else "-"
            dl = f"{s['paired_delta']:+.4f}" if "paired_delta" in s else "-"
            wn = f"{s['paired_wins']}/10" if "paired_wins" in s else "-"
            log(f"{cid:7s} {s['rkf_mean']:9.4f} {s['rkf_std']:7.4f} [{s['ci'][0]:7.4f}, {s['ci'][1]:7.4f}] {dl:>8s} "
                f"{wn:>5s} {s['family_gkf_mean']:7.3f} {s['legacy']['r2_25_rep0']:7.3f} "
                f"{s['legacy']['r2_non25_pooled_rep0']:7.3f} {s['max_oof']:7.0f} {conv:>6s}")

        sm = rep["secondary_metrics"]["models"]
        log(f"  secondary metrics, 10-repeat means (wins vs exp046) -- {rep['secondary_metrics']['label']}")
        log("  " + f"{'model':7s} " + " ".join(f"{k:>13s}" for k in ("S25", "S50", "S75", "S50_75", "LOG_ALL", "LOG25")))
        for cid in sm:
            log("  " + f"{cid:7s} " + " ".join(
                f"{sm[cid][k]['mean']:7.3f}" + (f" ({sm[cid][k]['wins_vs_exp046']:2d})" if sm[cid][k]['wins_vs_exp046'] is not None else "     ")
                for k in ("S25", "S50", "S75", "S50_75", "LOG_ALL", "LOG25")))

    # ---------------- exit trigger ----------------
    for rs in report:
        for cid in FAMILY:
            if cid in report[rs]["models"] and report[rs]["models"][cid]["ci"][0] > SOTA:
                flags.append(f"EXIT trigger: {cid} ci_lo > 0.58 on {rs}")
                log(f"EXIT TRIGGER: {cid} repeat-0 ci_lo > {SOTA} on {rs}")

    # ---------------- outputs: result JSONs, npz ----------------
    dC = D[CANON]
    RC = report[CANON]["models"]
    RN = report[NONCANON]["models"]
    scored = [c for c in FAMILY if c in RC]
    nonfinite = [c for c in scored if not np.isfinite(run[CANON]["material"][c]["oof"]).all()]
    nonconv = [c for c in AFT_IDS if RC[c]["fit"]["converged"] < RC[c]["fit"]["n_fits"]]
    selectable = [c for c in SELECTABLE0 if c in scored and c not in nonconv]
    DATASET = ("data/v2/cone_replicates_fsri.csv, canonical S675 (D4b): early_stop False, spark_ignitor True, "
               "did_not_ignite False, minus the 6 operator-flagged replicates (tti_round3.json data_qa_decision): "
               "675 rows / 76 materials; PHYS left joins as exp046 (onset-bc). a1/a2 also train on the 42 "
               "did_not_ignite rows of U as right-censored rows (end_of_test_time_s), admitted per fold only when "
               "their material is not a test material; never scored")
    DESC = {"a0": ("lognormal AFT (control)", "a0: lognormal AFT on ignited rows only (ridge 0.01 on slopes, NLL/n_ign), "
                   "imputer+indicators+scaler fit on ignited training rows, L-BFGS-B from the OLS start, prediction "
                   "min(exp(mu + sigma^2/2), max training TTI); CONTROL, never selectable"),
            "a1": ("lognormal AFT + censored rows", "a1: a0 plus right-censored non-ignition rows (-log_ndtr terms), "
                   "same preprocessing, start, optimizer and cap"),
            "o1": ("HGB poisson + offset", "o1: HistGradientBoostingRegressor(loss='poisson', v5.HGB) with fixed offset "
                   "log o = -2 log(q/50): fit y*(q/50)^2 with sample_weight (50/q)^2, predict * (50/q)^2"),
            "o2": ("Avg(exp046, o1)", "o2: equal average of exp046 (Poisson GLM on PHYS) and o1, raw seconds"),
            "a2": ("lognormal AFT, learned q_cr", "a2: t = A(x)/(q - q_cr(x))^2, q_cr = 13.5*expit(c + Zs.d), "
                   "lognormal with censored rows, start q_cr 6.75, unbounded L-BFGS-B, capped lognormal mean")}
    SHORT = {"a0": "aft_control", "a1": "aft_censored", "o1": "hgbpois_offset", "o2": "avg_glm_offset", "a2": "aft_qcr"}
    inc_s = RC["exp046"]
    paths = {}

    def note_for(cid):
        s = RC[cid]
        conv = (f"converged {s['fit']['converged']}/{s['fit']['n_fits']} primary fits" if "converged" in s["fit"]
                else "convergence n/a (no AFT fit)")
        tk = s["top_k_removal"]
        n = (f"TTI round 3 {cid} on canonical S675 (675 rows, D4b); family-GKF {s['family_gkf_mean']:.3f} "
             f"(incumbent {inc_s['family_gkf_mean']:.3f}); legacy repeat-0 R2 at 25 kW/m2 {s['legacy']['r2_25_rep0']:.3f} "
             f"(incumbent {inc_s['legacy']['r2_25_rep0']:.3f}), non-25 rows pooled (15/35/50/75/100) "
             f"{s['legacy']['r2_non25_pooled_rep0']:.3f}; {conv}; max OOF {s['max_oof']:.0f}s")
        if cid in RN:
            nn = RN[cid]
            n += (f"; non-canonical S681 (report only): rkf_mean {nn['rkf_mean']:.4f} CI [{nn['ci'][0]:.4f}, "
                  f"{nn['ci'][1]:.4f}], {nn['paired_wins']}/10 vs exp046 on S681")
        n += (f"; top-5-removed gain {tk['top5_removed']['mean_gain']:+.4f} ({tk['top5_removed']['wins']}/10)")
        if tk["not_established"]:
            n += "; gain concentrated in <= 5 materials, not established"
        n += "; secondary metrics report-only, never compared with 0.58; gate: experiments/runs/tti_round3_tti_gate_v2.json"
        return n

    def house(cid, slug, round_config, model, hyper, extra=None):
        s = RC[cid]
        out = dict(slug=slug, target="Time to Ignition (s)", round_config=round_config, dataset=DATASET, features=PHYS,
                   model=model, hyperparameters=hyper + "; fixed a priori", code="data/v2/train_tti_round3.py",
                   n_rows=int(len(dC["y"])), n_groups=76, protocol="gkf_material5x10",
                   rkf_mean=round(s["rkf_mean"], 4), rkf_std=round(s["rkf_std"], 4), ci_lo=round(s["ci"][0], 4),
                   ci_hi=round(s["ci"][1], 4), paired_vs="exp046 re-scored on S675 (tti_r3_exp046_rows675)",
                   paired_delta=round(s["paired_delta"], 4), paired_wins=f"{s['paired_wins']}/10", sota=SOTA,
                   sota_source="pomazi_2026", status="ok", blocked_reason="", note=note_for(cid),
                   rkf_mean_full=s["rkf_mean"], ci_lo_full=s["ci"][0], ci_hi_full=s["ci"][1],
                   per_repeat_r2=s["per_repeat_r2"])
        out.update(extra or {})
        p = RESULTS / f"{slug}.json"
        p.write_text(json.dumps(out, indent=2) + "\n")
        paths[cid] = str(p)

    for cid in B4:
        house(cid, f"tti_r3_{cid}_{SHORT[cid]}", CFG_PATH, DESC[cid][0], DESC[cid][1])
    if tp_ok:
        import tabpfn as tp
        import torch as _t
        import sklearn
        setup = json.loads(SETUP_JSON.read_text())
        for cid in TABPFN_IDS:
            s = RC[cid]
            pl, ph = bootstrap_ci(dC["y"], run[CANON]["material"][cid]["oof"].mean(axis=0), dC["g"])
            extra = dict(pooled_ci_lo=round(float(pl), 4), pooled_ci_hi=round(float(ph), 4),
                         pooled_ci_construction="eval_protocol.bootstrap_ci of the repeat-mean OOF (seed 0, 2000 draws), "
                                                "as train_phrr_promote_c2.pooled_ci; report only",
                         tabpfn_version=tp.__version__, torch_version=_t.__version__, sklearn_version=sklearn.__version__,
                         checkpoint="tabpfn-v2-regressor.ckpt (HF Prior-Labs/TabPFN-v2-reg)", checkpoint_sha256=SHA256,
                         fitted_attributes=setup["fitted"], wall_time_s=s["wall_s"],
                         attribution="Built with PriorLabs-TabPFN")
            desc = {"p1": ("TabPFN V2", "TabPFN V2 alone on PHYS, raw seconds (create_default_for_version(V2, "
                                        "device='cpu', random_state=0), package defaults)"),
                    "p2": ("Avg(exp046, TabPFN V2)", "Avg([exp046 Poisson GLM, TabPFN V2]) raw seconds, equal weights")}[cid]
            house(cid, f"r9_tabpfn_tti_{cid}", R9_PATH, desc[0], desc[1], extra)
            j = json.loads(Path(paths[cid]).read_text())
            j["note"] = j["note"].replace(f"TTI round 3 {cid}", f"round-9 tti_arm {cid} (joined TTI round 3 family, D4d)") \
                + "; Built with PriorLabs-TabPFN"
            Path(paths[cid]).write_text(json.dumps(j, indent=2) + "\n")
        # p2 should equal the mean of exp046 and p1 when TabPFN is deterministic (diagnostic only)
        p2chk = float(np.abs(run[CANON]["material"]["p2"]["oof"]
                             - np.mean([inc[CANON]["oof"], run[CANON]["material"]["p1"]["oof"]], axis=0)).max())
        log(f"  diagnostic: max |p2 - mean(exp046, p1)| = {p2chk:.3g}")
        report[CANON]["p2_vs_mean_exp046_p1_maxabs"] = p2chk
    # re-baselined incumbent on S675
    e = RC["exp046"]
    inc_out = dict(slug="tti_r3_exp046_rows675", target="Time to Ignition (s)", round_config=CFG_PATH, dataset=DATASET.split("; PHYS")[0] + "; PHYS left joins as exp046 (onset-bc)",
                   features=PHYS, model="Poisson GLM", hyperparameters="v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)) "
                   "on PHYS, raw seconds (exp046 pipeline re-scored on S675, pre-declared re-baseline); fixed a priori",
                   code="data/v2/train_tti_round3.py", n_rows=675, n_groups=76, protocol="gkf_material5x10",
                   rkf_mean=round(e["rkf_mean"], 4), rkf_std=round(e["rkf_std"], 4), ci_lo=round(e["ci"][0], 4),
                   ci_hi=round(e["ci"][1], 4), paired_vs="", paired_delta=None, paired_wins="", sota=SOTA,
                   sota_source="pomazi_2026", status="ok", blocked_reason="",
                   note=(f"exp046 pipeline re-scored on canonical S675 (D4b; becomes the TTI number of record after the "
                         f"critic, whichever way it moves; exp046 0.3133 [0.0715, 0.4958] on 681 rows stays as history); "
                         f"family-GKF {e['family_gkf_mean']:.3f}; legacy repeat-0 R2 at 25 kW/m2 "
                         f"{e['legacy']['r2_25_rep0']:.3f}, non-25 rows pooled (15/35/50/75/100) "
                         f"{e['legacy']['r2_non25_pooled_rep0']:.3f}; max OOF {e['max_oof']:.0f}s; incumbent of TTI round 3"),
                   rkf_mean_full=e["rkf_mean"], ci_lo_full=e["ci"][0], ci_hi_full=e["ci"][1],
                   per_repeat_r2=e["per_repeat_r2"])
    (RESULTS / "tti_r3_exp046_rows675.json").write_text(json.dumps(inc_out, indent=2) + "\n")

    npz_path = RUNS / "tti_round3_tti_oof.npz"
    arrs = {f"oof_{c}": run[CANON]["material"][c]["oof"] for c in scored}
    np.savez(npz_path, y=dC["y"], groups=dC["g"].astype(str), row_id=dC["rid"].astype(str),
             oof_incumbent=inc[CANON]["oof"], inc_per_repeat_r2=inc[CANON]["rk"], inc_ci=np.array(inc[CANON]["ci"]),
             order=np.array(scored, dtype=str), selectable=np.array(selectable, dtype=str), **arrs)
    log(f"\nwrote {npz_path.relative_to(ROOT)} (order {scored}, selectable {selectable}; non-converged {nonconv})")

    # ---------------- step 5: adoption gate v2 (rule A) ----------------
    gate, verdict = None, {}
    if not tp_ok:
        verdict = dict(decision="PARKED: family incomplete (TabPFN arm blocked); no gate run", adopt=False)
    elif nonfinite:
        flags.append(f"R2: non-finite OOF in {nonfinite} on primary rows; no adoption decision")
        verdict = dict(decision="FLAG R2: non-finite OOF, no adoption decision", adopt=False)
    else:
        log("\nSTEP 5: adoption_gate_v2 (rule A, D1 approved) over the family " + str(scored))
        gate = adoption_gate.run_npz(str(npz_path), paths, str(RUNS / "tti_round3_tti_gate_v2.json"))
        best = gate["best"]
        r1_fail = best == "a1" and RC["a0"]["rkf_mean"] >= RC["a1"]["rkf_mean"]
        adopt = bool(gate["adopt_if_critic_passes"] and not r1_fail)
        lb = gate["per_candidate"][best]["lb_maxT"]
        verdict = dict(best=best, G1=gate["G1"], G2=gate["G2"], G3=gate["G3"], R1_blocks=r1_fail, adopt=adopt,
                       K=gate["K"], q=gate["q"], lb_best=lb,
                       decision=(f"ADOPT {best} pending critic (q {gate['q']:.3f}, K {gate['K']}, LB {lb:+.4f})" if adopt
                                 else f"NOT adopted: incumbent (exp046 re-scored on S675) stays; selected {best} failed "
                                      + ", ".join(k for k in ("G1", "G2", "G3") if not gate[k])
                                      + (" and R1" if r1_fail else "")
                                      + (f"; gain not established after multiplicity adjustment (LB = {lb:+.4f}, K = {gate['K']})"
                                         if not gate["G3"] else "")))
        log("VERDICT: " + verdict["decision"])
        gl = (f"; gate_v2 (K {gate['K']}, q {gate['q']:.3f}): selected {best} -> "
              + ("ADOPT pending critic" if adopt else "not adopted"))
        for cid in scored:
            j = json.loads(Path(paths[cid]).read_text())
            j["note"] += gl + (f" (this candidate: G1 {gate['G1']}, G2 {gate['G2']}, G3 {gate['G3']}, LB {lb:+.4f})"
                               if cid == best else "")
            Path(paths[cid]).write_text(json.dumps(j, indent=2) + "\n")

    # ---------------- step 6: diagnostics, non-canonical, flags ----------------
    diag = dict(config=CFG_PATH, tti_arm_config=R9_PATH, canonical=CANON,
                generated_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                self_tests=selftests, amendments=amend, tabpfn=dict(ok=tp_ok, msg=tp_msg),
                incumbent_S681_repro=dict(rkf_mean=float(inc["S681"]["rk"].mean()), ci=list(map(float, inc["S681"]["ci"]))),
                incumbent_S675=dict(rkf_mean=float(inc["S675"]["rk"].mean()), ci=list(map(float, inc["S675"]["ci"]))),
                selectable=selectable, non_converged=nonconv, nonfinite=nonfinite, verdict=verdict,
                p2_vs_mean_exp046_p1_maxabs=report[CANON].get("p2_vs_mean_exp046_p1_maxabs"),
                fit_diagnostics={rs: {c: dict(primary=report[rs]["models"][c]["fit"], family=report[rs]["models"][c]["family_fit"],
                                              max_oof=report[rs]["models"][c]["max_oof"], warnings=report[rs]["models"][c]["warnings"])
                                      for c in report[rs]["models"]} for rs in report},
                censored_cells_only_for_a1_a2=dict(rows=CANON, folds="repeat 0, material GKF", table=cells,
                                                  note="these rows are never scored; table only"),
                secondary_metrics=report[CANON]["secondary_metrics"],
                top_k_removal={c: report[CANON]["models"][c]["top_k_removal"] for c in scored},
                sse_share={c: report[CANON]["models"][c]["sse_share"] for c in ["exp046"] + scored},
                primary=dict(models={c: {k: v for k, v in s.items() if k not in ("fit", "family_fit")}
                                     for c, s in report[CANON]["models"].items()}),
                runtime_s=round(time.time() - T_START))
    (RUNS / "tti_round3_fit_diagnostics.json").write_text(json.dumps(diag, indent=2, default=float) + "\n")
    nonc = dict(config=CFG_PATH, rows=NONCANON, label="non-canonical row set S681 (681 rows): report only; never selects, "
                "adopts, demotes or changes record wording (tti_round3.json adoption.non_canonical_rows). No gate run "
                "on these rows (adoption_gate_v2: one gate run per target per round).",
                incumbent="exp046 (record 0.3133 [0.0715, 0.4958])", models=report[NONCANON]["models"],
                secondary_metrics=report[NONCANON]["secondary_metrics"])
    (RUNS / "tti_round3_noncanonical.json").write_text(json.dumps(nonc, indent=2, default=float) + "\n")
    log(f"wrote result JSONs {sorted(Path(p).name for p in paths.values())} + tti_r3_exp046_rows675.json; "
        f"runs/tti_round3_fit_diagnostics.json, runs/tti_round3_noncanonical.json")
    if flags:
        log("FLAGS: " + " | ".join(flags))
    log(f"runtime {time.time() - T_START:.0f}s")
    return verdict, flags


if __name__ == "__main__":
    main()
