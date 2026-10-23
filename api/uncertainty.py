"""Paper-grouped conformal intervals, an out-of-domain flag and a CFRS interval for engine predictions.

Method and coverage: context/research/UNCERTAINTY.md; pre-declared in experiments/configs/conformal_round1.json.
The artifact api/uncertainty.joblib is written by api/build_uncertainty.py from the engine's training snapshot only
(wave 2 is sealed and never touched).

Per target t, with mu = mean of the engine's three member predictions on the log1p scale:
  interval_L = clip(expm1(mu -/+ q_L[t] * sigma_t(x)), ymin, ymax)
sigma_t(x) = 1 (method A_abs), ensemble spread + median (B_spread) or nearest-neighbour distance + median (C_knn),
whichever the declared rule picked for that target.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api import model

E = model.E
T = E.T
ARTIFACT = Path(__file__).resolve().parent / "uncertainty.joblib"
FLUX_RANGE = (10.0, 95.0)  # training range of heat_flux_kWm2
Z90 = 1.6448536269514722   # normal 95th percentile: sigma_draw = h90 / Z90
N_DRAWS = 4000


def features(df, B):
    """Engine feature matrix for a frame from E._frame(); same block as engine/dravix_engine.py predict()
    (re-implemented, not imported: engine/ is not edited). api/test_uncertainty.py checks it against the engine."""
    T.TOPN = 0  # the engine sets the same global before build_X at predict time
    X = T.build_X(df, B["cmap"], B["mat_cats"], B["form_cats"])
    idx = {k: i for i, k in enumerate(B["vocab"])}
    A = np.zeros((len(df), len(idx)))
    for i, comp in enumerate(df["components"]):
        cs = T.parse_components(comp)
        units = {c[3] for c in cs if c[1] != "matrix" and not np.isnan(c[2])}
        tot = 100.0 + sum(c[2] for c in cs if c[1] != "matrix" and not np.isnan(c[2])) if units == {"phr"} else None
        for name, role, v, u in cs:
            j = idx.get(T._norm(name))
            if j is not None and role != "matrix" and not np.isnan(v):
                A[i, j] = 100.0 * v / tot if tot else (v if u == "wt%" else 1.0)
    X = pd.concat([X, pd.DataFrame(A, columns=[f"a_{i}" for i in range(len(idx))], index=df.index)], axis=1)
    return X.reindex(columns=B["columns"], fill_value=0.0).to_numpy(float)


def n_additives(components) -> int:
    return sum(1 for c in T.parse_components(components) if c[1] != "matrix")


def members(mdl, X):
    """(3, n) log1p-scale predictions of the engine's HGB, ExtraTrees and Ridge members."""
    return np.array([m.predict(X) for m in mdl.fitted_])


def scale(X, sp):
    """Standardized feature space for distances (label-free; fit on the training snapshot)."""
    X = np.where(np.isnan(X), sp["med"], X)
    return ((X - sp["med"]) / sp["std"])[:, sp["keep"]].astype(np.float32)


def nn_dist(Q, R, chunk=2048):
    """Euclidean distance from each row of Q to its nearest row of R."""
    r2 = (R.astype(np.float64) ** 2).sum(1)
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk].astype(np.float64)
        d2 = (q ** 2).sum(1)[:, None] + r2[None, :] - 2.0 * q @ R.T.astype(np.float64)
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def sigma(method, sd, d, med):
    if method == "A_abs":
        return np.ones_like(sd)
    return (sd if method == "B_spread" else d) + med


def cfrs_draws(mu, h90, ymin, ymax, Z, weights):
    """CFRS (0-100) for each Monte-Carlo draw. mu, h90, ymin, ymax: per-target arrays (engine output order);
    Z: (n_draws, n_targets) correlated standard normals; weights: per-target CFRS weights (sum 1)."""
    from cfrps import subscore  # engine/score is on sys.path via api.main; imported lazily for tests
    vals = np.clip(np.expm1(mu[None, :] + Z * (h90 / Z90)[None, :]), ymin, ymax)
    return 100.0 * sum(w * subscore(k, vals[:, j]) for j, (k, w) in enumerate(weights.items()))


def cfrs_interval(point, draws):
    """{interval_80, interval_90, interval95, sigma}; percentile bounds widened to contain the point score."""
    out = {}
    for key, (a, b) in {"interval_80": (10, 90), "interval_90": (5, 95), "interval95": (2.5, 97.5)}.items():
        lo, hi = np.percentile(draws, [a, b])
        out[key] = [round(min(float(lo), point), 2), round(max(float(hi), point), 2)]
    out["sigma"] = float(draws.std())
    return out


class Uncertainty:
    def __init__(self, path=ARTIFACT):
        self.U = joblib.load(path)
        L = np.linalg.cholesky(self.U["corr"])
        self.Z = np.random.default_rng(0).standard_normal((N_DRAWS, len(self.U["targets"]))) @ L.T  # common random numbers

    def meta(self):
        U = self.U
        return {"version": U["version"], "levels": [0.8, 0.9], "method": U["method"],
                "coverage_validated": U["coverage"], "grouped_by": "paper (DOI)"}

    def annotate(self, rows, B):
        """rows: engine input dicts. Returns per row {target: {interval_80, interval_90, ood, ood_reasons}} and
        per row (mu, h90) arrays in self.U['targets'] order for cfrs()."""
        U = self.U
        df = E._frame(rows)
        X = features(df, B)
        Zq = scale(X, U["space"])
        common = []
        for i, r in enumerate(rows):
            why = []
            if r["matrix_canonical"] not in B["mat_cats"]:
                why.append("matrix not in training data (fewer than 3 papers)")
            if r["material_form"] not in B["form_cats"]:
                why.append("material form not in training data")
            if n_additives(r["components"]) > U["max_additives"]:
                why.append(f"more than {U['max_additives']} additives (99th percentile of training)")
            q = r.get("heat_flux_kWm2")
            if q is None or not FLUX_RANGE[0] <= q <= FLUX_RANGE[1]:
                why.append("heat flux missing or outside 10-95 kW/m2")
            common.append(why)
        out = [{} for _ in rows]
        mu_all = np.zeros((len(rows), len(U["targets"])))
        h90_all = np.zeros_like(mu_all)
        for j, t in enumerate(U["targets"]):
            b = B["models"][t]
            m = members(b["model"], X)
            mu, sd = m.mean(0), m.std(0)
            d = nn_dist(Zq, U["space"]["ref"][U["mask"][t]])
            s = sigma(U["method"][t], sd, d, U["med"][t])
            h80, h90 = U["q80"][t] * s, U["q90"][t] * s
            mu_all[:, j], h90_all[:, j] = mu, h90
            for i in range(len(rows)):
                band = lambda h: [round(float(np.clip(np.expm1(mu[i] + k * h[i]), b["ymin"], b["ymax"])), 2) for k in (-1, 1)]  # noqa: E731
                why = common[i] + (["far from all training formulations (distance > 95th percentile)"]
                                   if d[i] > U["ood_dist"][t] else [])
                out[i][t] = {"interval_80": band(h80), "interval_90": band(h90), "ood": bool(why), "ood_reasons": why}
        return out, mu_all, h90_all

    def cfrs(self, point, mu, h90, B, weights_used):
        """CFRS interval for one row; weights_used = cfrps()['weights_used'] (renormalized over present inputs)."""
        U = self.U
        idx = [U["targets"].index(k) for k in weights_used]
        ymin = np.array([B["models"][k]["ymin"] for k in weights_used])
        ymax = np.array([B["models"][k]["ymax"] for k in weights_used])
        draws = cfrs_draws(mu[idx], h90[idx], ymin, ymax, self.Z[:, idx], weights_used)
        return cfrs_interval(point, draws)
