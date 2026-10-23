#!/usr/bin/env python3
"""litcone_gt_round1 -- follows experiments/configs/litcone_gt_round1.json exactly (nothing tuned).

Literature cone-calorimeter set (Tran et al. 2025 style) scored under three protocols:
  P1 their protocol (k-fold ensemble scored on training data; NOT held-out evidence)
  P2 repeated random 5-fold CV (rows of one paper on both sides)
  P3 repeated GroupKFold by paper (Dravix-style headline)

  OMP_NUM_THREADS=1 python3 data/v2/train_litcone_gt_round1.py | tee experiments/runs/litcone_gt_round1.log
  --smoke : replace every target by random noise (pipeline test without looking at any real score)
"""
import argparse
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402

CLIP = False
PHYS = False
TOPN = 0
PARTS = ROOT / "data/v2/cone_lit_parts"
CLASS_MAP = PARTS / "component_class_map.csv"
SNAP = ROOT / "data/v2/cone_lit_benchmark_v1.csv"
OUT_JSON = ROOT / "experiments/runs/litcone_gt_round1.json"
TARGETS = {"TTI": "tti_s", "pHRR": "phrr_kWm2", "THR": "thr_MJm2", "avg_HRR": "avg_hrr_kWm2"}
GT = {"TTI": 0.95, "pHRR": 0.94, "avg_HRR": 0.96, "THR": 0.97}
EXCLUDED_FORMS = {"textile", "coated_substrate"}
CLASSES = ["P_FR", "N_FR", "APP_IFR", "halogen_FR", "Sb2O3", "metal_hydroxide", "EG", "nanoclay", "Si", "boron",
           "carbon_nano", "metal_oxide_LDH", "glass_fibre", "carbon_fibre", "natural_fibre", "mineral_filler", "other_FR",
           "other_filler", "other"]
NUM = ["heat_flux_kWm2", "thickness_mm", "density_kgm3", "area_cm2", "fr_total_wtpct", "filler_total_wtpct"]


# ---------------------------------------------------------------- data
def _num(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else np.nan


def load_rows():
    frames = []
    for f in sorted(glob.glob(str(PARTS / "oa_staging/rows_oa_*.csv"))):
        d = pd.read_csv(f, dtype=str); d["source_set"] = "oa"; frames.append(d)
    for pat in ("lib_staging/rows_lib_*.csv", "lib_staging/rows_libpdf_*.csv", "lpc2_staging/rows_lpc2_*.csv"):
        for f in sorted(glob.glob(str(PARTS / pat))):
            d = pd.read_csv(f, dtype=str); d["source_set"] = "lib"; frames.append(d)
    lpc = pd.read_csv(PARTS / "cone_lit_merged_raw.csv", dtype=str)
    lpc = lpc.drop_duplicates(subset=[c for c in lpc.columns if c not in ("row_id", "pdf_stem", "batch")])
    lpc = lpc.rename(columns={"polymer_system": "matrix_raw"})
    lpc["components"] = lpc["fr_summary"].fillna("") + "|FR|" + lpc["fr_loading_as_printed"].fillna("")
    lpc["matrix_canonical"] = lpc["matrix_raw"].map(_canon_matrix)
    lpc["material_form"] = np.where(lpc["matrix_raw"].str.contains("foam", case=False, na=False), "foam", "bulk")
    lpc["is_control"] = lpc["fr_summary"].str.contains(r"^none|neat|pure|blank|control", case=False, na=False).astype(int).astype(str)
    lpc["pmcid"] = "LPC:" + lpc["pdf_stem"]
    lpc["source_set"] = "lpc"
    frames.append(lpc)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["extraction"].fillna("").str.lower() != "figure"]
    for c in NUM + list(TARGETS.values()) + ["is_control"]:
        df[c] = df[c].map(_num)
    df["doi"] = df["doi"].fillna("").str.lower().str.strip().str.replace("https://doi.org/", "", regex=False)
    df["group"] = np.where(df["doi"] != "", df["doi"], df["pmcid"].fillna(""))
    key = ["group", "formulation_label", "heat_flux_kWm2", "thickness_mm"]
    df["_pri"] = df["source_set"].map({"oa": 0, "lib": 1, "lpc": 2})
    # paper-level dedupe (amendment 2): a DOI extracted from several sources keeps only its highest-priority source
    # (and, within that source, one extraction file), so no paper appears twice under different formulation labels
    df["_file"] = df["pmcid"].fillna("")
    best = df[df["doi"] != ""].sort_values(["_pri", "_file"]).groupby("doi")[["_pri", "_file"]].first()
    has = df["doi"] != ""
    keep = ~has | ((df["_pri"] == df["doi"].map(best["_pri"])) & (df["_file"] == df["doi"].map(best["_file"])))
    df = df[keep]
    df = df.sort_values("_pri").drop_duplicates(subset=key, keep="first").drop(columns=["_pri", "_file"])
    return df.reset_index(drop=True)


MATRIX_RX = [("RPUF", r"rigid.*(pu|polyurethane).*foam|rpuf"), ("FPUF", r"flexible.*(pu|polyurethane).*foam|fpuf"),
             ("EP", r"epoxy|\bep\b"), ("PP", r"polypropylene|\bpp\b"), ("HDPE", r"hdpe|high.density polyethylene"),
             ("LDPE", r"ldpe|low.density polyethylene"), ("EVA", r"\beva\b|ethylene.vinyl acetate"),
             ("ABS", r"\babs\b"), ("PC", r"polycarbonate|\bpc\b"), ("PLA", r"pla\b|polylactic|poly\(lactic"),
             ("PA6", r"pa ?6\b|polyamide ?6\b|nylon ?6\b"), ("PS", r"polystyrene|\bps\b"), ("TPU", r"\btpu\b")]


def _canon_matrix(s):
    t = str(s).lower()
    for k, rx in MATRIX_RX:
        if re.search(rx, t):
            return k
    return "other"


def parse_components(comp):
    """'name|role|amount; ...' -> list of (name, role, value, unit)."""
    out = []
    for part in str(comp or "").split(";"):
        bits = [b.strip() for b in part.split("|")]
        if len(bits) < 3 or not bits[0]:
            continue
        name, role, amt = bits[0], bits[1].lower(), "|".join(bits[2:])
        v = _num(amt)
        unit = "wt%" if re.search(r"wt|%|mass", amt, re.I) and not re.search(r"phr", amt, re.I) else \
            ("phr" if re.search(r"phr", amt, re.I) else ("g" if re.search(r"\bg\b", amt) else ""))
        out.append((name, role, v, unit))
    return out


def class_features(df, cmap):
    feats = np.zeros((len(df), len(CLASSES)))
    ncomp = np.zeros(len(df))
    for i, comp in enumerate(df["components"].tolist()):
        cs = parse_components(comp)
        ncomp[i] = sum(1 for c in cs if c[1] != "matrix")
        units = {c[3] for c in cs if c[1] != "matrix" and not np.isnan(c[2])}
        if units == {"phr"}:
            tot = 100.0 + sum(c[2] for c in cs if c[1] != "matrix" and not np.isnan(c[2]))
            conv = lambda v: 100.0 * v / tot  # noqa: E731
        elif units == {"wt%"}:
            conv = lambda v: v  # noqa: E731
        elif units == {"g"}:
            tot = sum(c[2] for c in cs if not np.isnan(c[2]))
            conv = lambda v: 100.0 * v / tot if tot else np.nan  # noqa: E731
        else:
            conv = None
        for name, role, v, unit in cs:
            k = cmap.get(name.strip().lower(), "other")
            if role in ("matrix", "curing_agent") or k == "not_additive" or conv is None or np.isnan(v):
                continue
            j = CLASSES.index(k) if k in CLASSES else CLASSES.index("other")
            feats[i, j] += conv(v)
    out = pd.DataFrame(feats, columns=[f"c_{c}" for c in CLASSES], index=df.index)
    out["n_components"] = ncomp
    return out


def build_X(df, cmap, mat_cats, form_cats):
    X = pd.DataFrame(index=df.index)
    for c in NUM + ["is_control"]:
        X[c] = df[c]
    X["log_thickness"] = np.log(df["thickness_mm"].where(df["thickness_mm"] > 0))
    if PHYS:
        q = df["heat_flux_kWm2"].where(df["heat_flux_kWm2"] > 0)
        X["log_q"] = np.log(q)
        X["log_thick_over_q"] = X["log_thickness"] - X["log_q"]
        X["inv_q2"] = 1.0 / q ** 2
    for m in mat_cats:
        X[f"m_{m}"] = (df["matrix_canonical"] == m).astype(float)
    X["m_other"] = (~df["matrix_canonical"].isin(mat_cats)).astype(float)
    for f in form_cats:
        X[f"f_{f}"] = (df["material_form"] == f).astype(float)
    X = pd.concat([X, class_features(df, cmap)], axis=1)
    if TOPN:
        X = pd.concat([X, specific_features(df, cmap)], axis=1)
    return X


def _norm(n):
    return re.sub(r"\s*\(.*?\)", "", str(n).lower()).strip()


def specific_features(df, cmap):
    """wt% of each of the TOPN additive names (normalised, by number of papers using them); names only."""
    papers = {}
    parsed = [parse_components(c) for c in df["components"]]
    for g, cs in zip(df["group"], parsed):
        for name, role, v, u in cs:
            if role not in ("matrix", "curing_agent") and cmap.get(name.strip().lower()) != "not_additive":
                papers.setdefault(_norm(name), set()).add(g)
    top = sorted(papers, key=lambda k: -len(papers[k]))[:TOPN]
    idx = {k: i for i, k in enumerate(top)}
    F = np.zeros((len(df), len(top)))
    for i, cs in enumerate(parsed):
        units = {c[3] for c in cs if c[1] != "matrix" and not np.isnan(c[2])}
        tot = 100.0 + sum(c[2] for c in cs if c[1] != "matrix" and not np.isnan(c[2])) if units == {"phr"} else None
        for name, role, v, u in cs:
            j = idx.get(_norm(name))
            if j is None or np.isnan(v) or role == "matrix":
                continue
            F[i, j] = 100.0 * v / tot if tot else (v if u == "wt%" else 1.0)
    return pd.DataFrame(F, columns=[f"a_{i}" for i in range(len(top))], index=df.index)


def control_feature(df, tcol):
    """Same paper's is_control value of the same target at the same heat flux (thickness-matched if several)."""
    ctrl = df[df["is_control"] == 1]
    out = np.full(len(df), np.nan)
    for i, (g, q, th) in enumerate(zip(df["group"], df["heat_flux_kWm2"], df["thickness_mm"])):
        c = ctrl[(ctrl["group"] == g) & (ctrl["heat_flux_kWm2"] == q) & ctrl[tcol].notna()]
        if len(c) > 1 and not np.isnan(th):
            c2 = c[c["thickness_mm"] == th]
            c = c2 if len(c2) else c
        if len(c):
            out[i] = np.log1p(c[tcol].astype(float).mean())
    out[(df["is_control"] == 1).to_numpy()] = np.nan  # amendment 7: no self-leak on control rows
    return out


# ---------------------------------------------------------------- models
def imp():
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


class Avg(BaseEstimator, RegressorMixin):
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.fitted_], axis=0)


def primary():
    return Avg([make_pipeline(imp(), HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                   min_samples_leaf=5, random_state=0)),
                make_pipeline(imp(), ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)),
                make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0))])


def gt_protocol_et():
    """amendment 8: single declared model for the competitor's (training-data) protocol."""
    return make_pipeline(imp(), ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=0, n_jobs=-1))


def gpr():
    return make_pipeline(imp(), StandardScaler(), _ARDGPR())


class _ARDGPR(BaseEstimator, RegressorMixin):
    """GPR whose RBF gets one length scale per input column (ARD), sized at fit time."""
    def fit(self, X, y):
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2)
        self.m_ = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


def mlp():
    return Avg([make_pipeline(imp(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                                    early_stopping=True, max_iter=2000, random_state=s))
                for s in range(5)])


def fit_pred(model, Xtr, ytr, Xte):
    m = clone(model).fit(Xtr, np.log1p(ytr))
    p = np.expm1(m.predict(Xte))
    return np.clip(p, ytr.min(), ytr.max()) if CLIP else p


# ---------------------------------------------------------------- protocols
def p1(model, X, y):
    kf = KFold(5, shuffle=True, random_state=0)
    preds = [fit_pred(model, X[tr], y[tr], X) for tr, _ in kf.split(X)]
    ens = np.mean(preds, axis=0)
    ins = fit_pred(model, X, y, X)
    return r2_score(y, ens), r2_score(y, ins)


def p2(model, X, y, repeats=10):
    rs, oof0 = [], None
    for r in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            oof[te] = fit_pred(model, X[tr], y[tr], X[te])
        rs.append(r2_score(y, oof))
        if r == 0:
            oof0 = oof
    rng = np.random.default_rng(0)
    boot = [r2_score(y[i], oof0[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    return np.array(rs), oof0, tuple(np.percentile(boot, [2.5, 97.5]))


def p3(model, X, y, g):
    from sklearn.model_selection import GroupKFold
    oof = np.empty((10, len(y)))
    for r in range(10):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, g):
            assert set(g[tr]).isdisjoint(g[te])
            oof[r, te] = fit_pred(model, X[tr], y[tr], X[te])
    rk = np.array([r2_score(y, o) for o in oof])
    lo, hi = bootstrap_ci(y, oof[0], g)
    return rk, oof[0], (lo, hi)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--include-excluded-forms", action="store_true", help="sensitivity: keep textile/coated rows")
    ap.add_argument("--only", default="", help="comma list of model ids")
    ap.add_argument("--v3", action="store_true", help="amendment 5: v2 + ignition-physics flux/thickness terms")
    ap.add_argument("--v2", action="store_true", help="amendment 4: top-100 additive features + clip to training range")
    a = ap.parse_args()
    global CLIP, TOPN, PHYS
    if a.v3:
        a.v2, PHYS = True, True
    if a.v2:
        CLIP, TOPN = True, 100
    t0 = time.time()
    df = load_rows()
    n_all = len(df)
    if not a.include_excluded_forms:
        df = df[~df["material_form"].fillna("").isin(EXCLUDED_FORMS)].reset_index(drop=True)
    cm = pd.read_csv(CLASS_MAP, dtype=str)
    cmap = dict(zip(cm["component_name"].str.strip().str.lower(), cm["class"]))
    paper_per_mat = df.groupby("matrix_canonical")["group"].nunique()
    mat_cats = sorted(paper_per_mat[paper_per_mat >= 3].index.difference(["other"]))
    form_cats = sorted(df["material_form"].dropna().unique())
    X_all = build_X(df, cmap, mat_cats, form_cats)
    snap = df.copy()
    snap.to_csv(SNAP, index=False)
    sha = hashlib.sha256(SNAP.read_bytes()).hexdigest()
    print(f"litcone_gt_round1 ({'SMOKE: random targets' if a.smoke else 'REAL'}) -- rows after merge/dedupe {n_all}, "
          f"after form filter {len(df)}, papers {df['group'].nunique()}; snapshot {SNAP.name} sha256 {sha[:16]}")
    print(f"matrix categories (>=3 papers): {mat_cats}; forms: {form_cats}; features {X_all.shape[1]}")
    MODELS = {"gt_protocol_et2": gt_protocol_et, "gt_protocol_et": gt_protocol_et, "primary": primary, "gt_replica_gpr": gpr, "gt_replica_mlp": mlp, "variant_C": primary}
    if a.only:
        MODELS = {k: v for k, v in MODELS.items() if k in a.only.split(",")}
    res = {}
    rng = np.random.default_rng(42)
    for tname, tcol in TARGETS.items():
        m = (df[tcol] > 0) & df["heat_flux_kWm2"].notna()
        sub = df[m].reset_index(drop=True)
        y = sub[tcol].to_numpy(float)
        if a.smoke:
            y = np.exp(rng.normal(4, 1, len(y)))
        g = sub["group"].to_numpy()
        Xb = X_all[m.to_numpy()].reset_index(drop=True)
        print(f"\n=== {tname}: {len(y)} rows / {len(set(g))} papers; competitor {GT[tname]} (training-data fit, "
              f"{'n/a' if a.smoke else 'their rows not released'})")
        if len(y) < 30 or len(set(g)) < 5:
            print("  too few rows/papers; not scored"); continue
        res[tname] = {"n_rows": int(len(y)), "n_papers": int(len(set(g)))}
        for mid, mk in MODELS.items():
            X = Xb.copy()
            if mid in ("variant_C", "gt_protocol_et", "gt_protocol_et2"):
                X["ctrl_log"] = control_feature(sub, tcol)
            yy, gg = y, g
            if mid == "gt_protocol_et2":
                X["paper_code"] = pd.factorize(g)[0]
                key = X.round(6).astype(str).agg("|".join, axis=1)
                agg = pd.DataFrame({"k": key, "y": y, "g": g}).groupby("k", sort=False).agg(y=("y", "mean"), g=("g", "first"))
                X = X.loc[~key.duplicated()].reset_index(drop=True)
                yy, gg = agg["y"].to_numpy(float), agg["g"].to_numpy()
                print(f"  gt_protocol_et2: {len(y)} rows -> {len(yy)} after averaging identical-input rows")
            X = X.to_numpy(float)
            y_, g_ = y, g
            y, g = yy, gg
            t = time.time()
            e1, i1 = p1(mk(), X, y)
            r2s, oof2, ci2 = p2(mk(), X, y)
            r3s, oof3, ci3 = p3(mk(), X, y, g)
            lg = lambda o: r2_score(np.log(y), np.log(np.clip(o, 1e-6, None)))  # noqa: E731
            res[tname][mid] = dict(P1_kfold_ensemble_on_training=e1, P1_in_sample=i1,
                                   P2_mean=float(r2s.mean()), P2_std=float(r2s.std()), P2_ci_rep0=list(map(float, ci2)),
                                   P3_mean=float(r3s.mean()), P3_std=float(r3s.std()), P3_ci_rep0=list(map(float, ci3)),
                                   P2_log_r2_rep0=float(lg(oof2)), P3_log_r2_rep0=float(lg(oof3)),
                                   P2_mae_rep0=float(mean_absolute_error(y, oof2)),
                                   P3_mae_rep0=float(mean_absolute_error(y, oof3)), runtime_s=round(time.time() - t))
            y, g = y_, g_
            r = res[tname][mid]
            print(f"  {mid:15s} P1 [THEIR PROTOCOL: computed on training data; not held-out evidence] ensemble "
                  f"{e1:.3f} in-sample {i1:.3f} | P2 random 5x10 {r['P2_mean']:.3f} +- {r['P2_std']:.3f} "
                  f"CI [{ci2[0]:.3f}, {ci2[1]:.3f}] | P3 paper-grouped 5x10 {r['P3_mean']:.3f} +- {r['P3_std']:.3f} "
                  f"CI [{ci3[0]:.3f}, {ci3[1]:.3f}]  ({r['runtime_s']}s)")
    out = {"config": "experiments/configs/litcone_gt_round1.json", "smoke": a.smoke, "snapshot_sha256": sha,
           "rows_after_dedupe": n_all, "rows_scored_pool": len(df), "results": res, "runtime_s": round(time.time() - t0)}
    if not a.smoke:
        (OUT_JSON.with_name(OUT_JSON.stem + (("_" + a.only.replace(",", "_")) if a.only.startswith("gt_protocol_et") else "_v3" if a.v3 else "_v2" if a.v2 else "") + ".json")).write_text(json.dumps(out, indent=2))
    print(f"\nruntime {time.time() - t0:.0f}s" + ("" if a.smoke else f"; wrote {OUT_JSON}"))


if __name__ == "__main__":
    main()
