#!/usr/bin/env python3
"""A4: group-conformal 90% prediction intervals for the records (report-only).

Follows experiments/configs/conformal_intervals.json exactly. Nothing here changes an R2, a CI, a leaderboard row or a
number of record, and nothing is written to experiments/results/.

  OMP_NUM_THREADS=1 python3 data/v2/conformal_intervals.py --target {pHRR,TTI,CR,FIGRA,THR} [--tti-rows S675]

Per target (one per run):
  1. Step 1: reproduce the record with eval_protocol.repeated_group_kfold + bootstrap_ci (mean and both CI bounds
     within 0.002 of the record in the config), else BLOCKED (exit 2, nothing else runs).
  2. Nested repeated GroupKFold 5x10: outer GroupKFold(5, shuffle, r) gives test groups; inner GroupKFold(5, shuffle, r)
     on the outer-train rows gives 5 inner models and inner OOF scores. Each outer-test row gets the group-weighted
     Mondrian CV+ interval (P) and the report-only variants V1 (no Mondrian), V2 (row weights), V3 (TTI Pearson score)
     and CQR (HGB quantile models).
  3. Writes experiments/runs/conformal/<t>_summary.json and <t>_rows.csv. --target pHRR also writes the demo artifact
     models/v2/phrr_hgb_v4_conformal.joblib (deployed CV+ on the repeat-0 folds).
LOI (exp115) runs through data/v2/conformal_loi_snapshot.py, which imports run() from this file.
"""
import argparse
import json
import os
import platform
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import conformal as C  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402

CONFIG = "experiments/configs/conformal_intervals.json"
REPEATS, K = 10, 5
ART = os.path.join(ROOT, "models", "v2", "phrr_hgb_v4_conformal.joblib")
BUNDLE = os.path.join(ROOT, "models", "v2", "phrr_hgb_v4.joblib")


def _ident(v):
    return np.asarray(v, float)


def say(*a):
    print(*a, flush=True)


# fit scale -> score scale -> raw units, per "records" in the config
SCALES = {
    "log1p": NS(fit_y=np.log1p, pred_to_score=_ident, y_to_score=np.log1p, back=np.expm1, log=True,
                  text="s = log1p(y); s_hat = model output (fitted on log1p(y)); back-transform expm1"),
    "log_glm": NS(fit_y=_ident, pred_to_score=np.log, y_to_score=np.log, back=np.exp, log=True,
                    text="s = ln(y), s_hat = ln(mu_hat) (Poisson GLM log link, fitted on raw seconds); back-transform exp"),
    "identity": NS(fit_y=_ident, pred_to_score=_ident, y_to_score=_ident, back=_ident, log=False,
                     text="s = y (raw units); back-transform identity"),
}


def qmodel(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=300,
                                         max_leaf_nodes=15, min_samples_leaf=10, random_state=0)


def record_cfg(key):
    with open(os.path.join(ROOT, CONFIG)) as f:
        cfg = json.load(f)
    return cfg["records (exact code paths; data prep imported, never re-implemented)"][key]


def call_capturing(fn, names):
    """Run a record data function unchanged and keep some of its locals (to read row identifiers it does not return)."""
    cap = {}

    def prof(frame, event, arg):
        if event == "return" and frame.f_code is fn.__code__:
            cap.update({n: frame.f_locals[n] for n in names if n in frame.f_locals})

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        out = fn()
    finally:
        sys.setprofile(old)
    missing = [n for n in names if n not in cap]
    assert not missing, missing
    return out, cap


# ---------------------------------------------------------------- record specs (data prep imported)
def spec_phrr():
    import train_blend_round1 as B
    D, loc = call_capturing(B.data_phrr, ["df"])
    df = loc["df"]
    assert np.array_equal(df[D["feats"]].to_numpy(float), D["X"], equal_nan=True)
    return dict(key="pHRR", tag="phrr", model=D["inc"], X=D["X"], y=D["y"], groups=D["g"], feats=D["feats"],
                ids=pd.DataFrame({"material": df["material"].to_numpy()}), flux=df["heat_flux_kWm2"].to_numpy(float),
                scale="log1p", log_target=True, clip=(0.0, np.inf), unit="kW/m2",
                group_def="material, PlasticC -> HIPS (merged; 64 groups), as exp074")


def spec_cr():
    import train_blend_round1 as B
    D, loc = call_capturing(B.data_cr, ["df"])
    df = loc["df"]
    return dict(key="Char Residue", tag="cr", model=B.Avg([D["inc"], B.et(), B.ridge()]), X=D["X"], y=D["y"],
                groups=D["g"], feats=D["feats"], ids=pd.DataFrame({"material": df["material"].to_numpy()}),
                flux=df["heat_flux_kWm2"].to_numpy(float), scale="identity", log_target=False, clip=(0.0, 100.0),
                unit="%", group_def="material, PlasticC -> HIPS (merged; 61 groups), as exp117")


def spec_figra():
    import train_blend_round1 as B
    D, loc = call_capturing(B.data_figra, ["d"])
    d = loc["d"]
    return dict(key="FIGRA", tag="figra", model=B.Avg([D["inc"], B.et(), B.ridge()]), X=D["X"], y=D["y"],
                groups=D["g"], feats=D["feats"], ids=pd.DataFrame({"material": d["material"].to_numpy()}),
                flux=d["heat_flux_kWm2"].to_numpy(float), scale="log1p", log_target=True, clip=(0.0, np.inf),
                unit="kW/m2/s", group_def="material (67 groups), as exp119")


def spec_tti(rows="S681"):
    import train_blend_round1 as B
    D = B.data_tti()
    import train_tti_v5_physics as v5
    df = v5.df
    S = dict(key="TTI", tag="tti", model=D["inc"], X=D["X"], y=D["y"], groups=D["g"], feats=D["feats"],
             ids=pd.DataFrame({"material": df["material"].to_numpy()}), flux=df["heat_flux_kWm2"].to_numpy(float),
             scale="log_glm", log_target=False, clip=(0.0, np.inf), unit="s", v3=True,
             group_def="material (unmerged; 76 groups), as exp046; 681-row record population (B4/D4 rows kept)")
    if rows == "S681":
        return S
    # S675 (tti_round3.json D4b canonical): drop exactly the 6 operator-flagged replicates, then reproduce the TTI
    # number of record (exp154 = exp046 pipeline on S675) from numbers_of_record.json in step 1.
    with open(os.path.join(ROOT, "experiments", "configs", "tti_round3.json")) as f:
        qa = json.load(f)["data_qa_decision"]
    flagged = next(v for k, v in qa.items() if k.startswith("flagged_ids"))
    rid = df["material"] + "|" + df["hf_tag"] + "|R" + df["replicate"].astype(str)
    keep = ~rid.isin(flagged).to_numpy()
    assert len(flagged) == 6 and (~keep).sum() == 6 and keep.sum() == 675, (len(flagged), (~keep).sum())
    assert len(np.unique(D["g"][keep])) == 76
    with open(os.path.join(ROOT, "experiments", "numbers_of_record.json")) as f:
        nor = json.load(f)["targets"]["TTI"]
    S.update(tag="tti_s675", X=S["X"][keep], y=S["y"][keep], groups=S["groups"][keep], flux=S["flux"][keep],
             ids=S["ids"][keep].reset_index(drop=True),
             rec={**record_cfg("TTI"), "exp_id": nor["exp_id"], "record": [nor["value"], nor["ci_lo"], nor["ci_hi"]]},
             group_def="material (unmerged; 76 groups); S675 = 681-row population minus the 6 tti_round3.json "
                       "data_qa_decision flagged replicates (D4b canonical); step 1 vs the TTI number of record")
    return S


def spec_thr():
    # The data-prep lines of train_thr_v1_replicates.py, verbatim (that script has module-level code, so it cannot be
    # imported); build_features and scalar_table are imported from train_thr_v1 (__main__-guarded).
    from train_thr_v1 import LIN, build_features, scalar_table
    DATA_DIR = HERE
    r = pd.read_csv(f"{DATA_DIR}/cone_replicates_fsri.csv")
    r = r[~r["did_not_ignite"] & ~r["early_stop"]]
    thr = r.groupby(["material", "heat_flux_kWm2"], as_index=False).agg(
        total_heat_release_MJm2=("thr_MJm2", "mean"), n_reps=("thr_MJm2", "size"), n_trunc=("truncated_test", "sum"))
    df = build_features(thr, scalar_table())
    df["phys_MJm2"] = df.mpa_flux_kgm2 * df.hc_bc_kJg  # exp040: h_c = baseline-corrected MCC total heat release
    core = df.dropna(subset=["phys_MJm2"]).reset_index(drop=True)
    assert (len(core), core.material.nunique()) == (191, 65), (len(core), core.material.nunique())
    return dict(key="THR", tag="thr", model=LIN, X=core[["phys_MJm2"]].to_numpy(float),
                y=core.total_heat_release_MJm2.to_numpy(float), groups=core.material.to_numpy(), feats=["phys_MJm2"],
                ids=pd.DataFrame({"material": core["material"].to_numpy()}),
                flux=core["heat_flux_kWm2"].to_numpy(float), scale="identity", log_target=False, clip=(0.0, np.inf),
                unit="MJ/m2", group_def="material (unmerged; 65 groups), as exp045")


SPECS = {"pHRR": spec_phrr, "TTI": spec_tti, "CR": spec_cr, "FIGRA": spec_figra, "THR": spec_thr}


# ---------------------------------------------------------------- step 1
def step1(S, log):
    rec = S.get("rec") or record_cfg(S["key"])
    t = time.time()
    rk, oof = repeated_group_kfold(S["model"], S["X"], S["y"], S["groups"], log_target=S["log_target"])
    lo, hi = bootstrap_ci(S["y"], oof[0], S["groups"])
    ref = rec["record"]
    ok = bool(abs(rk.mean() - ref[0]) <= 0.002 and abs(lo - ref[1]) <= 0.002 and abs(hi - ref[2]) <= 0.002)
    log(f"STEP 1 {rec['exp_id']}: GKF {rk.mean():.4f} +- {rk.std():.4f} CI [{lo:.4f}, {hi:.4f}] (record {ref[0]} "
        f"[{ref[1]}, {ref[2]}]); within 0.002: {ok}  ({time.time() - t:.0f}s)")
    return dict(exp_id=rec["exp_id"], mean=float(rk.mean()), std=float(rk.std()), ci_lo=float(lo), ci_hi=float(hi),
                ref=ref, ok=ok, per_repeat=[float(v) for v in rk]), oof


# ---------------------------------------------------------------- nested evaluation
def variants_of(S):
    return ["P"] + (["V1"] if S["mondrian"] else []) + ["V2"] + (["V3"] if S.get("v3") else []) + ["CQR"]


def nested(S, log):
    X, y, g, st = S["X"], S["y"], S["groups"], S["strata"]
    sc = SCALES[S["scale"]]
    s = sc.y_to_score(y)
    n = len(y)
    V = variants_of(S)
    LO = {v: np.full((REPEATS, n), np.nan) for v in V}
    HI = {v: np.full((REPEATS, n), np.nan) for v in V}
    qlo_m, qhi_m = qmodel(0.05), qmodel(0.95)
    for r in range(REPEATS):
        t = time.time()
        seen = np.zeros(n, int)
        for otr, ote in GroupKFold(n_splits=K, shuffle=True, random_state=r).split(X, y, g):
            assert set(g[otr]).isdisjoint(g[ote]), "outer train/test groups overlap"
            go = g[otr]
            nte = len(ote)
            cal_fold = np.full(len(otr), -1)
            cal_pred, cal_qlo, cal_qhi = (np.full(len(otr), np.nan) for _ in range(3))
            te_pred, te_qlo, te_qhi = (np.full((K, nte), np.nan) for _ in range(3))
            for k, (itr, ite) in enumerate(GroupKFold(n_splits=K, shuffle=True, random_state=r).split(X[otr], y[otr], go)):
                assert set(go[itr]).isdisjoint(go[ite]), "inner train/held-out groups overlap"
                tr, ho = otr[itr], otr[ite]  # positions into the full table: every inner row is an outer-train row
                assert np.isin(tr, otr).all() and np.isin(ho, otr).all()
                assert set(g[tr]).isdisjoint(g[ote]) and set(g[ho]).isdisjoint(g[ote]), "outer test group in inner fit"
                assert (cal_fold[ite] == -1).all()
                m = clone(S["model"]).fit(X[tr], sc.fit_y(y[tr]))
                cal_pred[ite], te_pred[k], cal_fold[ite] = m.predict(X[ho]), m.predict(X[ote]), k
                ql = clone(qlo_m).fit(X[tr], s[tr])
                qh = clone(qhi_m).fit(X[tr], s[tr])
                cal_qlo[ite], cal_qhi[ite] = ql.predict(X[ho]), qh.predict(X[ho])
                te_qlo[k], te_qhi[k] = ql.predict(X[ote]), qh.predict(X[ote])
            assert (cal_fold >= 0).all() and np.isfinite(cal_pred).all() and np.isfinite(te_pred).all()
            s_cal, s_hat_cal, s_hat_te = s[otr], sc.pred_to_score(cal_pred), sc.pred_to_score(te_pred)
            R = np.abs(s_cal - s_hat_cal)
            E = np.maximum(cal_qlo - s_cal, s_cal - cal_qhi)
            if "V3" in V:  # Poisson-normalised score on seconds (TTI only; model output is mu in seconds)
                R3 = np.abs(y[otr] - cal_pred) / np.sqrt(cal_pred)
            for v in V:
                cst = np.zeros(len(otr)) if v == "V1" else st[otr]
                tst = np.zeros(nte) if v == "V1" else st[ote]
                for u in np.unique(tst):
                    cm, tm = cst == u, tst == u
                    if not cm.any():
                        lo, hi = np.full(tm.sum(), -np.inf), np.full(tm.sum(), np.inf)
                    else:
                        w = np.ones(cm.sum()) if v == "V2" else C.group_weights(go[cm])
                        if v in ("P", "V1", "V2"):
                            lo, hi = C.cvplus(s_hat_te[:, tm], cal_fold[cm], R[cm], w)
                        elif v == "V3":
                            mu = te_pred[:, tm]
                            lo, hi = C.cvplus(mu, cal_fold[cm], R3[cm], w, scale_test=np.sqrt(mu))
                        else:
                            lo, hi = C.cqr(te_qlo[:, tm], te_qhi[:, tm], cal_fold[cm], E[cm], w)
                    LO[v][r, ote[tm]], HI[v][r, ote[tm]] = lo, hi
            seen[ote] += 1
        assert (seen == 1).all(), "a row did not get exactly one interval in this repeat"
        log(f"  nested repeat {r}: {time.time() - t:.0f}s")
    for v in V:
        assert not np.isnan(LO[v]).any() and not np.isnan(HI[v]).any()
    return LO, HI


# ---------------------------------------------------------------- summaries
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if np.isfinite(o) else ("inf" if o > 0 else "-inf" if o < 0 else "nan")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


VAR_TEXT = {
    "P": "primary (deployed): group-weighted CV+, Mondrian by flux anchor (FSRI) or one stratum (LOI)",
    "V1": "report-only: P with one stratum (no Mondrian)",
    "V2": "report-only: P with all w_i = 1 (row-exchangeable textbook CV+)",
    "V3": "report-only (TTI): Poisson-normalised score |y - mu| / sqrt(mu) in seconds, group-weighted, Mondrian",
    "CQR": "report-only: CQR with HGB quantile 0.05 / 0.95 models on the score scale, group-weighted, Mondrian",
}


def summarise(S, LO, HI, log):
    sc = SCALES[S["scale"]]
    s, y, g, st = sc.y_to_score(S["y"]), S["y"], S["groups"], S["strata"]
    out, cov_all = {}, {}
    for v in LO:
        tgt = y if v == "V3" else s
        back = _ident if v == "V3" else sc.back
        logs = False if v == "V3" else sc.log
        cov = ((LO[v] <= tgt) & (tgt <= HI[v])).astype(float)
        cov_all[v] = cov
        d = dict(description=VAR_TEXT[v], coverage_scale="seconds" if v == "V3" else "score",
                 overall=C.cell_summary(cov, LO[v], HI[v], g, back, S["clip"], logs))
        if S["mondrian"]:
            d["by_stratum"] = {f"{u:g}": C.cell_summary(cov[:, st == u], LO[v][:, st == u], HI[v][:, st == u],
                                                        g[st == u], back, S["clip"], logs) for u in np.unique(st)}
            fl = S["flux"]
            d["by_raw_flux"] = {f"{f:g}": dict(n=int((fl == f).sum()), row_coverage=float(cov[:, fl == f].mean()))
                                for f in np.unique(fl)}
        out[v] = d
        o = d["overall"]
        log(f"  {v:4s} overall: group_all {o['group_all']['mean']:.3f}  group_mean {o['group_mean']['mean']:.3f} "
            f"[{o['group_mean']['ci'][0]:.3f}, {o['group_mean']['ci'][1]:.3f}] ({o['group_mean']['reading']})  "
            f"row {o['row']['mean']:.3f}  median raw width "
            + (f"{o['raw_width_median']['mean']:.4g} {S['unit']}" if o["raw_width_median"] else "n/a")
            + f"  unbounded {o['n_unbounded']['total_over_repeats']}  empty {o['n_empty']['total_over_repeats']}")
        for u, c in d.get("by_stratum", {}).items():
            log(f"       stratum {u:>3}: group_all {c['group_all']['mean']:.3f}  group_mean {c['group_mean']['mean']:.3f} "
                f"[{c['group_mean']['ci'][0]:.3f}, {c['group_mean']['ci'][1]:.3f}] ({c['group_mean']['reading']})  "
                f"row {c['row']['mean']:.3f}  median raw width "
                + (f"{c['raw_width_median']['mean']:.4g}" if c["raw_width_median"] else "n/a")
                + f"  (rows {c['n_rows']}, groups {c['n_groups']}, unbounded {c['n_unbounded']['total_over_repeats']})")
    return out, cov_all


def strata_counts(S):
    st, g = S["strata"], S["groups"]
    out = {f"{u:g}": dict(rows=int((st == u).sum()), groups=int(len(np.unique(g[st == u])))) for u in np.unique(st)}
    if S["mondrian"]:
        fl = S["flux"]
        out["raw_flux"] = {f"{f:g}": dict(rows=int((fl == f).sum()), groups=int(len(np.unique(g[fl == f]))))
                           for f in np.unique(fl)}
    return out


def run(S, out_dir, log=say):
    """Step 1, nested evaluation, summary JSON and rows CSV for one record. Returns (summary, step-1 OOF, LO, HI)."""
    t0 = time.time()
    S.setdefault("mondrian", S.get("flux") is not None)
    S["strata"] = C.stratum_of(S["flux"]) if S["mondrian"] else np.zeros(len(S["y"]))
    rec = S.get("rec") or record_cfg(S["key"])
    assert rec["mondrian"] == S["mondrian"] and rec["exp_id"]
    n, G = len(S["y"]), len(np.unique(S["groups"]))
    log(f"{S['key']} ({rec['exp_id']}): {n} rows / {G} groups; groups = {S['group_def']}; "
        f"score: {SCALES[S['scale']].text}; display clip {S['clip']}")
    sc_counts = strata_counts(S)
    for u, c in sc_counts.items():
        if u != "raw_flux":
            log(f"  stratum {u}: {c['rows']} rows / {c['groups']} groups")
    for f, c in sc_counts.get("raw_flux", {}).items():
        log(f"    raw flux {f:>3}: {c['rows']} rows / {c['groups']} groups")
    s1, oof = step1(S, log)
    summ = dict(config=CONFIG, target=S["key"], exp_id=rec["exp_id"], n_rows=n, n_groups=G,
                group_definition=S["group_def"], score_scale=SCALES[S["scale"]].text,
                display_clip=list(S["clip"]), alpha=C.ALPHA, nominal=1 - C.ALPHA,
                strata=("nearest of 25/50/75 kW/m2, ties low" if S["mondrian"] else "one stratum (no heat flux)"),
                strata_counts=sc_counts, step1=s1)
    os.makedirs(out_dir, exist_ok=True)
    if not s1["ok"]:
        log(f"BLOCKED: {rec['exp_id']} not reproduced within 0.002; no conformal computation for {S['key']}.")
        summ.update(status="BLOCKED", runtime_s=round(time.time() - t0, 1))
        with open(os.path.join(out_dir, f"{S['tag']}_summary.json"), "w") as f:
            json.dump(jsonable(summ), f, indent=2)
        return summ, oof, None, None
    log(f"Nested GroupKFold {K}x{REPEATS} (outer r, inner r), variants {variants_of(S)}")
    LO, HI = nested(S, log)
    log("nested-disjointness asserts passed; every row got exactly one interval per repeat")
    var, cov = summarise(S, LO, HI, log)
    summ.update(status="ok", variants=var, nested_asserts="passed", runtime_s=round(time.time() - t0, 1),
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                              pandas=pd.__version__))
    with open(os.path.join(out_dir, f"{S['tag']}_summary.json"), "w") as f:
        json.dump(jsonable(summ), f, indent=2)
    # per-row CSV
    sc = SCALES[S["scale"]]
    rows = pd.DataFrame({"row_index": np.arange(n), "group": S["groups"]})
    for c in S["ids"].columns:
        rows[c] = S["ids"][c].to_numpy()
    if S["mondrian"]:
        rows["heat_flux_kWm2"] = S["flux"]
        rows["stratum"] = S["strata"]
    rows["y"] = S["y"]
    rows["record_oof_r0"] = oof[0]
    rows["P_r0_lo"] = sc.back(LO["P"][0])
    rows["P_r0_hi"] = sc.back(HI["P"][0])
    for v in LO:
        rows[f"{v}_cov_freq"] = cov[v].mean(axis=0)
    rows["P_mean_score_width"] = (HI["P"] - LO["P"]).mean(axis=0)
    rows.to_csv(os.path.join(out_dir, f"{S['tag']}_rows.csv"), index=False)
    log(f"wrote {out_dir}/{S['tag']}_summary.json and {S['tag']}_rows.csv; runtime {time.time() - t0:.0f}s")
    return summ, oof, LO, HI


# ---------------------------------------------------------------- pHRR demo artifact
def write_phrr_artifact(S, summ, oof, log):
    import joblib
    mtime = os.stat(BUNDLE).st_mtime_ns
    bundle = joblib.load(BUNDLE)
    assert list(bundle["features"]) == list(S["feats"]), "artifact features != phrr_hgb_v4 bundle features"
    X, y, g = S["X"], S["y"], S["groups"]
    fold = np.full(len(y), -1)
    oof_log = np.full(len(y), np.nan)
    models = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=K, shuffle=True, random_state=0).split(X, y, g)):
        assert set(g[tr]).isdisjoint(g[te])
        m = clone(S["model"]).fit(X[tr], np.log1p(y[tr]))
        oof_log[te], fold[te] = m.predict(X[te]), k
        models.append(m)
    assert (fold >= 0).all()
    assert np.allclose(np.expm1(oof_log), oof[0], rtol=0, atol=1e-8), "deployed repeat-0 OOF != step-1 repeat-0 OOF"
    log(f"pHRR artifact: repeat-0 OOF of the 5 fold models equals step-1 repeat-0 OOF "
        f"(max |diff| {np.abs(np.expm1(oof_log) - oof[0]).max():.2e} kW/m2)")
    fog = {}
    for gg, k in zip(g, fold):
        assert fog.setdefault(gg, int(k)) == int(k)
    st = S["strata"]
    P = summ["variants"]["P"]
    art = dict(
        fold_models=models, fold_of_group=fog, features=list(S["feats"]), scale="log1p", alpha=C.ALPHA,
        anchors=list(C.ANCHORS),
        calib=dict(material=S["ids"]["material"].to_numpy(), group=np.asarray(g), fold=fold, stratum=st,
                   R=np.abs(np.log1p(y) - oof_log), weight=C.group_weights(g, st), heat_flux_kWm2=S["flux"]),
        coverage=jsonable(dict(overall=P["overall"], by_stratum=P["by_stratum"])),
        method=("Group-weighted Mondrian CV+ (Barber et al. 2021; group weights after Dunn, Wasserman & Ramdas 2023), "
                "nominal 90% (alpha 0.10), on the log1p(pHRR) scale. Five HGB fold models (hgb() of "
                "train_phrr_v3_robust, fitted on log1p) on the record's repeat-0 folds, GroupKFold(5, shuffle=True, "
                "random_state=0) over the 64 merged material groups; calibration scores R = |log1p(y) - OOF| of the "
                "191 rows; each material group carries total weight 1 per stratum; strata = nearest of 25/50/75 kW/m2 "
                "(ties low). Coverage = nested repeated GroupKFold 5x10 (report-only)."),
        config=CONFIG, point_model="models/v2/phrr_hgb_v4.joblib (unchanged; the interval is not re-centred on it)",
        n_groups=int(len(np.unique(g))),
    )
    joblib.dump(art, ART)
    assert os.stat(BUNDLE).st_mtime_ns == mtime, "models/v2/phrr_hgb_v4.joblib was modified"
    log(f"wrote {os.path.relpath(ART, ROOT)}; models/v2/phrr_hgb_v4.joblib mtime unchanged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(SPECS))
    ap.add_argument("--tti-rows", choices=["S681", "S675"], default="S681",
                    help="TTI only: S675 drops the 6 tti_round3 D4 flagged replicates; outputs tagged tti_s675")
    a = ap.parse_args()
    tname = a.target
    assert a.tti_rows == "S681" or tname == "TTI", "--tti-rows applies to --target TTI only"
    S = spec_tti(a.tti_rows) if tname == "TTI" else SPECS[tname]()
    out_dir = os.path.join(ROOT, "experiments", "runs", "conformal")
    summ, oof, LO, HI = run(S, out_dir)
    if summ["status"] != "ok":
        sys.exit(2)
    if tname == "pHRR":
        write_phrr_artifact(S, summ, oof, say)


if __name__ == "__main__":
    main()
