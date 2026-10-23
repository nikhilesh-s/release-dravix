#!/usr/bin/env python3
"""REPORT-ONLY secondary TTI metrics -- follows experiments/configs/tti_secondary_metrics.json exactly.

Not a modelling round: no candidate, no selection, no adoption, no number of record changes. Every number here was
defined post hoc, never promotes, and is never compared with the 0.58 bar (pomazi_2026).

What it does (config sections in brackets):
- [data] df = v5.df.copy() after importing train_tti_v5_physics with --onset-bc --replicate-table (as
  train_blend_round1.data_tti()); 681 rows / 76 materials, all rows train and score, no exclusion.
- [protocol] one eval_protocol.repeated_group_kfold call per model (GroupKFold 5 x 10, groups = material,
  random_state 0..9). Every metric comes from that call's OOF array (10 x 681). No other fitting.
- [models] the 10 committed models in fixed order: exp046, exp047, b1, b2, et_alone, c5_ridge, t1, t2, t3, t4.
- [step1_reproduction] pooled-seconds rkf_mean and repeat-0 CI reproduce each record (or et_alone's anchor);
  any miss stops the run before any secondary metric is printed.
- [step2_definition_check] the lever-sweep values reproduce; any miss stops the run before any table is printed.
- [metrics] S25/S50/S75/S50_75, LOG_ALL/LOG25/LOG50/LOG75, SPR25/50/75, SPM25/50/75; repeat-0 bootstrap CIs on
  S* and LOG_ALL; paired wins vs exp046; legacy NOT25 / NOT25_NO15 masks (labelled, NOT secondary metrics);
  label checks and label-only reference ceilings.

Executor notes (listed for the critic):
- data.extra_columns: the round-1 columns (q_kWm2, q_over_dT; inv_z) are copied verbatim from
  train_tti_round1.py lines 32-33 and 39-40. The round-2 columns (train_tti_round2.py line 29 and lines 35-55,
  emissivity join) are NOT built: after the Oct 6 amendment dropped e1-e4, no model on the list uses them, and the
  config's own banned_moves_note limits thermal-property joins to "those already inside the refitted models"
  (banned_moves: emissivity_fsri.csv exempt "for TTI round 2 only"). Extra columns never change a model's X, so
  no number is affected.
- paired_report says "each of the 16 secondary metrics" but lists 14 (S25, S50, S75, S50_75, LOG_ALL, LOG25,
  LOG50, LOG75, SPR25, SPR50, SPR75, SPM25, SPM50, SPM75). Wins are computed for exactly the 14 listed.
- Tolerance comparisons use a 1e-9 float guard (|got - expected| <= 0.002 + 1e-9).

Run: OMP_NUM_THREADS=1 python3 data/v2/report_tti_secondary_metrics.py | tee experiments/runs/tti_secondary_metrics.log
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))

CONFIG = "experiments/configs/tti_secondary_metrics.json"
TAG = ("REPORT-ONLY, defined post hoc (experiments/configs/tti_secondary_metrics.json); never promotes; "
       "not comparable to the 0.58 bar")
OUT_JSON = ROOT / "experiments/runs/tti_secondary_metrics.json"
CLIP = 1e-6
TOL = 0.002
EPS = 1e-9
MODEL_ORDER = ["exp046", "exp047", "b1", "b2", "et_alone", "c5_ridge", "t1", "t2", "t3", "t4"]
SECONDARY = ["S25", "S50", "S75", "S50_75", "LOG_ALL", "LOG25", "LOG50", "LOG75",
             "SPR25", "SPR50", "SPR75", "SPM25", "SPM50", "SPM75"]
CI_METRICS = ["S25", "S50", "S75", "S50_75", "LOG_ALL"]


def inv_z(z):
    return 1.0 / np.maximum(z, 0.02) ** 2      # cap 2500 s, declared  (verbatim train_tti_round1.py lines 39-40)


def r4(x):
    """Stored to 4 decimals; NaN stored as null."""
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def f3(x):
    x = float(x)
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def within(got, exp, tol=TOL):
    return bool(np.isfinite(got) and abs(got - exp) <= tol + EPS)


def wins_ok(got, exp):
    """Paired-win count inside the expected inclusive range (lo, hi)."""
    return bool(exp[0] <= got <= exp[1])


def eq2(got, exp):
    """Values given to 2 decimals: equal after rounding."""
    return bool(np.isfinite(got) and abs(round(float(got), 2) - exp) <= EPS)


def main():
    t_start = time.time()
    # ---------------- data (config: data.build, imports_allowed) ----------------
    argv = sys.argv
    sys.argv = [argv[0], "--onset-bc", "--replicate-table"]
    import train_tti_v5_physics as v5
    sys.argv = argv
    import train_blend_round1 as B
    from eval_protocol import bootstrap_ci, repeated_group_kfold

    df = v5.df.copy()
    assert len(df) == 681, len(df)
    assert df["material"].nunique() == 76, df["material"].nunique()
    assert (df["time_to_ignition_s"] > 0).all()
    assert (df["spark_ignitor"] == 1).all()
    counts = {float(k): int(v) for k, v in df["heat_flux_kWm2"].value_counts().items()}
    assert counts == {15.0: 3, 25.0: 223, 35.0: 16, 50.0: 221, 75.0: 217, 100.0: 1}, counts

    # round-1 extra columns, verbatim train_tti_round1.py lines 32-33 (inv_z above: lines 39-40)
    df["q_kWm2"] = df["heat_flux_kWm2"].astype(float)
    df["q_over_dT"] = df["q_kWm2"] / (df["t_onset_C"] - v5.T0_C)

    y = df["time_to_ignition_s"].to_numpy(float)
    groups = df["material"].to_numpy()
    flux = df["heat_flux_kWm2"].to_numpy(float)

    masks = {"F25": flux == 25.0, "F50": flux == 50.0, "F75": flux == 75.0,
             "F50_75": np.isin(flux, [50.0, 75.0]), "ALL": np.ones(len(y), bool)}
    expect_mask = {"F25": (223, 69), "F50": (221, 72), "F75": (217, 72), "F50_75": (438, 72), "ALL": (681, 76)}
    for k, m in masks.items():
        got = (int(m.sum()), int(pd.Series(groups[m]).nunique()))
        assert got == expect_mask[k], (k, got)
    not25 = flux != 25.0
    not25_no15 = ~np.isin(flux, [15.0, 25.0])
    assert int(not25.sum()) == 458 and int(not25_no15.sum()) == 455
    masonite15 = (flux == 15.0) & (groups == "Masonite_Board")
    assert int(masonite15.sum()) == 3 and int((flux == 15.0).sum()) == 3
    print(f"data: {len(y)} rows / {len(np.unique(groups))} materials; flux counts {counts}; all spark-ignited")
    print("masks: " + "; ".join(f"{k} {v[0]} rows / {v[1]} materials" for k, v in expect_mask.items()))

    # ---------------- models (fixed order; config 'models') ----------------
    assert v5.MODELS[0][0].startswith("C0")

    def glm():
        return v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000))

    MODELS = {  # id: (estimator, features, log_target)
        "exp046": (glm(), v5.PHYS, False),
        "exp047": (v5.MODELS[0][1], v5.F0, True),
        "b1": (B.Avg([glm(), B.log1p_wrap(B.et()), B.log1p_wrap(B.ridge())]), v5.PHYS, False),
        "b2": (B.Avg([glm(), B.log1p_wrap(B.et())]), v5.PHYS, False),
        "et_alone": (B.log1p_wrap(B.et()), v5.PHYS, False),
        "c5_ridge": (v5.linear(Ridge(alpha=1.0)), v5.PHYS, True),
        # t1-t4: verbatim train_tti_round1.py CANDS (lines 45-50)
        "t1": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS + ["log_q_excess"], False),
        "t2": (TransformedTargetRegressor(regressor=v5.linear(Ridge(alpha=1.0)), func=lambda t: t ** -0.5,
                                          inverse_func=inv_z, check_inverse=False),
               ["q_kWm2", "q_over_dT", "log_L", "log_rho", "log_kpc"], False),
        "t3": (HistGradientBoostingRegressor(loss="poisson", **v5.HGB), v5.F3 + ["log_q_excess", "log_t_thick_loss"], False),
        "t4": (HistGradientBoostingRegressor(**v5.HGB), v5.F0, True),
    }
    assert list(MODELS) == MODEL_ORDER
    RECORD = {"exp046": (0.3133, 0.0715, 0.4958), "exp047": (0.2613, 0.13, 0.441), "b1": (0.3265, 0.2068, 0.535),
              "b2": (0.3316, 0.2163, 0.5348), "t1": (0.2979, 0.0467, 0.4938), "t2": (-38.1081, -85.4366, -25.1826),
              "t3": (0.2123, -0.0036, 0.4418), "t4": (0.2671, 0.1705, 0.4649)}
    ANCHOR = {"et_alone": 0.255}

    # ---------------- step 1: CV once per model + reproduction ----------------
    print(f"\n(1) STEP-1 REPRODUCTION, pooled seconds R2 (existing headline; context only, not a secondary metric) -- {TAG}")
    print(f"{'model':9s} {'rkf_mean':>9s} {'std':>7s}  {'rep-0 CI':>20s}  {'record / anchor':>32s}  ok   time")
    oof_all, step1, failed = {}, {}, []
    for mid, (est, feats, logt) in MODELS.items():
        t = time.time()
        X = df[feats].to_numpy(float)
        rk, oof = repeated_group_kfold(est, X, y, groups, n_splits=5, repeats=10, log_target=logt)
        lo, hi = bootstrap_ci(y, oof[0], groups)
        oof_all[mid] = oof
        if mid in RECORD:
            m0, l0, h0 = RECORD[mid]
            if mid == "exp047":  # 3-decimal CI vs 3-decimal record
                ok = within(rk.mean(), m0) and within(round(lo, 3), l0) and within(round(hi, 3), h0)
            else:
                ok = within(rk.mean(), m0) and within(lo, l0) and within(hi, h0)
            rec = dict(rkf_mean=m0, ci=[l0, h0])
            rec_s = f"{m0:.4f} [{l0:.4f}, {h0:.4f}]"
        elif mid in ANCHOR:
            ok = within(rk.mean(), ANCHOR[mid])
            rec = dict(anchor_rkf_mean=ANCHOR[mid])
            rec_s = f"anchor {ANCHOR[mid]:.3f}"
        else:
            ok = None
            rec = None
            rec_s = "no record, no anchor"
        step1[mid] = dict(rkf_mean=r4(rk.mean()), rkf_std=r4(rk.std()), per_repeat=[r4(v) for v in rk],
                          ci=[r4(lo), r4(hi)], record=rec, ok=ok)
        print(f"{mid:9s} {rk.mean():9.4f} {rk.std():7.4f}  [{lo:8.4f}, {hi:8.4f}]  {rec_s:>32s}  "
              f"{'-' if ok is None else ('OK' if ok else 'FAIL'):4s} {time.time() - t:5.0f}s", flush=True)
        if ok is False:
            failed.append(mid)
            print(f"BLOCKED: step 1 -- {mid} did not reproduce its record/anchor within {TOL}; stopping the whole run. "
                  "No table published. Write blockers/BLOCKER-<YYYYMMDD-HHMM>-tti-secondary-repro.md.")
            sys.exit(2)
    print("step 1: all records / anchors reproduced")

    # ---------------- metrics ----------------
    def metric_rep(mname, p):
        if mname in ("S25", "S50", "S75", "S50_75"):
            m = masks["F" + mname[1:]]
            return r2_score(y[m], p[m])
        if mname.startswith("LOG"):
            m = masks["ALL" if mname == "LOG_ALL" else "F" + mname[3:]]
            return r2_score(np.log(y[m]), np.log(np.clip(p[m], CLIP, None)))
        if mname.startswith("SPR"):
            m = masks["F" + mname[3:]]
            return spearmanr(y[m], p[m]).statistic
        if mname.startswith("SPM"):
            m = masks["F" + mname[3:]]
            med = pd.DataFrame({"g": groups[m], "y": y[m], "p": p[m]}).groupby("g")[["y", "p"]].median()
            return spearmanr(med["y"], med["p"]).statistic
        raise KeyError(mname)

    def ci_rep0(mname, p0):
        if mname == "LOG_ALL":
            return bootstrap_ci(np.log(y), np.log(np.clip(p0, CLIP, None)), groups, n_boot=2000, seed=0)
        m = masks["F" + mname[1:]]
        return bootstrap_ci(y[m], p0[m], groups[m], n_boot=2000, seed=0)

    vals = {}   # vals[mid][metric] = np.array(10)
    res = {}
    nan_cells = []
    for mid in MODEL_ORDER:
        oof = oof_all[mid]
        vals[mid] = {}
        res[mid] = {}
        for mname in SECONDARY:
            pr = np.array([metric_rep(mname, oof[r]) for r in range(10)], dtype=float)
            vals[mid][mname] = pr
            mean, std = np.mean(pr), np.std(pr)
            if not np.isfinite(mean):
                nan_cells.append(f"{mid}.{mname}")
            ci = None
            if mname in CI_METRICS:
                lo, hi = ci_rep0(mname, oof[0])
                ci = [float(lo), float(hi)]
            res[mid][mname] = dict(per_repeat=[r4(v) for v in pr], mean=r4(mean), std=r4(std), rep0=r4(pr[0]),
                                   ci_rep0=None if ci is None else [r4(ci[0]), r4(ci[1])], wins_vs_exp046=None,
                                   _mean=mean, _std=std, _ci=ci)
        res[mid]["n_clipped"] = int((oof <= CLIP).sum())
    for mid in MODEL_ORDER:
        if mid == "exp046":
            continue
        for mname in SECONDARY:
            a, b = vals[mid][mname], vals["exp046"][mname]
            ok = np.isfinite(a) & np.isfinite(b)
            res[mid][mname]["wins_vs_exp046"] = int(((a > b) & ok).sum())

    # legacy reconciliation (NOT secondary metrics)
    legacy = {}
    for mid in MODEL_ORDER:
        oof = oof_all[mid]
        n25 = np.array([r2_score(y[not25], oof[r][not25]) for r in range(10)])
        n2515 = np.array([r2_score(y[not25_no15], oof[r][not25_no15]) for r in range(10)])
        legacy[mid] = dict(NOT25=dict(per_repeat=[r4(v) for v in n25], rep0=r4(n25[0]), mean=r4(np.mean(n25)),
                                      std=r4(np.std(n25)), _rep0=n25[0], _mean=np.mean(n25), _std=np.std(n25)),
                           NOT25_NO15=dict(per_repeat=[r4(v) for v in n2515], mean=r4(np.mean(n2515)),
                                           std=r4(np.std(n2515)), _mean=np.mean(n2515), _std=np.std(n2515)))
    yn = y[not25]
    ss_n = (yn - yn.mean()) ** 2
    share_masonite = ss_n[masonite15[not25]].sum() / ss_n.sum()
    ss_all = (y - y.mean()) ** 2
    share_f25 = ss_all[masks["F25"]].sum() / ss_all.sum()

    # reference ceilings (label-only), same formulas as train_tti_v5_physics.ceiling()
    def ceil_vals(data):
        yy = data[v5.TARGET]
        cell = data.groupby(["material", "heat_flux_kWm2", "spark_ignitor"])[v5.TARGET]
        n, s = cell.transform("size"), cell.transform("sum")
        m = n >= 2
        return (r2_score(yy, cell.transform("mean")), r2_score(yy[m], ((s - yy) / (n - 1))[m]), int(m.sum()))

    c_all = ceil_vals(df)
    c25 = ceil_vals(df[masks["F25"]])
    c50 = ceil_vals(df[masks["F50"]])
    c75 = ceil_vals(df[masks["F75"]])
    ceilings = {"ALL": dict(in_sample=c_all[0], loro=c_all[1], loro_rows=c_all[2]),
                "F25": dict(in_sample=c25[0], loro=c25[1], loro_rows=c25[2]),
                "F50": dict(loro=c50[1], loro_rows=c50[2]),
                "F75": dict(loro=c75[1], loro_rows=c75[2])}

    # ---------------- step 2: definition check ----------------
    checks = []  # (name, expected, got, kind) kind: '3' tol 0.002 | '2' equal after rounding | 'wins' (lo, hi)

    def M(mid, k):
        return res[mid][k]["_mean"]

    def R0(mid, k):
        return vals[mid][k][0]

    def CI(mid, k, i):
        return res[mid][k]["_ci"][i]

    e = "exp046"
    checks += [(f"{e}.S25 mean", 0.042, M(e, "S25"), "3"), (f"{e}.S50 mean", 0.274, M(e, "S50"), "3"),
               (f"{e}.S75 mean", 0.283, M(e, "S75"), "3"), (f"{e}.S50_75 mean", 0.368, M(e, "S50_75"), "3"),
               (f"{e}.S50_75 rep0 CI lo", -0.42, CI(e, "S50_75", 0), "2"),
               (f"{e}.S50_75 rep0 CI hi", 0.59, CI(e, "S50_75", 1), "2"),
               (f"{e}.LOG_ALL mean", 0.391, M(e, "LOG_ALL"), "3"),
               (f"{e}.LOG_ALL rep0 CI lo", 0.09, CI(e, "LOG_ALL", 0), "2"),
               (f"{e}.LOG_ALL rep0 CI hi", 0.57, CI(e, "LOG_ALL", 1), "2"),
               (f"{e}.LOG25 mean", 0.025, M(e, "LOG25"), "3"), (f"{e}.LOG50 mean", 0.018, M(e, "LOG50"), "3"),
               (f"{e}.LOG75 mean", -0.006, M(e, "LOG75"), "3")]
    for fl, ex10, ex0 in (("25", 0.51, 0.50), ("50", 0.50, 0.46), ("75", 0.44, 0.40)):
        checks += [(f"{e}.SPM{fl} mean", ex10, M(e, "SPM" + fl), "2"), (f"{e}.SPM{fl} rep0", ex0, R0(e, "SPM" + fl), "2")]
    # amendment 2 (Oct 6): the SPR50 reference was double-rounded (0.4346 -> 0.435 -> 0.44); now 0.435, 3 dp, +-0.002
    for fl, ex0, kind in (("25", 0.49, "2"), ("50", 0.435, "3"), ("75", 0.39, "2")):
        checks.append((f"{e}.SPR{fl} rep0", ex0, R0(e, "SPR" + fl), kind))
    checks += [(f"{e}.NOT25 rep0", 0.798, legacy[e]["NOT25"]["_rep0"], "3"),
               (f"{e}.NOT25 mean", 0.768, legacy[e]["NOT25"]["_mean"], "3"),
               (f"{e}.NOT25 std", 0.070, legacy[e]["NOT25"]["_std"], "3"),
               (f"{e}.NOT25_NO15 mean", 0.294, legacy[e]["NOT25_NO15"]["_mean"], "3"),
               (f"{e}.NOT25_NO15 std", 0.205, legacy[e]["NOT25_NO15"]["_std"], "3")]
    b = "b2"
    checks += [(f"{b}.S50 mean", 0.501, M(b, "S50"), "3"),
               (f"{b}.S50 rep0 CI lo", 0.115, CI(b, "S50", 0), "3"), (f"{b}.S50 rep0 CI hi", 0.651, CI(b, "S50", 1), "3"),
               (f"{b}.S75 mean", 0.550, M(b, "S75"), "3"), (f"{b}.S50_75 mean", 0.572, M(b, "S50_75"), "3"),
               (f"{b}.S50_75 rep0 CI lo", 0.30, CI(b, "S50_75", 0), "2"),
               (f"{b}.S50_75 rep0 CI hi", 0.69, CI(b, "S50_75", 1), "2"),
               (f"{b}.LOG_ALL mean", 0.567, M(b, "LOG_ALL"), "3"),
               (f"{b}.LOG_ALL rep0 CI lo", 0.42, CI(b, "LOG_ALL", 0), "2"),
               (f"{b}.LOG_ALL rep0 CI hi", 0.68, CI(b, "LOG_ALL", 1), "2"),
               (f"{b}.LOG25 mean", 0.216, M(b, "LOG25"), "3")]
    for fl, ex0 in (("25", 0.57), ("50", 0.61), ("75", 0.62)):
        checks.append((f"{b}.SPR{fl} rep0", ex0, R0(b, "SPR" + fl), "2"))
    for k in ("S50", "S75", "S50_75", "LOG_ALL", "LOG25"):
        checks.append((f"{b}.{k} wins", (10, 10), res[b][k]["wins_vs_exp046"], "wins"))
    a = "et_alone"
    checks += [(f"{a}.LOG_ALL mean", 0.633, M(a, "LOG_ALL"), "3"), (f"{a}.LOG25 mean", 0.292, M(a, "LOG25"), "3"),
               (f"{a}.S75 mean", 0.613, M(a, "S75"), "3"), (f"{a}.S50_75 mean", 0.580, M(a, "S50_75"), "3")]
    for k in ("LOG_ALL", "LOG25", "S75", "S50_75"):
        checks.append((f"{a}.{k} wins", (10, 10), res[a][k]["wins_vs_exp046"], "wins"))
    c = "c5_ridge"
    for k in ("S50", "S75", "S50_75", "LOG_ALL", "LOG25"):
        checks.append((f"{c}.{k} wins", (8, 10), res[c][k]["wins_vs_exp046"], "wins"))
    checks += [("labels.masonite15_share_of_NOT25_SS", 0.745, share_masonite, "3"),
               ("labels.F25_share_of_all_SS", 0.852, share_f25, "3"),
               ("ceiling.ALL in_sample", 0.843, c_all[0], "3"), ("ceiling.ALL loro", 0.690, c_all[1], "3"),
               ("ceiling.F25 in_sample", 0.776, c25[0], "3"), ("ceiling.F25 loro", 0.558, c25[1], "3"),
               ("ceiling.F50 loro", 0.958, c50[1], "3"), ("ceiling.F75 loro", 0.955, c75[1], "3")]

    print(f"\nSTEP-2 DEFINITION CHECK (lever-sweep values; tolerance 0.002 for 3-decimal values, equal after rounding "
          f"for 2-decimal values) -- {TAG}")
    step2, bad = {}, []
    for name, exp, got, kind in checks:
        if kind == "wins":
            ok = wins_ok(got, exp)
            exp_s = f"{exp[0]}/10" if exp[0] == exp[1] else f"{exp[0]}-{exp[1]}/10"
            got_s = f"{got}/10"
            step2[name] = dict(expected=list(exp), got=int(got), ok=bool(ok))
        else:
            ok = within(got, exp) if kind == "3" else eq2(got, exp)
            exp_s = f"{exp:.3f}" if kind == "3" else f"{exp:.2f}"
            got_s = f"{got:.4f}" if np.isfinite(got) else "nan"
            step2[name] = dict(expected=exp, got=r4(got), ok=bool(ok), rule="+-0.002" if kind == "3" else "2dp")
        print(f"  {name:38s} expected {exp_s:>8s}  got {got_s:>8s}  {'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(name)
    if bad:
        print(f"BLOCKED: step 2 -- {len(bad)} definition check(s) failed: {bad}. Implementation fault per config; "
              "stopping. No table published. Write blockers/BLOCKER-<YYYYMMDD-HHMM>-tti-secondary-repro.md.")
        sys.exit(3)
    print(f"step 2: all {len(checks)} definition checks pass")

    # ---------------- printed tables (2)-(7) ----------------
    def cell(mid, k):
        r = res[mid][k]
        return f"{f3(r['_mean'])} +- {f3(r['_std'])}"

    def cis(mid, k):
        ci = res[mid][k]["_ci"]
        return f"[{f3(ci[0])}, {f3(ci[1])}]"

    print(f"\n(2) SECONDS R2 WITHIN FLUX: 10-repeat mean +- std, repeat-0 material-bootstrap CI -- {TAG}")
    print(f"{'model':9s} | {'S25':>14s} {'CI':>16s} | {'S50':>14s} {'CI':>16s} | {'S75':>14s} {'CI':>16s} | "
          f"{'S50_75':>14s} {'CI':>16s}")
    for mid in MODEL_ORDER:
        print(f"{mid:9s} | " + " | ".join(f"{cell(mid, k):>14s} {cis(mid, k):>16s}" for k in ("S25", "S50", "S75", "S50_75")))

    print(f"\n(3) LOG-SPACE R2 (natural log of seconds, predictions clipped at 1e-6 s): 10-repeat mean +- std; "
          f"repeat-0 CI on LOG_ALL only -- {TAG}")
    print(f"{'model':9s} | {'LOG_ALL':>14s} {'CI':>16s} | {'LOG25':>14s} | {'LOG50':>14s} | {'LOG75':>14s} | n_clipped")
    for mid in MODEL_ORDER:
        print(f"{mid:9s} | {cell(mid, 'LOG_ALL'):>14s} {cis(mid, 'LOG_ALL'):>16s} | {cell(mid, 'LOG25'):>14s} | "
              f"{cell(mid, 'LOG50'):>14s} | {cell(mid, 'LOG75'):>14s} | {res[mid]['n_clipped']}")

    print(f"\n(4) SPEARMAN WITHIN FLUX: row-level (SPR) and material-median (SPM), 10-repeat mean +- std "
          f"(repeat-0 in the JSON) -- {TAG}")
    print(f"{'model':9s} | " + " | ".join(f"{k:>14s}" for k in ("SPR25", "SPR50", "SPR75", "SPM25", "SPM50", "SPM75")))
    for mid in MODEL_ORDER:
        print(f"{mid:9s} | " + " | ".join(f"{cell(mid, k):>14s}" for k in ("SPR25", "SPR50", "SPR75", "SPM25", "SPM50", "SPM75")))

    print(f"\n(5) PAIRED REPEATS ABOVE exp046 (of 10; strict >; NaN never counts; descriptive, never a selection rule) "
          f"-- {TAG}")
    print(f"{'model':9s} | " + " ".join(f"{k:>7s}" for k in SECONDARY))
    for mid in MODEL_ORDER[1:]:
        print(f"{mid:9s} | " + " ".join(f"{res[mid][k]['wins_vs_exp046']:>7d}" for k in SECONDARY))

    print(f"\n(6) LEGACY RECONCILIATION -- legacy mask, NOT a secondary metric -- {TAG}")
    print("    NOT25 = non-25 rows pooled (15/35/50/75/100 kW/m2), 458 rows; NOT25_NO15 = non-25 rows without the "
          "15 kW/m2 rows (35/50/75/100), 455 rows, printed only to show what drove the old figure (not a proposal "
          "to drop rows)")
    print(f"{'model':9s} | {'NOT25 rep0':>10s} | {'NOT25 mean+-std':>16s} | {'NOT25_NO15 mean+-std':>20s}")
    for mid in MODEL_ORDER:
        L = legacy[mid]
        print(f"{mid:9s} | {f3(L['NOT25']['_rep0']):>10s} | "
              f"{f3(L['NOT25']['_mean']) + ' +- ' + f3(L['NOT25']['_std']):>16s} | "
              f"{f3(L['NOT25_NO15']['_mean']) + ' +- ' + f3(L['NOT25_NO15']['_std']):>20s}")
    print(f"  label check: 3 Masonite_Board rows at 15 kW/m2 hold {share_masonite:.4f} of the NOT25 target SS "
          f"(around the NOT25 mean; expected 0.745)")
    print(f"  label check: F25 rows hold {share_f25:.4f} of the all-row target SS (expected 0.852)")

    print(f"\n(7) REFERENCE CEILINGS (label-only, cell = material x flux x spark; context, not metrics) -- {TAG}")
    print(f"  ALL: in-sample cell-mean R2 {c_all[0]:.4f} (expected 0.843); LORO {c_all[1]:.4f} on {c_all[2]} rows (expected 0.690)")
    print(f"  F25: in-sample cell-mean R2 {c25[0]:.4f} (expected 0.776); LORO {c25[1]:.4f} on {c25[2]} rows (expected 0.558)")
    print(f"  F50: LORO {c50[1]:.4f} on {c50[2]} rows (expected 0.958)")
    print(f"  F75: LORO {c75[1]:.4f} on {c75[2]} rows (expected 0.955)")
    print("  The own-50/75 oracle is not recomputed (tti-3 range 0.29-0.49 at 25 kW/m2, 0.50-0.62 overall; prose only).")

    print("\nNaN cells (listed for the critic): " + (", ".join(nan_cells) if nan_cells else "none"))
    print("n_clipped (repeat, row) pairs with OOF <= 1e-6 s: "
          + ", ".join(f"{mid} {res[mid]['n_clipped']}" for mid in MODEL_ORDER))

    # ---------------- JSON ----------------
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    out = dict(
        config=CONFIG, generated_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        label=TAG,
        step1=step1, step2=step2,
        models={mid: {**{k: clean(res[mid][k]) for k in SECONDARY}, "n_clipped": res[mid]["n_clipped"]}
                for mid in MODEL_ORDER},
        nan_cells=nan_cells,
        legacy_reconciliation=dict(
            label="legacy mask, NOT a secondary metric; NOT25 = non-25 rows pooled (15/35/50/75/100), 458 rows; "
                  "NOT25_NO15 = heat flux not in {15, 25}, 455 rows",
            models={mid: {k: clean(v) for k, v in legacy[mid].items()} for mid in MODEL_ORDER}),
        label_checks=dict(masonite15_share_of_NOT25_SS=r4(share_masonite), F25_share_of_all_SS=r4(share_f25)),
        reference_ceilings={k: {kk: (r4(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                            for k, v in ceilings.items()},
        runtime_s=round(time.time() - t_start),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}; runtime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
