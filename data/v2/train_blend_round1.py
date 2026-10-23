#!/usr/bin/env python3
"""Blend round 1 -- follows experiments/configs/blend_round1.json exactly (b1, b2 per target; nothing tuned).

The LOI round-8 averaging recipe tested once on pHRR, Char Residue, FIGRA and TTI. Data prep for each target is
copied from (or imported from) the script behind its record; each incumbent must reproduce its record first.

Run one target: OMP_NUM_THREADS=1 python3 data/v2/train_blend_round1.py --target pHRR|CR|FIGRA|TTI
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402


class Avg(BaseEstimator, RegressorMixin):
    def __init__(self, ests):
        self.ests = ests

    def fit(self, X, y):
        self.f_ = [clone(e).fit(X, y) for e in self.ests]
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.f_], axis=0)


def et():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10.0))


def log1p_wrap(m):
    return TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)


# ---------------- per-target data (copied from the record scripts) ----------------
def data_phrr():
    import train_phrr_v3_robust as T
    T.EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")  # as train_phrr_promote_c2.py
    full = T.load(replicate_target=True)
    df = full[full["has_mcc"]].reset_index(drop=True)
    mat = df["material"].to_numpy()
    grp = np.where(mat == "PlasticC", "HIPS", mat)
    feats = T.BASE + T.PHYS
    assert (len(df), len(np.unique(grp))) == (191, 64)
    return dict(X=df[feats].to_numpy(float), y=df[T.TARGET].to_numpy(float), g=grp, log=True, inc=T.hgb(),
                feats=feats, ref=(0.6113, 0.516, 0.838), sota=0.30, sota_src="pomazi_2026", exp="exp074",
                extra=("category-GKF", df["category"].to_numpy()), target="Peak HRR (kW/m2)",
                dataset="FSRI pHRR replicate-mean, has_mcc (191 rows / 64 merged groups), as exp074")


def data_cr():
    cone = pd.read_csv(f"{HERE}/char_residue_cone_fsri.csv")
    sta = pd.read_csv(f"{HERE}/char_yield_sta.csv")
    mcc = pd.read_csv(f"{HERE}/mcc_hrr.csv")
    props = pd.read_csv(f"{HERE}/material_props_fsri.csv")
    sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct", aggfunc="mean")
                .rename(columns=lambda r: f"sta_res_{int(r)}K").reset_index())
    mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                          mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean")).reset_index()
    feat = (sta_wide.merge(mcc_agg, on="material", how="outer")
            .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left"))
    df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()].merge(feat, on="material", how="left")
    sta_cols = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
    df = df[df[sta_cols].notna().any(axis=1) & df.mcc_char_yield_pct.notna()].reset_index(drop=True)
    groups = np.where(df.material.values == "PlasticC", "HIPS", df.material.values)
    base = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct"] + sta_cols
    cat_l = df.categories.fillna("").str.lower()
    flags = []
    for c in ["interior finish", "exterior finish", "insulation", "roofing", "structural"]:
        col = "cat_" + c.replace(" ", "_")
        df[col] = cat_l.apply(lambda s, c=c: float(c in [t.strip() for t in s.split(";")]))
        flags.append(col)
    hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)
    return dict(X=df[base + flags].values.astype(float), y=df.cone_residue_pct.values.astype(float), g=groups, log=False,
                inc=hgb, feats=base + flags, ref=(0.5491, 0.342, 0.693), sota=0.82, sota_src="pomazi_2026", exp="exp081",
                extra=None, target="Char Residue (cone residue %, FSRI)",
                dataset=f"char_residue_cone_fsri x STA/MCC/props ({len(df)} rows / {len(set(groups))} merged groups), as exp081")


def data_figra():
    reps = pd.read_csv(f"{HERE}/figra_fsri.csv")
    reps = reps[reps.spark_ignitor & reps.ignition_recorded]
    df = (reps.groupby(["material", "heat_flux_kWm2"])
          .agg(figra_kWm2s=("figra_kWm2s", "mean"), phrr_kWm2=("peak_hrrpua_kWm2", "mean"),
               t_phrr_s=("time_to_peak_s", "mean"), n_reps=("figra_kWm2s", "size")).reset_index())
    mcc = pd.read_csv(f"{HERE}/mcc_hrr.csv").groupby("material").agg(
        mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean"))
    extras = pd.read_csv(f"{HERE}/mcc_extras_fsri.csv").groupby("material").agg(
        mcc_hrc_JgK=("hrc_JgK", "mean"), mcc_total_hr_kJg=("total_hr_kJg", "mean"), mcc_t_peak_C=("t_peak_C", "mean"))
    props = pd.read_csv(f"{HERE}/material_props_fsri.csv").set_index("material")[
        ["categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
    df = df.join(mcc, on="material").join(extras, on="material").join(props, on="material")
    f2 = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "density_kgm3", "cone_thickness_mm",
          "cone_mass_per_area_kgm2", "mcc_hrc_JgK", "mcc_total_hr_kJg", "mcc_t_peak_C"]
    d = df[df.mcc_peak_hrr_Wg.notna()].reset_index(drop=True)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=0)
    return dict(X=d[f2].to_numpy(float), y=d.figra_kWm2s.to_numpy(float), g=d.material.to_numpy(), log=True, inc=hgb,
                feats=f2, ref=(0.4563, 0.245, 0.639), sota=0.9724, sota_src="phan_morgan", exp="exp073", extra=None,
                target="Fire Growth Rate FIGRA (kW/m2/s)",
                dataset=f"figra_fsri replicate mean per material x flux, materials with MCC ({len(d)} rows / {d.material.nunique()} materials), as exp073")


def data_tti():
    argv = sys.argv
    sys.argv = [argv[0], "--onset-bc", "--replicate-table"]
    import train_tti_v5_physics as v5
    sys.argv = argv
    from sklearn.linear_model import PoissonRegressor
    df = v5.df
    assert len(df) == 681 and df["material"].nunique() == 76
    return dict(X=df[v5.PHYS].to_numpy(float), y=df[v5.TARGET].to_numpy(float), g=df["material"].to_numpy(), log=False,
                inc=v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), feats=v5.PHYS, ref=(0.3133, 0.071, 0.496),
                sota=0.58, sota_src="pomazi_2026", exp="exp046", extra=("family-GKF", df["family"].to_numpy()),
                f25=df["heat_flux_kWm2"].to_numpy() == 25, target="Time to Ignition (s)",
                dataset="FSRI cone replicates minus did_not_ignite/early_stop (681 rows / 76 materials), as exp046")


DATA = {"pHRR": data_phrr, "CR": data_cr, "FIGRA": data_figra, "TTI": data_tti}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(DATA))
    tname = ap.parse_args().target
    t0 = time.time()
    D = DATA[tname]()
    X, y, g, logt = D["X"], D["y"], D["g"], D["log"]
    if tname == "TTI":  # GLM on raw scale; ET and Ridge each in log1p, averaged on the raw scale
        cands = {"exp": D["inc"], "b1": Avg([D["inc"], log1p_wrap(et()), log1p_wrap(ridge())]),
                 "b2": Avg([D["inc"], log1p_wrap(et())])}
    else:
        cands = {"exp": D["inc"], "b1": Avg([D["inc"], et(), ridge()]), "b2": Avg([D["inc"], et()])}
    print(f"{tname}: {len(y)} rows / {len(np.unique(g))} groups; log1p wrapper on whole model: {logt}")
    res = {}
    for cid, m in cands.items():
        t = time.time()
        rk, oof = repeated_group_kfold(m, X, y, g, log_target=logt)
        lo, hi = bootstrap_ci(y, oof[0], g)
        r = dict(rk=rk, mean=rk.mean(), std=rk.std(), lo=lo, hi=hi)
        if D["extra"] is not None:
            ek, _ = repeated_group_kfold(m, X, y, D["extra"][1], log_target=logt)
            r["extra"] = ek.mean()
        if "f25" in D:
            r["r25"] = r2_score(y[D["f25"]], oof[0][D["f25"]])
        res[cid] = r
        ex = f"  {D['extra'][0]} {r['extra']:.3f}" if "extra" in r else ""
        ex += f"  R2@25 {r['r25']:.3f}" if "r25" in r else ""
        print(f"{cid:4s} GKF {rk.mean():.4f} +- {rk.std():.4f} CI [{lo:.4f}, {hi:.4f}]{ex}  ({time.time()-t:.0f}s)", flush=True)
        if cid == "exp":
            m0, l0, h0 = D["ref"]
            if not (abs(rk.mean() - m0) <= 0.002 and abs(lo - l0) <= 0.002 and abs(hi - h0) <= 0.002):
                print(f"BLOCKED: {D['exp']} not reproduced within 0.002; stopping.")
                sys.exit(2)
            print(f"REPRO {D['exp']} ok")
    inc = res["exp"]
    for cid in ("b1", "b2"):
        res[cid]["wins"] = int((res[cid]["rk"] > inc["rk"]).sum())
        res[cid]["delta"] = float((res[cid]["rk"] - inc["rk"]).mean())
        print(f"  {cid} vs {D['exp']}: dGKF {res[cid]['delta']:+.4f}, wins {res[cid]['wins']}/10")
    best = max(("b1", "b2"), key=lambda c: res[c]["mean"])
    rb = res[best]
    adopt = rb["wins"] >= 8 and rb["lo"] >= D["ref"][1]
    print(f"SELECTION {tname}: {best} {rb['mean']:.4f}; wins {rb['wins']}/10 (need >= 8), ci_lo {rb['lo']:.4f} "
          f"(need >= {D['ref'][1]}) -> " + (f"ADOPT over {D['exp']} pending critic" if adopt else f"NOT adopted; {D['exp']} stays"))
    print("Triggers:", [c for c in ("b1", "b2") if res[c]["lo"] > D["sota"]] or "none")
    DESC = {"b1": "Avg(incumbent, ExtraTrees 500/2, Ridge alpha 10)", "b2": "Avg(incumbent, ExtraTrees 500/2)"}
    for cid in ("b1", "b2"):
        r = res[cid]
        note = f"blend round 1 {tname} {cid}"
        if "extra" in r:
            note += f"; {D['extra'][0]} {r['extra']:.3f} (incumbent {inc['extra']:.3f})"
        if "r25" in r:
            note += f"; R2 at 25 kW/m2 {r['r25']:.3f} (incumbent {inc['r25']:.3f})"
        out = dict(slug=f"blend_r1_{tname.lower()}_{cid}", target=D["target"], round_config="experiments/configs/blend_round1.json",
                   dataset=D["dataset"], features=D["feats"], model={"b1": "Avg(incumbent, ExtraTrees, Ridge)", "b2": "Avg(incumbent, ExtraTrees)"}[cid],
                   hyperparameters=DESC[cid] + "; fixed a priori", code="data/v2/train_blend_round1.py",
                   n_rows=int(len(y)), n_groups=int(len(np.unique(g))), protocol="gkf_group5x10",
                   rkf_mean=round(float(r["mean"]), 4), rkf_std=round(float(r["std"]), 4), ci_lo=round(float(r["lo"]), 4),
                   ci_hi=round(float(r["hi"]), 4), paired_vs=D["exp"], paired_delta=round(r["delta"], 4),
                   paired_wins=f"{r['wins']}/10", sota=D["sota"], sota_source=D["sota_src"], status="ok", blocked_reason="",
                   note=note)
        with open(f"{ROOT}/experiments/results/{out['slug']}.json", "w") as f:
            json.dump(out, f, indent=2)
    print(f"wrote 2 result JSONs; runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
