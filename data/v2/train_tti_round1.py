#!/usr/bin/env python3
"""TTI round 1 (V3) -- follows experiments/configs/tti_round1.json exactly (t1-t4; nothing tuned).

Data, joins, physics columns, HGB params and the linear() pipeline come from train_tti_v5_physics.py imported with
--onset-bc --replicate-table (its module-level code builds the exp046 table; its __main__ block does not run).
Step 1 reproduces exp046 (C6 Poisson GLM on PHYS) or stops BLOCKED.

Run: OMP_NUM_THREADS=1 python3 data/v2/train_tti_round1.py | tee experiments/runs/train_tti_round1.log
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
sys.argv = [sys.argv[0], "--onset-bc", "--replicate-table"]
import train_tti_v5_physics as v5  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402

T0 = time.time()
SOTA = 0.58
df = v5.df.copy()
TARGET = v5.TARGET
assert len(df) == 681 and df["material"].nunique() == 76, (len(df), df["material"].nunique())
df["q_kWm2"] = df["heat_flux_kWm2"].astype(float)
df["q_over_dT"] = df["q_kWm2"] / (df["t_onset_C"] - v5.T0_C)
print(f"rows {len(df)} / materials {df['material'].nunique()}; log_q_excess coverage "
      f"{df['log_q_excess'].notna().sum()} rows / {df.loc[df['log_q_excess'].notna(), 'material'].nunique()} materials; "
      f"rows with q'' <= q_cr: {(df['q_kWm2'] <= df['q_cr_kWm2']).sum()}")


def inv_z(z):
    return 1.0 / np.maximum(z, 0.02) ** 2      # cap 2500 s, declared


CANDS = {  # id: (estimator, features, log1p target?)
    "exp046": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS, False),
    "t1": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS + ["log_q_excess"], False),
    "t2": (TransformedTargetRegressor(regressor=v5.linear(Ridge(alpha=1.0)), func=lambda t: t ** -0.5,
                                      inverse_func=inv_z, check_inverse=False),
           ["q_kWm2", "q_over_dT", "log_L", "log_rho", "log_kpc"], False),
    "t3": (HistGradientBoostingRegressor(loss="poisson", **v5.HGB), v5.F3 + ["log_q_excess", "log_t_thick_loss"], False),
    "t4": (HistGradientBoostingRegressor(**v5.HGB), v5.F0, True),
}
y = df[TARGET].to_numpy(float)
g = df["material"].to_numpy()
fam = df["family"].to_numpy()
f25 = df["heat_flux_kWm2"].to_numpy() == 25
res = {}
for cid, (est, feats, logt) in CANDS.items():
    t = time.time()
    X = df[feats].to_numpy(float)
    rk, oof = repeated_group_kfold(est, X, y, g, log_target=logt)
    lo, hi = bootstrap_ci(y, oof[0], g)
    fk, _ = repeated_group_kfold(est, X, y, fam, log_target=logt)
    res[cid] = dict(rk=rk, mean=rk.mean(), std=rk.std(), lo=lo, hi=hi, fam=fk.mean(),
                    r25=r2_score(y[f25], oof[0][f25]), rother=r2_score(y[~f25], oof[0][~f25]),
                    maxpred=float(oof[0].max()), feats=feats)
    print(f"{cid:7s} GKF {rk.mean():.4f} +- {rk.std():.4f} CI [{lo:.3f}, {hi:.3f}]  family-GKF {fk.mean():.3f}  "
          f"R2@25 {res[cid]['r25']:.3f} other {res[cid]['rother']:.3f}  max OOF {oof[0].max():.0f}s  ({time.time()-t:.0f}s)")
    if cid == "exp046":
        if not (abs(rk.mean() - 0.313) <= 0.002 and abs(lo - 0.071) <= 0.002 and abs(hi - 0.496) <= 0.002):
            print("BLOCKED: exp046 not reproduced within 0.002; stopping.")
            sys.exit(2)
        print("REPRO exp046 ok")

inc = res["exp046"]["rk"]
for cid in ("t1", "t2", "t3", "t4"):
    res[cid]["wins"] = int((res[cid]["rk"] > inc).sum())
    res[cid]["delta"] = float((res[cid]["rk"] - inc).mean())
best = max(("t1", "t2", "t3", "t4"), key=lambda c: res[c]["mean"])
rb = res[best]
adopt = rb["wins"] >= 8 and rb["lo"] >= 0.071
print(f"\nSELECTION (highest GKF mean): {best} {rb['mean']:.4f}; paired wins {rb['wins']}/10 (need >= 8), "
      f"ci_lo {rb['lo']:.4f} (need >= 0.071) -> " + ("ADOPT over exp046 pending critic" if adopt
                                                    else "NOT adopted; exp046 stays the record; FLAG TTI parked"))
trig = [f"TRIGGER: ci_lo ({c}) {res[c]['lo']:.3f} > {SOTA}" for c in ("t1", "t2", "t3", "t4") if res[c]["lo"] > SOTA]
print("Triggers:", trig if trig else "none")

DESC = {"t1": "Poisson GLM on exp046 PHYS + log(q''-q_cr) (q_cr from MCC onset, eps 1, h 10)",
        "t2": "Ridge on TTI^-1/2 (q'', q''/dT, log L, log rho, log kpc), back-transform 1/max(z,0.02)^2",
        "t3": "HGB poisson on v5 F3 + log_q_excess + log_t_thick_loss",
        "t4": "HGB log1p on v5 F0 (model-family control)"}
SHORT = {"t1": "glm_qexcess", "t2": "ridge_invsqrt", "t3": "hgbpois_chf", "t4": "hgb_f0"}
for cid in ("t1", "t2", "t3", "t4"):
    r = res[cid]
    out = dict(slug=f"tti_r1_{cid}_{SHORT[cid]}", target="Time to Ignition (s)",
               round_config="experiments/configs/tti_round1.json",
               dataset="data/v2/cone_replicates_fsri.csv minus did_not_ignite/early_stop (681 rows / 76 materials), "
                       "left joins as exp046",
               features=r["feats"], model=DESC[cid].split(" ")[0] + " " + DESC[cid].split(" ")[1],
               hyperparameters=DESC[cid] + "; fixed a priori", code="data/v2/train_tti_round1.py",
               n_rows=int(len(y)), n_groups=76, protocol="gkf_material5x10",
               rkf_mean=round(float(r["mean"]), 4), rkf_std=round(float(r["std"]), 4),
               ci_lo=round(float(r["lo"]), 4), ci_hi=round(float(r["hi"]), 4),
               paired_vs="exp046", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10", sota=SOTA,
               sota_source="pomazi_2026", status="ok", blocked_reason="",
               note=(f"family-GKF {r['fam']:.3f}; R2 at 25 kW/m2 {r['r25']:.3f}, other fluxes {r['rother']:.3f}; "
                     f"max OOF {r['maxpred']:.0f}s; TTI round 1 per tti_round1.json"))
    (ROOT / f"experiments/results/tti_r1_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 4 result JSONs; runtime {time.time()-T0:.0f}s")
