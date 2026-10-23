#!/usr/bin/env python3
"""TTI round 2 (emissivity) -- follows experiments/configs/tti_round2.json exactly (e1-e4; nothing tuned).

Same base as train_tti_round1.py: train_tti_v5_physics.py imported with --onset-bc --replicate-table builds the
exp046 table. The only new source is data/v2/emissivity_fsri.csv (left join on material; ban lifted for this
source only). Step 1 reproduces exp046 or stops BLOCKED.

Run: OMP_NUM_THREADS=1 python3 data/v2/train_tti_round2.py | tee experiments/runs/train_tti_round2.log
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
sys.argv = [sys.argv[0], "--onset-bc", "--replicate-table"]
import train_tti_v5_physics as v5  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402

T0 = time.time()
SOTA = 0.58
SIGMA = 5.670e-8
df = v5.df.copy()
TARGET = v5.TARGET
assert len(df) == 681 and df["material"].nunique() == 76, (len(df), df["material"].nunique())

# ---- emissivity join + physics terms (config, fixed a priori) ----
em = pd.read_csv(ROOT / "data/v2/emissivity_fsri.csv").set_index("material")
TGRID = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
q_w = df["heat_flux_kWm2"].astype(float) * 1e3
T_h = (q_w / SIGMA) ** 0.25


def eps_at(mat, t):
    if mat not in em.index:
        return np.nan
    return float(np.interp(t, TGRID, [em.loc[mat, f"emissivity_{k}K"] for k in TGRID]))


df["eps_h"] = [eps_at(m, t) for m, t in zip(df["material"], T_h)]
df["eps_s"] = [em.loc[m, "emissivity_600K"] if m in em.index else np.nan for m in df["material"]]
df["log_eps_h"] = np.log(df["eps_h"])
Tk, T0k = df["t_onset_C"] + 273.15, v5.T0_C + 273.15
q_cr_eps = df["eps_s"] * SIGMA * (Tk ** 4 - T0k ** 4) + 10.0 * (Tk - T0k)
exc = df["eps_h"] * q_w - q_cr_eps
df["log_q_excess_eps"] = np.where(exc > 0, np.log(exc.where(exc > 0)), np.nan)
df["has_emissivity"] = df["material"].isin(em.index).astype(float)
cov = df["eps_h"].notna()
print(f"rows {len(df)} / materials {df['material'].nunique()}; emissivity rows {int(cov.sum())} / materials "
      f"{df.loc[cov, 'material'].nunique()}; log_q_excess_eps rows {int(df['log_q_excess_eps'].notna().sum())}; "
      f"T_h at 25/50/75 kW/m2: {', '.join(f'{(q * 1e3 / SIGMA) ** 0.25:.0f} K' for q in (25, 50, 75))}")

CANDS = {  # id: (estimator, features, log1p target?)
    "exp046": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS, False),
    "e1": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS + ["log_eps_h"], False),
    "e2": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS + ["log_eps_h", "log_q_excess_eps"], False),
    "e3": (HistGradientBoostingRegressor(loss="poisson", **v5.HGB), v5.F3 + ["log_eps_h", "log_q_excess_eps"], False),
    "e4": (v5.linear(PoissonRegressor(alpha=0.01, max_iter=3000)), v5.PHYS + ["has_emissivity"], False),
}
y = df[TARGET].to_numpy(float)
g = df["material"].to_numpy()
fam = df["family"].to_numpy()
f25 = df["heat_flux_kWm2"].to_numpy() == 25
cv = cov.to_numpy()
res = {}
for cid, (est, feats, logt) in CANDS.items():
    t = time.time()
    X = df[feats].to_numpy(float)
    rk, oof = repeated_group_kfold(est, X, y, g, log_target=logt)
    lo, hi = bootstrap_ci(y, oof[0], g)
    fk, _ = repeated_group_kfold(est, X, y, fam, log_target=logt)
    res[cid] = dict(rk=rk, mean=rk.mean(), std=rk.std(), lo=lo, hi=hi, fam=fk.mean(),
                    r25=r2_score(y[f25], oof[0][f25]), rother=r2_score(y[~f25], oof[0][~f25]),
                    rcov=r2_score(y[cv], oof[0][cv]), r25cov=r2_score(y[cv & f25], oof[0][cv & f25]),
                    maxpred=float(oof[0].max()), feats=feats)
    r = res[cid]
    print(f"{cid:7s} GKF {rk.mean():.4f} +- {rk.std():.4f} CI [{lo:.3f}, {hi:.3f}]  family-GKF {fk.mean():.3f}  "
          f"R2@25 {r['r25']:.3f} other {r['rother']:.3f}  eps-rows {r['rcov']:.3f} (@25 {r['r25cov']:.3f})  "
          f"max OOF {oof[0].max():.0f}s  ({time.time()-t:.0f}s)")
    if cid == "exp046":
        if not (abs(rk.mean() - 0.313) <= 0.002 and abs(lo - 0.071) <= 0.002 and abs(hi - 0.496) <= 0.002):
            print("BLOCKED: exp046 not reproduced within 0.002; stopping.")
            sys.exit(2)
        print("REPRO exp046 ok")

inc = res["exp046"]["rk"]
for cid in ("e1", "e2", "e3", "e4"):
    res[cid]["wins"] = int((res[cid]["rk"] > inc).sum())
    res[cid]["delta"] = float((res[cid]["rk"] - inc).mean())
best = max(("e1", "e2", "e3"), key=lambda c: res[c]["mean"])
rb = res[best]
ctrl = res["e4"]["delta"] >= rb["delta"]
adopt = rb["wins"] >= 8 and rb["lo"] >= 0.071 and not ctrl
print(f"\ncontrol e4 (has_emissivity only): dGKF {res['e4']['delta']:+.4f} vs {best} dGKF {rb['delta']:+.4f}"
      + (" -> gain matched by the identity control" if ctrl else ""))
print(f"SELECTION (highest GKF mean of e1-e3): {best} {rb['mean']:.4f}; paired wins {rb['wins']}/10 (need >= 8), "
      f"ci_lo {rb['lo']:.4f} (need >= 0.071) -> " + ("ADOPT over exp046 pending critic" if adopt
                                                    else "NOT adopted; exp046 stays the record"))
trig = [f"TRIGGER: ci_lo ({c}) {res[c]['lo']:.3f} > {SOTA}" for c in ("e1", "e2", "e3", "e4") if res[c]["lo"] > SOTA]
print("Triggers:", trig if trig else "none")

DESC = {"e1": "Poisson GLM on exp046 PHYS + log eps(T_heater) (FSRI IS emissivity)",
        "e2": "Poisson GLM on PHYS + log eps_h + log(eps_h q'' - q_cr(eps_s))",
        "e3": "HGB poisson on v5 F3 + log eps_h + log(eps_h q'' - q_cr(eps_s))",
        "e4": "Poisson GLM on PHYS + has_emissivity indicator (identity control)"}
SHORT = {"e1": "glm_logeps", "e2": "glm_eps_qexcess", "e3": "hgbpois_eps", "e4": "glm_hasemis_control"}
for cid in ("e1", "e2", "e3", "e4"):
    r = res[cid]
    out = dict(slug=f"tti_r2_{cid}_{SHORT[cid]}", target="Time to Ignition (s)",
               round_config="experiments/configs/tti_round2.json",
               dataset="data/v2/cone_replicates_fsri.csv minus did_not_ignite/early_stop (681 rows / 76 materials), "
                       "left joins as exp046 + data/v2/emissivity_fsri.csv (383 rows / 42 materials covered)",
               features=r["feats"], model=DESC[cid].split(" ")[0] + " " + DESC[cid].split(" ")[1],
               hyperparameters=DESC[cid] + "; fixed a priori", code="data/v2/train_tti_round2.py",
               n_rows=int(len(y)), n_groups=76, protocol="gkf_material5x10",
               rkf_mean=round(float(r["mean"]), 4), rkf_std=round(float(r["std"]), 4),
               ci_lo=round(float(r["lo"]), 4), ci_hi=round(float(r["hi"]), 4),
               paired_vs="exp046", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10", sota=SOTA,
               sota_source="pomazi_2026", status="ok", blocked_reason="",
               note=(f"family-GKF {r['fam']:.3f}; R2 at 25 kW/m2 {r['r25']:.3f}, other fluxes {r['rother']:.3f}; "
                     f"emissivity-covered rows {r['rcov']:.3f} (at 25: {r['r25cov']:.3f}); max OOF {r['maxpred']:.0f}s; "
                     f"TTI round 2 per tti_round2.json"))
    (ROOT / f"experiments/results/tti_r2_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 4 result JSONs; runtime {time.time()-T0:.0f}s")
