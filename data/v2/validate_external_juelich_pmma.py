#!/usr/bin/env python3
"""Dravix V2 -- external Juelich black/transparent PMMA cone check of the pHRR record (report-only).

Follows experiments/configs/external_juelich_pmma.json exactly (committed before this first run).
No training, tuning, selection or adoption on any Juelich row. No number of record can change.

Run (repo root): OMP_NUM_THREADS=1 .venv312/bin/python data/v2/validate_external_juelich_pmma.py \
                   > experiments/runs/external_juelich_pmma.log
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import train_phrr_v3_robust as rec  # noqa: E402

CONFIG = "experiments/configs/external_juelich_pmma.json"
SRC = os.path.join(ROOT, "data", "sources", "incoming", "juelich_pmma_cone")
OUT = os.path.join(ROOT, "experiments", "runs", "external_juelich_pmma")
REPORT = os.path.join(ROOT, "reports", "external_juelich_pmma.md")
MODEL = os.path.join(ROOT, "models", "v2", "phrr_hgb_v4.joblib")
THICK_MM = 6.0
SHA = {"Black.zip": "9c6344c37722fcd4aef1c9b6fa65b6d9fe6ce7c6ca5fb8a99a9d60afab46d2cf",
       "Transparent.zip": "9d50133f7ea6c6eba41d14c947c0f3458500ee1f7a7426f60343dca9f7738cc0"}
FSRI_MATCH = {"Black": "Black_PMMA", "Transparent": "PMMA"}


def log(*a):
    print(*a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def stop(msg):
    log("STOP:", msg)
    sys.exit(3)


# 0. gates -------------------------------------------------------------------------------------------------
tracked = subprocess.run(["git", "log", "-1", "--format=%H", "--", CONFIG], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "--", CONFIG], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
if not tracked or dirty:
    stop("config is not committed or has uncommitted changes; refusing to run")
log(f"config {CONFIG} committed at {tracked[:7]}")
for name, want in SHA.items():
    got = hashlib.sha256(open(os.path.join(SRC, name), "rb").read()).hexdigest()
    if got != want:
        stop(f"{name} sha256 mismatch")
files = sorted(glob.glob(os.path.join(SRC, "extracted", "*", "PMMA_*_HRR_*kWm2_R*.csv")))
if len(files) != 24:
    stop(f"expected 24 files, found {len(files)}")
log("sha256 verified for both zips; 24 files")

# 1. inputs (no label read) ----------------------------------------------------------------------------------
rec.EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")
frame = rec.load(replicate_target=True)
frame = frame[frame["has_mcc"]]
bundle = joblib.load(MODEL)
feats = bundle["features"]
if len(frame) != 191:
    stop(f"loader returned {len(frame)} rows, expected 191")
if feats != ["heat_flux_kWm2"] + rec.BASE[1:] + rec.PHYS:
    stop(f"model feature order differs from loader: {feats}")
mat_cols = [f for f in feats if f != "heat_flux_kWm2"]
mat = {}
for color, fm in FSRI_MATCH.items():
    sub = frame[frame["material"] == fm]
    if sub.empty:
        stop(f"FSRI material {fm} not in the 191-row frame")
    row = sub.iloc[0]
    allowed_nan = {"log_kpc"} if color == "Transparent" else set()  # config amendment_2
    missing = set(row[mat_cols][row[mat_cols].isna()].index) - allowed_nan
    if missing:
        stop(f"{fm} is missing inputs: {sorted(missing)}")
    mat[color] = row[mat_cols].astype(float).to_dict()
    log(f"{color:11s} -> FSRI {fm}: " + ", ".join(f"{k}={v:.4g}" for k, v in mat[color].items()))

tests = []
for p in files:
    m = re.search(r"PMMA_(Black|Transparent)_HRR_(\d+)kWm2_R(\d)", os.path.basename(p))
    tests.append(dict(color=m.group(1), flux=int(m.group(2)), rep=int(m.group(3)), path=p))
T = pd.DataFrame(tests)


def X_for(variant):
    rows = []
    for _, t in T.iterrows():
        d = dict(mat[t.color])
        d["heat_flux_kWm2"] = float(t.flux)
        if variant == "A":
            d["cone_thickness_mm"] = THICK_MM
            d["cone_mass_per_area_kgm2"] = THICK_MM * 1e-3 * d["density_kgm3"]
        rows.append([d[f] for f in feats])
    return np.array(rows, float)


# 2. predictions -----------------------------------------------------------------------------------------------
for v in ("A", "B"):
    T["pred_" + v] = bundle["model"].predict(X_for(v))
os.makedirs(OUT, exist_ok=True)
T.drop(columns="path").to_csv(os.path.join(OUT, "predictions.csv"), index=False)
json.dump(jsonable(dict(config_commit=tracked, material_inputs=mat, features=feats, thickness_mm=THICK_MM)),
          open(os.path.join(OUT, "inputs.json"), "w"), indent=1)
log("predictions written; reading labels now")

# 3. labels and metrics ---------------------------------------------------------------------------------------
lab, lab_s = [], []
for p in T["path"]:
    h = pd.read_csv(p)["HRR (kW/m2)"].to_numpy(float)
    lab.append(np.nanmax(h))
    lab_s.append(np.nanmax(pd.Series(h).rolling(5, center=True, min_periods=1).mean().to_numpy()))
T["obs"], T["obs_smooth5"] = lab, lab_s
T.drop(columns="path").to_csv(os.path.join(OUT, "tests.csv"), index=False)


def metrics(df, pred, obs="obs"):
    y, p = df[obs].to_numpy(), df[pred].to_numpy()
    return dict(n=len(df), r2=r2_score(y, p) if len(df) > 2 else None, mae=float(np.mean(np.abs(y - p))),
                mae_pct_of_mean=float(np.mean(np.abs(y - p)) / y.mean() * 100), bias=float(np.mean(p - y)),
                mean_obs=float(y.mean()), mean_pred=float(p.mean()))


S = {"config_commit": tracked}
for v in ("A", "B"):
    S[v] = {"all": metrics(T, "pred_" + v)}
    for c in ("Black", "Transparent"):
        S[v][c] = metrics(T[T.color == c], "pred_" + v)
S["A_label_smooth5"] = metrics(T, "pred_A", "obs_smooth5")
cell = T.groupby(["color", "flux"]).agg(obs_min=("obs", "min"), obs_mean=("obs", "mean"), obs_max=("obs", "max"),
                                         pred_A=("pred_A", "first"), pred_B=("pred_B", "first")).reset_index()
S["cells"] = cell.to_dict("records")
json.dump(jsonable(S), open(os.path.join(OUT, "summary.json"), "w"), indent=1)

log("\n=== results (report-only; no win language, no bar) ===")
for v in ("A", "B"):
    for k in ("all", "Black", "Transparent"):
        m = S[v][k]
        r2 = "n/a" if m["r2"] is None else f"{m['r2']:.3f}"
        log(f"variant {v} {k:11s} n={m['n']:2d} R2={r2} MAE={m['mae']:.0f} kW/m2 ({m['mae_pct_of_mean']:.1f}% of mean) "
            f"bias={m['bias']:+.0f}  mean obs {m['mean_obs']:.0f} / pred {m['mean_pred']:.0f}")
m = S["A_label_smooth5"]
log(f"variant A, smoothed-label sensitivity: R2={m['r2']:.3f} MAE={m['mae']:.0f}")
log("\nper (color, flux) cell: observed min/mean/max vs predicted A / B")
log(cell.round(0).to_string(index=False))


# 4. post-hoc variant U (config amendment_3_post_hoc): label / 0.0088 m2, exploratory, label-informed ----------------
AREA = 0.0088
T["obs_U"] = T["obs"] / AREA
U = {"all": metrics(T, "pred_A", "obs_U")}
for c in ("Black", "Transparent"):
    U[c] = metrics(T[T.color == c], "pred_A", "obs_U")
UB = {"all": metrics(T, "pred_B", "obs_U")}
cellU = T.groupby(["color", "flux"]).agg(obsU_min=("obs_U", "min"), obsU_mean=("obs_U", "mean"),
                                          obsU_max=("obs_U", "max"), pred_A=("pred_A", "first"),
                                          pred_B=("pred_B", "first")).reset_index()
S["post_hoc_U"] = dict(A=U, B=UB, cells=cellU.to_dict("records"), area_m2=AREA)
json.dump(jsonable(S), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
log("\n=== POST-HOC variant U (label / 0.0088 m2; exploratory, label-informed, NOT confirmatory) ===")
for k in ("all", "Black", "Transparent"):
    m = U[k]
    log(f"variant A {k:11s} n={m['n']:2d} R2={m['r2']:.3f} MAE={m['mae']:.0f} kW/m2 ({m['mae_pct_of_mean']:.1f}% of mean) "
        f"bias={m['bias']:+.0f}  mean obs {m['mean_obs']:.0f} / pred {m['mean_pred']:.0f}")
m = UB["all"]
log(f"variant B all         n={m['n']:2d} R2={m['r2']:.3f} MAE={m['mae']:.0f} bias={m['bias']:+.0f}")
log(cellU.round(0).to_string(index=False))
