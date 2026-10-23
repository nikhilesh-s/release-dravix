#!/usr/bin/env python3
"""Dravix V2 -- exp001 successor: pHRR with the CORRECTED STA char feature (2026-09-29, stream sta_fix).

WHY: exp001's sta_char_yield_pct was the material mean of char_yield_sta.csv's
final_residual_mass_pct, which before the unit fix mixed mg lost (93% of rows)
with % (the 4 '%'-export materials). exp002 then dropped STA and did better,
partly on more rows. This reruns exp001's exact config -- features, inner join,
RF(300, max_depth=6, min_samples_leaf=2) -- with the corrected feature, on ONE row
set (exp001's phrr x MCC x STA join), three ways:
  (1) exp002 features (no STA); (2) + old committed STA 'char'; (3) + corrected STA residue.
The old value comes from the audit probe's per-file table
(reports/sprint_0929/repro_audit_scripts/probe_sta_rows.csv, column
final_residual_mass_pct = the pre-fix committed value). Nothing tuned; one variant.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from eval_paired import print_paired
from eval_protocol import report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(DATA_DIR, "..", "..", "reports", "sprint_0929", "repro_audit_scripts", "probe_sta_rows.csv")
RF = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)

phrr = pd.read_csv(f"{DATA_DIR}/phrr_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
new = pd.read_csv(f"{DATA_DIR}/char_yield_sta.csv")
old = pd.read_csv(PROBE)

mcc_agg = mcc.groupby("material").agg(mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
                                      mcc_char_yield_pct=("char_yield_pct", "mean")).reset_index()
sta = (new.groupby("material").final_residual_mass_pct.mean().rename("sta_char_fixed_pct").to_frame()
       .join(old.groupby("material").final_residual_mass_pct.mean().rename("sta_char_old")).reset_index())
df = phrr.merge(mcc_agg, on="material", how="inner").merge(sta, on="material", how="inner")
BASE = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
df = df.dropna(subset=BASE + ["sta_char_old", "sta_char_fixed_pct", "peak_hrrpua_kWm2"]).reset_index(drop=True)
y, groups = df.peak_hrrpua_kWm2.values, df.material.values
print(f"Same rows for all three: {len(df)} rows / {df.material.nunique()} materials (exp001 had 158/55)")

SETS = {"exp002 features (no STA)": BASE,
        "+ old committed STA 'char' (exp001 as run)": BASE + ["sta_char_old"],
        "+ corrected STA residue": BASE + ["sta_char_fixed_pct"]}
for name, cols in SETS.items():
    X = df[cols].values
    leg = []
    for seed in range(10):  # exp001's exact loop (RF random_state=seed)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
        mdl = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, min_samples_leaf=2)
        leg.append(r2_score(y[te], mdl.fit(X[tr], y[tr]).predict(X[te])))
    print(f"\n{name}\n  exp001 exact loop (rs=seed): mean R2={np.mean(leg):.4f} std={np.std(leg):.4f} "
          f"min={np.min(leg):.4f} max={np.max(leg):.4f}")
    report(name, RF, X, y, groups, sota=0.30)

print("\nPaired same-rows same-folds deltas (repeated GroupKFold 5x10):")
print_paired("corrected STA vs no STA", RF, df[BASE].values, df[BASE + ["sta_char_fixed_pct"]].values, y, groups)
print_paired("corrected STA vs old STA", RF, df[BASE + ["sta_char_old"]].values, df[BASE + ["sta_char_fixed_pct"]].values, y, groups)
