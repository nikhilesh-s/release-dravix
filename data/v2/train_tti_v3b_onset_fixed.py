#!/usr/bin/env python3
"""Dravix V2 -- exp007 successor: TTI with the CORRECTED STA onset (2026-09-29, stream sta_fix).

WHY: exp007 reported onset lifting TTI R2 0.10 -> 0.167, but its onset was mostly
the STA scan start temperature (unit bug, repro audit 3c), and the audit showed
the "gain" was a row-set change (0.233 -> 0.233 on identical rows). This reruns
exp007's exact config -- features, RF(300, max_depth=8, min_samples_leaf=2),
log1p target scored in seconds -- on ONE row set, three ways:
  (1) exp004 features, no onset; (2) + old committed onset; (3) + corrected onset.
Rows = exp007's TTI x MCC x onset join restricted to rows that have BOTH onset
versions (Mineral_Wool_Insulation drops: it never loses 5%, so no corrected onset).
The old onset comes from the audit probe's per-file table
(reports/sprint_0929/repro_audit_scripts/probe_sta_rows.csv, column
onset_temp_5pct_C = the pre-fix committed value), so this script does not depend
on git state. Nothing tuned; one variant.
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
RF = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=0)

tti = pd.read_csv(f"{DATA_DIR}/tti_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
new = pd.read_csv(f"{DATA_DIR}/onset_temp_sta.csv")
old = pd.read_csv(PROBE)

mcc_agg = mcc.groupby("material").agg(mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
                                      mcc_char_yield_pct=("char_yield_pct", "mean")).reset_index()
on = (new.groupby("material").onset_temp_5pct_C.mean().rename("onset_fixed_C").to_frame()
      .join(old.groupby("material").onset_temp_5pct_C.mean().rename("onset_old_C")).reset_index())
df = tti.merge(mcc_agg, on="material", how="inner").merge(on, on="material", how="inner")
df["spark_ignitor"] = df["spark_ignitor"].astype(int)
BASE = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
df = df.dropna(subset=BASE + ["onset_old_C", "onset_fixed_C", "time_to_ignition_s"]).reset_index(drop=True)
y, groups = df.time_to_ignition_s.values, df.material.values
print(f"Same rows for all three: {len(df)} rows / {df.material.nunique()} materials")
print("onset material means (old -> fixed), spot checks:")
print(df.groupby("material")[["onset_old_C", "onset_fixed_C"]].first()
      .loc[lambda d: d.index.isin(["ABS", "HDPE", "PET", "Oak_Flooring", "Black_PMMA", "Gypsum_Wallboard"])].round(1).to_string())

SETS = {"exp004 features, no onset": BASE,
        "+ old committed onset (exp007 as run)": BASE + ["onset_old_C"],
        "+ corrected onset": BASE + ["onset_fixed_C"]}
for name, cols in SETS.items():
    X = df[cols].values
    leg = []
    for seed in range(10):  # exp007's exact loop (RF random_state=seed), for leaderboard comparability
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
        mdl = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, min_samples_leaf=2)
        leg.append(r2_score(y[te], np.expm1(mdl.fit(X[tr], np.log1p(y[tr])).predict(X[te]))))
    print(f"\n{name}\n  exp007 exact loop (rs=seed): mean R2={np.mean(leg):.4f} std={np.std(leg):.4f} "
          f"min={np.min(leg):.4f} max={np.max(leg):.4f}")
    report(name, RF, X, y, groups, log_target=True, sota=0.58)

print("\nPaired same-rows same-folds deltas (repeated GroupKFold 5x10):")
print_paired("corrected onset vs no onset", RF, df[BASE].values, df[BASE + ["onset_fixed_C"]].values, y, groups, log_target=True)
print_paired("corrected onset vs old onset", RF, df[BASE + ["onset_old_C"]].values, df[BASE + ["onset_fixed_C"]].values, y, groups,
             log_target=True)
print_paired("old onset vs no onset", RF, df[BASE].values, df[BASE + ["onset_old_C"]].values, y, groups, log_target=True)
