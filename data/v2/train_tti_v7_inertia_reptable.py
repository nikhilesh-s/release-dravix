#!/usr/bin/env python3
"""Dravix V2 -- exp053. TTI: real thermal inertia on the canonical per-replicate table.

Combines two independent findings from the 2026-09-29/30 sprints that never got
tried together:
  - the canonical per-replicate cone table (cone_replicates_fsri.csv), which fixes
    did_not_ignite (45 rows) and early_stop (9 rows) handling and recovers 16 more
    Frame+Grid replicates that tti_fsri.csv missed -- exp047's baseline on this
    table (mcc features only) scored GKF mean R2=0.2613, CI [0.130, 0.441].
  - real thermal inertia (k * rhocp, from the HFM instrument) as a TTI feature,
    the textbook ignition-theory predictor -- exp048 (old exp009) found it raised
    the legacy-protocol CV mean from 0.167 to 0.267 on tti_fsri.csv (53 materials),
    but that table drops the did_not_ignite/early_stop rows incorrectly and misses
    the 16 Frame+Grid replicates.

This script re-runs the thermal-inertia feature set on the corrected table, scored
with the same repeated-GroupKFold + bootstrap-CI protocol as exp047, so the two
numbers are directly comparable.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from eval_protocol import report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

cr = pd.read_csv(f"{DATA_DIR}/cone_replicates_fsri.csv")
k_df = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
cp_df = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")

tti = cr[~cr["did_not_ignite"].astype(bool) & ~cr["early_stop"].astype(bool)].copy()

k_agg = k_df.groupby("material")["Average_Cond"].mean().rename("k_W_mK").reset_index()
cp_agg = cp_df.groupby("material")["Spc_Heat"].mean().rename("rhocp_J_m3K").reset_index()
rho_agg = props[["material", "density_kgm3"]].rename(columns={"density_kgm3": "density_kg_m3"})

df = (tti
      .merge(k_agg, on="material", how="inner")
      .merge(cp_agg, on="material", how="inner")
      .merge(rho_agg, on="material", how="inner"))
df["thermal_inertia"] = df["k_W_mK"] * df["rhocp_J_m3K"]
df["spark_ignitor"] = df["spark_ignitor"].astype(int)

FEATURES = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor",
            "k_W_mK", "rhocp_J_m3K", "density_kg_m3", "thermal_inertia"]
TARGET = "tti_s"
df = df.dropna(subset=FEATURES + [TARGET])

print(f"Rows: {len(df)} / {df['material'].nunique()} materials "
      f"(exp047 baseline on the same corrected table: 681 rows / 76 materials, no thermal props)")

X = df[FEATURES].values
y = df[TARGET].values
groups = df["material"].values

model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
out = report("exp053 TTI + thermal inertia (reptable)", model, X, y, groups, log_target=True, sota=0.58)

print()
print(f"For comparison, exp047 (same table, no thermal props): GKF mean R2=0.2613, CI [0.130, 0.441]")
print(f"Delta (GKF mean, this - exp047): {out['rkf_mean'] - 0.2613:+.4f}")

full_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
full_model.fit(X, np.log1p(y))
print()
print("Feature importances (full-data fit):")
for feat, imp in sorted(zip(FEATURES, full_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")
