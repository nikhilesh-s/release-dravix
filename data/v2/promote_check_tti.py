#!/usr/bin/env python3
"""Dravix V2 -- exp011. Promotion check for TTI (exp009's feature set).

Same discipline as promote_check_phrr.py (exp006): exp009's mean R2=0.267
was a 10-seed CV mean. This commits to ONE new split (seed=42, matching
the pHRR promotion check's convention, chosen before running) as the
actual claim, not the CV average.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = "~/Dravix/dravix-v2/data/v2"

tti = pd.read_csv(f"{DATA_DIR}/tti_fsri.csv")
k_df = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
cp_df = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")
rho_df = pd.read_csv(f"{DATA_DIR}/density_fsri.csv")

k_agg = k_df.groupby("material")["Average_Cond"].mean().rename("k_W_mK").reset_index()
cp_agg = cp_df.groupby("material")["Spc_Heat"].mean().rename("rhocp_J_m3K").reset_index()
rho_agg = rho_df.groupby("material")["density_kg_m3"].mean().reset_index()

df = (tti
      .merge(k_agg, on="material", how="inner")
      .merge(cp_agg, on="material", how="inner")
      .merge(rho_agg, on="material", how="inner"))
df["thermal_inertia"] = df["k_W_mK"] * df["rhocp_J_m3K"]
df["spark_ignitor"] = df["spark_ignitor"].astype(int)

FEATURES = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor",
            "k_W_mK", "rhocp_J_m3K", "density_kg_m3", "thermal_inertia"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES].values
y_raw = df[TARGET].values
y_log = np.log1p(y_raw)
groups = df["material"].values

HELD_OUT_SEED = 42
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=HELD_OUT_SEED)
train_idx, test_idx = next(gss.split(X, y_raw, groups))
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
model.fit(X[train_idx], y_log[train_idx])
preds_raw = np.expm1(model.predict(X[test_idx]))
r2 = r2_score(y_raw[test_idx], preds_raw)

print(f"Locked held-out set (seed={HELD_OUT_SEED}): {len(test_idx)} rows, "
      f"{len(set(groups[test_idx]))} materials, never used to pick features/hyperparameters")
print(f"Held-out R2: {r2:.4f}")
print(f"SOTA (pomazi_2026, real held-out ANN): R2=0.58")
print(f"Delta: {r2 - 0.58:+.4f}")
print(f"Result: {'BEATS SOTA -- promote' if r2 > 0.58 else 'does not beat SOTA on this locked set'}")
