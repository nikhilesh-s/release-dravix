#!/usr/bin/env python3
"""Dravix V2 -- exp010. TTI, take 5: drop spark_ignitor (0.000 importance
in exp009) and specimen_mass_g (weak, 0.083, likely test-geometry noise
rather than a material property) to see if less noise beats more features.
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

FEATURES = ["heat_flux_kWm2", "k_W_mK", "rhocp_J_m3K", "density_kg_m3", "thermal_inertia"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials")

X = df[FEATURES].values
y_raw = df[TARGET].values
y_log = np.log1p(y_raw)
groups = df["material"].values

SEEDS = list(range(10))
scores = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y_raw, groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, min_samples_leaf=2)
    model.fit(X[train_idx], y_log[train_idx])
    preds_raw = np.expm1(model.predict(X[test_idx]))
    r2 = r2_score(y_raw[test_idx], preds_raw)
    scores.append(r2)
    print(f"  seed={seed}: n_test={len(test_idx)}, held-out R2={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 across {len(SEEDS)} splits: mean={scores.mean():.4f}, "
      f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out ANN): R2=0.58")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.58:+.4f}")
