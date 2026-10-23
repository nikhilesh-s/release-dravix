#!/usr/bin/env python3
"""Dravix V2 -- exp012. TTI, take 6: drop the unnecessary density join.

thermal_inertia = k * rhocp works directly from k (thermal_conductivity_fsri.csv)
and rhocp (specific_heat_fsri.csv, already volumetric J/m3-K) -- density
was only needed as its OWN separate feature in exp009, not for computing
the inertia term. Requiring a THIRD real dataset (density_fsri.csv, only
117/183 materials) to be present shrank the join from 56 to 53 materials
for no benefit to the inertia calculation itself. Dropping density as its
own feature recovers those materials while keeping the same core physics
feature.
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

k_agg = k_df.groupby("material")["Average_Cond"].mean().rename("k_W_mK").reset_index()
cp_agg = cp_df.groupby("material")["Spc_Heat"].mean().rename("rhocp_J_m3K").reset_index()

df = tti.merge(k_agg, on="material", how="inner").merge(cp_agg, on="material", how="inner")
df["thermal_inertia"] = df["k_W_mK"] * df["rhocp_J_m3K"]
df["spark_ignitor"] = df["spark_ignitor"].astype(int)

FEATURES = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor",
            "k_W_mK", "rhocp_J_m3K", "thermal_inertia"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials (dropped density requirement)")

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

# locked promotion check, same convention (seed=42, committed before seeing result)
HELD_OUT_SEED = 42
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=HELD_OUT_SEED)
train_idx, test_idx = next(gss.split(X, y_raw, groups))
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
model.fit(X[train_idx], y_log[train_idx])
preds_raw = np.expm1(model.predict(X[test_idx]))
r2_locked = r2_score(y_raw[test_idx], preds_raw)
print()
print(f"Locked promotion check (seed=42, {len(set(groups[test_idx]))} materials held out): R2={r2_locked:.4f}")
print(f"Result: {'BEATS SOTA -- promote' if r2_locked > 0.58 else 'does not beat SOTA on this locked set'}")
