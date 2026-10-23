#!/usr/bin/env python3
"""Dravix V2 -- exp013. pHRR, take 3: add real thermal conductivity +
specific heat (same properties that helped TTI in exp009) on top of
exp006's promoted feature set, to see if it improves on R2=0.5744 (CV
mean 0.346) and -- more importantly -- whether it narrows the cross-lab
generalization gap found in exp008 (R2=0.20 against MaCFP's independent
PMMA measurements).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = "~/Dravix/dravix-v2/data/v2"

phrr = pd.read_csv(f"{DATA_DIR}/phrr_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
k_df = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
cp_df = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()
k_agg = k_df.groupby("material")["Average_Cond"].mean().rename("k_W_mK").reset_index()
cp_agg = cp_df.groupby("material")["Spc_Heat"].mean().rename("rhocp_J_m3K").reset_index()

df = (phrr
      .merge(mcc_agg, on="material", how="inner")
      .merge(k_agg, on="material", how="inner")
      .merge(cp_agg, on="material", how="inner"))
df["thermal_inertia"] = df["k_W_mK"] * df["rhocp_J_m3K"]

FEATURES = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct",
            "k_W_mK", "rhocp_J_m3K", "thermal_inertia"]
TARGET = "peak_hrrpua_kWm2"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials")

X = df[FEATURES].values
y = df[TARGET].values
groups = df["material"].values

SEEDS = list(range(10))
scores = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, min_samples_leaf=2)
    model.fit(X[train_idx], y[train_idx])
    r2 = r2_score(y[test_idx], model.predict(X[test_idx]))
    scores.append(r2)
    print(f"  seed={seed}: held-out R2={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 across {len(SEEDS)} splits: mean={scores.mean():.4f}, std={scores.std():.4f}, "
      f"min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out RFR): R2=0.30")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.30:+.4f}")
print(f"For comparison, exp002 (MCC only, no thermal props): CV mean was 0.346")

# locked promotion check, same seed=42 convention as exp006
HELD_OUT_SEED = 42
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=HELD_OUT_SEED)
train_idx, test_idx = next(gss.split(X, y, groups))
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0, min_samples_leaf=2)
model.fit(X[train_idx], y[train_idx])
r2_locked = r2_score(y[test_idx], model.predict(X[test_idx]))
print()
print(f"Locked promotion check (seed=42, {len(set(groups[test_idx]))} materials held out): R2={r2_locked:.4f}")
print(f"For comparison, exp006 (MCC only) locked check was 0.5744")

full_model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0, min_samples_leaf=2)
full_model.fit(X, y)
print()
print("Feature importances (full-data fit):")
for feat, imp in sorted(zip(FEATURES, full_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")
