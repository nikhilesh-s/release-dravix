#!/usr/bin/env python3
"""Dravix V2 -- exp003. First real model on Time to Ignition.

Target: time_to_ignition_s (tti_fsri.csv, 717 real replicate-level rows --
by far the biggest real table we have). Joined with material-level MCC
features (peak specific HRR, char yield) as material-identity proxies,
same idea as the pHRR runs.

Competitor bar: Pomazi et al. 2026 (BME), Time to Ignition, REAL HELD-OUT
R2 = 0.58 (their best, ANN) -- see benchmarks/sota_board.csv. Their
full-dataset numbers (RFR/XGBoost tied at 0.95) are NOT the bar, same
methodology note as the pHRR runs.

This is a harder bar than pHRR's 0.30, but we have 2.5x the data (546 vs
175/158 rows) going in, which is the actual trade being tested here.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

tti = pd.read_csv(f"{DATA_DIR}/tti_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()

df = tti.merge(mcc_agg, on="material", how="inner")
df["spark_ignitor"] = df["spark_ignitor"].astype(int)

FEATURES = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor",
            "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials (TTI x MCC inner join)")

X = df[FEATURES].values
y = df[TARGET].values
groups = df["material"].values

SEEDS = list(range(10))
scores = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, min_samples_leaf=2)
    model.fit(X[train_idx], y[train_idx])
    r2 = r2_score(y[test_idx], model.predict(X[test_idx]))
    scores.append(r2)
    print(f"  seed={seed}: n_test={len(test_idx)}, held-out R2={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 across {len(SEEDS)} material-grouped splits: mean={scores.mean():.4f}, "
      f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out ANN): R2=0.58")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.58:+.4f}")
