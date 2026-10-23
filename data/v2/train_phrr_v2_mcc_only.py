#!/usr/bin/env python3
"""Dravix V2 -- exp002. Same target/competitor as exp001 (train_phrr_baseline.py):
Peak HRR vs pomazi_2026 real held-out RFR SOTA R2=0.30.

Change from exp001: drop the STA char-yield feature (sta_char_yield_pct). That
feature only exists for materials with BOTH cone-calorimeter AND STA testing,
which shrank exp001 to 158 rows/55 materials. Dropping it and keeping only
heat_flux_kWm2 + the two MCC-derived features recovers 175 rows/60 materials
(every material with cone + MCC coverage, whether or not it also has STA).
Testing whether more data beats a feature dropped mattered more than the
feature itself, or vice versa.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

phrr = pd.read_csv(f"{DATA_DIR}/phrr_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()

df = phrr.merge(mcc_agg, on="material", how="inner")

FEATURES = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
TARGET = "peak_hrrpua_kWm2"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials (cone x MCC inner join, no STA requirement)")

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
print(f"Held-out R2 across {len(SEEDS)} material-grouped splits: mean={scores.mean():.4f}, "
      f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out RFR): R2=0.30")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.30:+.4f}")
