#!/usr/bin/env python3
"""Dravix V2 -- exp004. TTI, take 2: log-transform the target before training.

exp003 (train_tti_v1.py) trained directly on time_to_ignition_s and got
mean R2=-0.26 with one seed at -3.07 -- TTI spans 3 orders of magnitude
(0 to 1,102s, median 17.5s) and a few slow-igniting materials (PET, carpet,
upholstery) dominate squared-error whenever they land in a test fold.

Fix: train on log1p(TTI), predict, then expm1 back to seconds before
scoring -- R2 is still computed in the ORIGINAL seconds scale (fair
comparison to Pomazi et al.'s R2=0.58, which is on their raw TTI values),
only the training target changed.
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
    preds_log = model.predict(X[test_idx])
    preds_raw = np.expm1(preds_log)
    r2 = r2_score(y_raw[test_idx], preds_raw)   # scored back in seconds, same scale as SOTA
    scores.append(r2)
    print(f"  seed={seed}: n_test={len(test_idx)}, held-out R2 (seconds)={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 (seconds scale) across {len(SEEDS)} material-grouped splits: "
      f"mean={scores.mean():.4f}, std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out ANN): R2=0.58")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.58:+.4f}")
