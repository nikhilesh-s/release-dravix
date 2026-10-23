#!/usr/bin/env python3
"""Dravix V2 -- exp007. TTI, take 3: add STA decomposition onset temperature.

exp003/exp004 (train_tti_v1/v2) used only heat_flux, specimen_mass,
spark_ignitor, and bulk MCC combustion-energy proxies -- best result
(exp004, log target) was R2=0.10 vs SOTA 0.58. None of those features
describe WHEN a material starts to decompose, which is the actual physical
driver of time-to-ignition. onset_temp_sta.csv (ingest_sta_onset_temp.py)
adds that directly: temperature at 5% mass loss from each material's own
STA curve.

Keeps exp004's log1p(target) fix (still scored back in raw seconds).
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
onset = pd.read_csv(f"{DATA_DIR}/onset_temp_sta.csv")

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()
onset_agg = onset.groupby("material").agg(
    onset_temp_C=("onset_temp_5pct_C", "mean"),
).reset_index()

df = tti.merge(mcc_agg, on="material", how="inner").merge(onset_agg, on="material", how="inner")
df["spark_ignitor"] = df["spark_ignitor"].astype(int)

FEATURES = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor",
            "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "onset_temp_C"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials (TTI x MCC x STA-onset inner join)")

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
    print(f"  seed={seed}: n_test={len(test_idx)}, held-out R2 (seconds)={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 (seconds) across {len(SEEDS)} splits: mean={scores.mean():.4f}, "
      f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out ANN): R2=0.58")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.58:+.4f}")

# feature importance on a full-data fit, for interpretation only (not used in scoring)
full_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
full_model.fit(X, y_log)
print()
print("Feature importances (full-data fit, for interpretation only):")
for feat, imp in sorted(zip(FEATURES, full_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")
