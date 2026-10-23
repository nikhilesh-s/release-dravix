#!/usr/bin/env python3
"""Dravix V2 -- exp005. First real model on Char Residue.

Target: final_residual_mass_pct from char_yield_sta.csv, FLOORED AT 0 (see
reports/V2_DATA_ACQUISITION_LOG.md 2026-09-28 note: ~94% of raw values are
slightly negative, a systematic ~3% baseline artifact in the instrument's
blank-subtracted curve, not a real negative char yield -- real char-forming
materials like MDF/OSB still come through correctly at 20-26%, clearly
separated from the artifact band).

Competitor bar: Pomazi et al. 2026 (BME), Char Residue, REAL HELD-OUT R2 =
0.82 (their best, RFR) -- the toughest real bar of the three Tier-2 targets
attempted so far (pHRR 0.30 beaten in exp002, TTI 0.58 missed in exp003/4).

Only 36 materials have both STA and MCC coverage -- the smallest dataset
tried yet. Expect high variance and say so, don't round it off.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

sta = pd.read_csv(f"{DATA_DIR}/char_yield_sta.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")

sta["char_yield_pct_floored"] = sta["final_residual_mass_pct"].clip(lower=0)

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
).reset_index()

df = sta.merge(mcc_agg, on="material", how="inner")

FEATURES = ["heating_rate_K_min", "mcc_peak_hrr_Wg"]
TARGET = "char_yield_pct_floored"
df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials (STA x MCC inner join)")

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
    print(f"  seed={seed}: n_test={len(test_idx)}, held-out R2={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 across {len(SEEDS)} material-grouped splits: mean={scores.mean():.4f}, "
      f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (pomazi_2026, real held-out RFR): R2=0.82")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.82:+.4f}")
