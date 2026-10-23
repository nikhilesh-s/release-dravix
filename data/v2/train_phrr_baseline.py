#!/usr/bin/env python3
"""Dravix V2 — first real baseline model, Tier 2 / Peak HRR.

Target: peak_hrrpua_kWm2 (phrr_fsri.csv), joined with material-level features
derived from the OTHER real FSRI apparatus data already ingested (MCC peak
specific HRR + char yield, STA char yield). All real, all traceable to
data/v2/{phrr_fsri,mcc_hrr,char_yield_sta}.csv.

Competitor bar: Pomazi et al. 2026 (BME), Peak HRR, REAL HELD-OUT R2 = 0.30
(their best model, RFR) -- see benchmarks/sota_board.csv, pomazi_2026 row.
That is the number this script's held-out R2 is measured against. Their
FULL-DATASET-fit R2=0.96 is NOT the bar (see V2_TARGET_DEFINITION.md Tier 2
methodology note) -- do not compare against it.

Split: GroupShuffleSplit BY MATERIAL, so no material's rows appear in both
train and test (a material tested at 3 heat fluxes must not leak). Repeated
over multiple seeds because the material count is small (high variance is
expected and reported honestly, not hidden behind a single lucky split).
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
sta = pd.read_csv(f"{DATA_DIR}/char_yield_sta.csv")

mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()

sta_agg = sta.groupby("material").agg(
    sta_char_yield_pct=("final_residual_mass_pct", "mean"),
).reset_index()

df = (phrr
      .merge(mcc_agg, on="material", how="inner")
      .merge(sta_agg, on="material", how="inner"))

FEATURES = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "sta_char_yield_pct"]
TARGET = "peak_hrrpua_kWm2"

df = df.dropna(subset=FEATURES + [TARGET])
n_materials = df["material"].nunique()
print(f"Training rows: {len(df)} across {n_materials} materials "
      f"(inner join of phrr x mcc x sta -- every row has real values for all 4 features)")

X = df[FEATURES].values
y = df[TARGET].values
groups = df["material"].values

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    train_materials = set(groups[train_idx])
    test_materials = set(groups[test_idx])
    assert train_materials.isdisjoint(test_materials), "material leaked across split"

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, min_samples_leaf=2)
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    r2 = r2_score(y[test_idx], preds)
    scores.append(r2)
    print(f"  seed={seed}: test_materials={len(test_materials)}, held-out R2={r2:.4f}")

scores = np.array(scores)
print()
print(f"Held-out R2 across {len(SEEDS)} material-grouped splits: "
      f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
      f"min={scores.min():.4f}, max={scores.max():.4f}")
print(f"SOTA (Pomazi et al. 2026, real held-out, RFR): R2=0.30")
print(f"Delta (mean Dravix - SOTA): {scores.mean() - 0.30:+.4f}")
