#!/usr/bin/env python3
"""Dravix V2 -- exp006. Promotion check for exp002 (pHRR).

exp002 reported mean R2=0.346 across 10 material-grouped CV splits (seeds
0-9) -- a real result, but a CV mean isn't the same claim as "beats SOTA
on a held-out test." All 60 materials rotated through some seed's test
fold, so none of those 10 splits is a clean "never touched" set on its own.

This script commits to ONE new split (seed=42, never used in exp001/2/3/4/5),
trains once, evaluates once, and reports that single number as the actual
promotion criterion. No re-running with different seeds after seeing the
result -- that would just be seed-shopping. If the number here holds up,
exp002's feature set/model gets promoted; if not, exp002 stays "beats SOTA
on average across seeds" and nothing stronger.
"""
import os
import sys
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

X = df[FEATURES].values
y = df[TARGET].values
groups = df["material"].values

HELD_OUT_SEED = 42  # committed before running, not chosen after seeing the result
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=HELD_OUT_SEED)
train_idx, test_idx = next(gss.split(X, y, groups))
train_materials = set(groups[train_idx])
test_materials = set(groups[test_idx])
assert train_materials.isdisjoint(test_materials)

model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0, min_samples_leaf=2)
model.fit(X[train_idx], y[train_idx])
preds = model.predict(X[test_idx])
r2 = r2_score(y[test_idx], preds)

print(f"Locked held-out set (seed={HELD_OUT_SEED}): {len(test_idx)} rows, "
      f"{len(test_materials)} materials, never used to pick features/hyperparameters")
print(f"Held-out R2: {r2:.4f}")
print(f"SOTA (pomazi_2026, real held-out RFR): R2=0.30")
print(f"Delta: {r2 - 0.30:+.4f}")
print(f"Result: {'BEATS SOTA -- promote' if r2 > 0.30 else 'does not beat SOTA on this locked set'}")
print("NOTE (2026-09-29): that verdict rests on one split. Under repeated GroupKFold this recipe scores "
      "0.304, CI [-0.16, 0.67], a tie with SOTA; see exp014 and the exp006 erratum.")

# Added 2026-09-29: the step that wrote models/v2/phrr_rf_v1.joblib was never committed.
# The repro audit showed the saved model is exactly this refit on all 175 rows (max |dpred| = 0).
# Usage: promote_check_phrr.py --save <path>
if "--save" in sys.argv:
    import joblib
    out = sys.argv[sys.argv.index("--save") + 1]
    final = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0, min_samples_leaf=2).fit(X, y)
    joblib.dump(dict(model=final, features=FEATURES, target=TARGET,
                     trained_on=f"data/v2/phrr_fsri.csv + mcc_hrr.csv (inner join), all {len(df)} rows / "
                                f"{df['material'].nunique()} materials",
                     validation=f"exp006 locked held-out (seed={HELD_OUT_SEED}, {len(test_materials)} materials): "
                                f"R2={r2:.4f} vs SOTA 0.30. Superseded: repeated GroupKFold gives 0.304, CI [-0.16, 0.67] "
                                f"(exp014), a tie with SOTA",
                     sota_reference="pomazi_2026 Peak HRR real held-out RFR R2=0.30 (benchmarks/sota_board.csv)"), out)
    print(f"Saved refit on all rows to {out}")
