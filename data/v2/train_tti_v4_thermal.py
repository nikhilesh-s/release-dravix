#!/usr/bin/env python3
"""Dravix V2 -- exp009. TTI, take 4: add real HFM thermal properties.

Classical ignition theory for a thermally thick solid gives
t_ig ~ k*rho*c * (T_ig - T_0)^2 / q''^2, so thermal inertia (k*rho*c) is the
physical driver the MCC proxies were standing in for. thermal_conductivity_fsri.csv
gives k (W/m-K); specific_heat_fsri.csv's Spc_Heat is volumetric (J/m^3-K --
EPS reads ~1.6e4, consistent with rho~20 kg/m^3), so k * Spc_Heat is thermal
inertia directly, no density lookup needed. Dry-condition readings only,
lowest Mean_Temp setpoint per material (closest to ambient).

Onset temp (exp007) dropped: its join cut the set to 16 materials and it
ranked last in importance there.

Also adds the physics-shaped feature kpc / q''^2. Scored as an ablation on the
SAME rows and SAME splits with and without the new features, so any gain is
attributable to them and not to the inner join dropping hard materials.

`--no-mcc` runs the robustness check logged with exp009: TTI x HFM only
(no MCC requirement, so more materials), base = heat flux, mass, ignitor.

Numbers in the leaderboard come from scikit-learn 1.7.1 (requirements.txt).
The first run of this script used 1.9.1 and gave different forests from the
same seeds (0.091 -> 0.163 instead of 0.103 -> 0.194), so pin the version.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

tti = pd.read_csv(f"{DATA_DIR}/tti_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
k = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
c = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")


def ambient_dry(df, col):
    df = df[df["condition"] == "Dry"]
    lowest = df.groupby("material")["Mean_Temp"].transform("min")
    return df[df["Mean_Temp"] == lowest].groupby("material")[col].mean()


mcc_agg = mcc.groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
    mcc_char_yield_pct=("char_yield_pct", "mean"),
).reset_index()
thermal = pd.DataFrame({
    "k_WmK": ambient_dry(k, "Average_Cond"),
    "rhoc_Jm3K": ambient_dry(c, "Spc_Heat"),
}).dropna().reset_index().rename(columns={"index": "material"})
thermal["log_kpc"] = np.log(thermal["k_WmK"] * thermal["rhoc_Jm3K"])

NO_MCC = "--no-mcc" in sys.argv
df = (tti if NO_MCC else tti.merge(mcc_agg, on="material")).merge(thermal, on="material")
df["spark_ignitor"] = df["spark_ignitor"].astype(int)
df["log_kpc_over_q2"] = df["log_kpc"] - 2 * np.log(df["heat_flux_kWm2"])

BASE = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor"]
if not NO_MCC:
    BASE += ["mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
NEW = ["k_WmK", "rhoc_Jm3K", "log_kpc", "log_kpc_over_q2"]
TARGET = "time_to_ignition_s"
df = df.dropna(subset=BASE + NEW + [TARGET])
print(f"Rows: {len(df)} across {df['material'].nunique()} materials "
      f"({'TTI x HFM' if NO_MCC else 'TTI x MCC x HFM'} inner join)")

y_raw = df[TARGET].values
y_log = np.log1p(y_raw)
groups = df["material"].values
SEEDS = list(range(10))


def evaluate(features):
    X = df[features].values
    scores = []
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, te = next(gss.split(X, y_raw, groups))
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        m = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, min_samples_leaf=2)
        m.fit(X[tr], y_log[tr])
        scores.append(r2_score(y_raw[te], np.expm1(m.predict(X[te]))))
    return np.array(scores)


results = {"base": evaluate(BASE),
           "base + thermal": evaluate(BASE + NEW)}
for name, s in results.items():
    print(f"{name:24s} per-seed: {np.round(s, 3)}")
    print(f"{'':24s} mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f} "
          f"delta vs SOTA 0.58: {s.mean() - 0.58:+.4f}")
d = results["base + thermal"] - results["base"]
print(f"Paired gain from thermal features: mean={d.mean():+.4f}, seeds improved={int((d > 0).sum())}/10")

from eval_protocol import report  # noqa: E402

rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
print()
for name, feats in [("base", BASE), ("base + thermal", BASE + NEW)]:
    report(name, rf, df[feats].values, y_raw, groups, log_target=True, sota=0.58)

full = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, min_samples_leaf=2)
full.fit(df[BASE + NEW].values, y_log)
print("\nFeature importances (full-data fit, interpretation only):")
for f, imp in sorted(zip(BASE + NEW, full.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")
