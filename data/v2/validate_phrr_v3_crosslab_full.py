#!/usr/bin/env python3
"""Dravix V2 -- exp054. Cross-lab validation of the promoted pHRR model
(phrr_rf_v3.joblib, exp043) against the FULL MaCFP PMMA cross-lab file.

train_phrr_v3_robust.py's own crosslab() check (used to validate v3 at
promotion time) reads phrr_macfp_pmma.csv -- 67 tests / 10 institutions.
phrr_macfp_pmma_all.csv (added in the 2026-09-29 sprint) has 73 tests / 12
institutions: the same 10 plus UClan (2 tests, a real lab never checked
against) and HKPoly (4 tests). The sprint handoff flagged HKPoly's export
as "about 2x the heat every other lab reports" and recommended keeping it
out of cross-lab checks pending confirmation with MaCFP -- that exclusion
is applied here, not assumed silently (both numbers are printed).

Uses FSRI's own Black_PMMA material-level features (matching v3's feature
set) with MaCFP's actual specimen geometry (6mm nominal cast PMMA, per
matl-db/PMMA/README.md) substituted for the geometry features, the same
substitution train_phrr_v3_robust.py's crosslab() uses for its "+ MaCFP
geometry" row.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DATA_DIR, "..", "..", "models", "v2")

bundle = joblib.load(os.path.join(MODEL_DIR, "phrr_rf_v3.joblib"))
model = bundle["model"]
FEATURES = bundle["features"]
print(f"Loaded phrr_rf_v3.joblib, features: {FEATURES}")

phrr = pd.read_csv(f"{DATA_DIR}/phrr_fsri.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
mcc_extras = pd.read_csv(f"{DATA_DIR}/mcc_extras_fsri.csv")
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")
k_df = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
cp_df = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")

MAT = "Black_PMMA"

mcc_row = mcc[mcc["material"] == MAT]
mcc_peak_hrr_Wg = mcc_row["peak_specific_hrr_Wg"].mean()
mcc_char_yield_pct = mcc_row["char_yield_pct"].mean()

extras_row = mcc_extras[mcc_extras["material"] == MAT]
mcc_total_hr_kJg = extras_row["total_hr_bc_kJg"].mean()
mcc_t_peak_C = extras_row["t_peak_C"].mean()
mcc_t_onset_C = extras_row["t_onset_bc_C"].mean()

props_row = props[props["material"] == MAT].iloc[0]
density_kgm3 = props_row["density_kgm3"]

k_row = k_df[k_df["material"] == MAT]
cp_row = cp_df[cp_df["material"] == MAT]
k_val = k_row["Average_Cond"].mean()
cp_val = cp_row["Spc_Heat"].mean()
log_kpc = np.log(k_val * cp_val) if pd.notna(k_val) and pd.notna(cp_val) else np.nan

# MaCFP specimens: nominally 6 mm black cast PMMA (raw/matl-db/PMMA/README.md).
# Substitute MaCFP's own geometry for FSRI's, same override train_phrr_v3_robust.py's
# crosslab() uses -- FSRI's Black_PMMA cone specimens are ~8.4mm, not what MaCFP tested.
macfp_geom = {
    "cone_thickness_mm": 6.0,
    "cone_mass_per_area_kgm2": 6.0e-3 * density_kgm3,
    "density_kgm3": density_kgm3,
}

material_features = {
    "mcc_peak_hrr_Wg": mcc_peak_hrr_Wg,
    "mcc_char_yield_pct": mcc_char_yield_pct,
    "mcc_total_hr_kJg": mcc_total_hr_kJg,
    "mcc_t_peak_C": mcc_t_peak_C,
    "mcc_t_onset_C": mcc_t_onset_C,
    "log_kpc": log_kpc,
    **macfp_geom,
}
print(f"FSRI {MAT} material-level features (MaCFP geometry substituted): {material_features}")
assert not any(pd.isna(v) for v in material_features.values()), "missing a material-level feature"

macfp = pd.read_csv(f"{DATA_DIR}/phrr_macfp_pmma_all.csv")
for f, v in material_features.items():
    macfp[f] = v

X = macfp[FEATURES].to_numpy(float)
y_true = macfp["peak_hrr_kWm2"].to_numpy(float)
y_pred = model.predict(X)

print(f"\n=== All 12 institutions, {len(macfp)} tests (includes HKPoly, flagged mis-scaled) ===")
print(f"Cross-lab R2: {r2_score(y_true, y_pred):.4f}, MAE={np.mean(np.abs(y_true - y_pred)):.1f} kW/m2")

keep = macfp["institution"] != "HKPoly"
macfp_ex = macfp[keep]
X_ex = macfp_ex[FEATURES].to_numpy(float)
y_true_ex = macfp_ex["peak_hrr_kWm2"].to_numpy(float)
y_pred_ex = model.predict(X_ex)
print(f"\n=== 11 institutions, {len(macfp_ex)} tests (HKPoly excluded, per sprint's mis-scaling flag) ===")
print(f"Cross-lab R2: {r2_score(y_true_ex, y_pred_ex):.4f}, MAE={np.mean(np.abs(y_true_ex - y_pred_ex)):.1f} kW/m2")
print(f"(exp008/v1's original check on 10 labs, 67 tests, was R2=0.1986)")

print("\nPer institution (HKPoly-excluded set):")
for inst in sorted(macfp_ex["institution"].unique()):
    mask = macfp_ex["institution"] == inst
    n = mask.sum()
    print(f"  {inst:12s} (n={n}): true mean={y_true_ex[mask].mean():.1f} kW/m2, "
          f"pred={y_pred_ex[mask][0]:.1f} kW/m2, mean abs err={np.mean(np.abs(y_true_ex[mask]-y_pred_ex[mask])):.1f}")

print("\nHKPoly alone (for reference, not included in the headline number above):")
hk = macfp[macfp["institution"] == "HKPoly"]
hk_pred = model.predict(hk[FEATURES].to_numpy(float))
print(f"  n={len(hk)}: true mean={hk['peak_hrr_kWm2'].mean():.1f} kW/m2, pred={hk_pred[0]:.1f} kW/m2 "
      f"(true/pred ratio {hk['peak_hrr_kWm2'].mean()/hk_pred[0]:.2f} -- "
      f"{'consistent with the ~2x mis-scaling flag' if hk['peak_hrr_kWm2'].mean()/hk_pred[0] > 1.5 else 'NOT clearly 2x off here'})")
