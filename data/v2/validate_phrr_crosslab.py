#!/usr/bin/env python3
"""Dravix V2 -- cross-lab out-of-distribution validation of the promoted
pHRR model (models/v2/phrr_rf_v1.joblib, exp006).

Every prior evaluation of this model was a material-grouped split WITHIN
FSRI's own single lab/instrument. This script tests it against 67 real
PMMA cone-calorimeter tests from 10 INDEPENDENT INSTITUTIONS (MaCFP
matl-db: Aalto, DBI_Lund, Edinburgh, GIDAZE+, LCPP, NIST, TIFP, ULSTER,
UDRI, UQ) -- different labs, different cone rigs, different operators,
same material. This is a genuinely harder generalization test than
another same-source split.

Features (heat_flux, mcc_peak_hrr_Wg, mcc_char_yield_pct) use FSRI's own
MCC measurement of PMMA as the material-level descriptor -- legitimate,
since those describe an intrinsic material property, not the target lab's
own equipment. The prediction is then checked against each institution's
INDEPENDENTLY measured pHRR.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

bundle = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "v2", "phrr_rf_v1.joblib"))
model = bundle["model"]
FEATURES = bundle["features"]

macfp = pd.read_csv(f"{DATA_DIR}/phrr_macfp_pmma.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
pmma_mcc = mcc[mcc["material"] == "PMMA"]
mcc_peak_hrr_Wg = pmma_mcc["peak_specific_hrr_Wg"].mean()
mcc_char_yield_pct = pmma_mcc["char_yield_pct"].mean()
print(f"FSRI's own PMMA MCC features used for all rows: "
      f"mcc_peak_hrr_Wg={mcc_peak_hrr_Wg:.2f}, mcc_char_yield_pct={mcc_char_yield_pct:.2f}")

macfp["mcc_peak_hrr_Wg"] = mcc_peak_hrr_Wg
macfp["mcc_char_yield_pct"] = mcc_char_yield_pct

X = macfp[FEATURES].values
y_true = macfp["peak_hrr_kWm2"].values
y_pred = model.predict(X)

overall_r2 = r2_score(y_true, y_pred)
print(f"\nOverall cross-lab R2 (67 tests, 10 institutions, model never trained on any of them): {overall_r2:.4f}")
print(f"Mean abs error: {np.mean(np.abs(y_true - y_pred)):.1f} kW/m2 "
      f"(mean true pHRR: {y_true.mean():.1f} kW/m2, {np.mean(np.abs(y_true-y_pred))/y_true.mean()*100:.1f}% of mean)")

print("\nPer heat-flux breakdown:")
for hf in sorted(macfp["heat_flux_kWm2"].unique()):
    mask = macfp["heat_flux_kWm2"] == hf
    if mask.sum() < 2:
        continue
    r2_hf = r2_score(y_true[mask], y_pred[mask])
    print(f"  {hf} kW/m2 (n={mask.sum()}): true mean={y_true[mask].mean():.1f}, "
          f"pred={y_pred[mask][0]:.1f} (single value, features constant per material), R2={r2_hf:.4f}")

print("\nPer institution:")
for inst in sorted(macfp["institution"].unique()):
    mask = macfp["institution"] == inst
    n = mask.sum()
    mean_err = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    print(f"  {inst} (n={n}): true mean={y_true[mask].mean():.1f} kW/m2, "
          f"pred={y_pred[mask][0]:.1f} kW/m2, mean abs err={mean_err:.1f}")
