#!/usr/bin/env python3
"""Dravix V2 -- Char Residue re-attempt after the 2026-09-29 STA unit fix (stream sta_fix).

WHY: exp005 concluded "Char Residue ruled out, only 4 materials char". The repro
audit showed that was a unit artifact: char_yield_sta.csv held mg lost, not %,
for 93% of rows. With the corrected STA residues (ingest_fsri_sta_mcc.py) this
script re-asks the question twice:

Part A (headline): target = CONE residue (ingest_fsri_cone_residue.py), the
  closest FSRI analog to Pomazi et al. 2026's CR (residue after a mass-loss-cone
  test in air; real held-out R2 0.82, Table 9). Features are bench-scale only:
  corrected N2 STA residue at 3/10/30 K/min (material means), MCC char yield
  (material mean), density (material_props_fsri.csv), heat flux. Same-rows ablation
  against flux + density alone. Comparability with Pomazi is loose: they predict
  MLC residue of fibre-reinforced thermoset composites from formulation inputs;
  this is a standard cone on ~70 diverse products, and the residue includes inert
  substrates and fillers (gypsum, cement, glass fibre, mineral wool).
Part B: target = corrected STA residue itself (exp005 successor). exp005's own
  config first (its exact seed loop), then eval_protocol (a)+(b).

Model and settings fixed a priori (the house RF config of exp001/exp005, NaN
handled natively by sklearn>=1.4 RF); nothing tuned. Rows exclude the non-standard
'HF15-noSpark' PMMA tests (time-capped before burnout) and QA-gated replicates.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from eval_paired import print_paired
from eval_protocol import report, repeated_group_kfold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RF = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)
SOTA_CR = 0.82
# identical STA files filed under two material names (md5-checked): one group each, so they never straddle folds
ALIAS = {"Cardboard": "Corrugated_Cardboard", "Exterior_Plywood": "Plywood"}

cone = pd.read_csv(f"{DATA_DIR}/char_residue_cone_fsri.csv")
sta = pd.read_csv(f"{DATA_DIR}/char_yield_sta.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")

sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct", aggfunc="mean")
            .rename(columns=lambda r: f"sta_res_{int(r)}K").reset_index())
mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                      mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean")).reset_index()
feat = (sta_wide.merge(mcc_agg, on="material", how="outer")
        .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left"))

# ---------------- Part A: cone residue ----------------
df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()].merge(feat, on="material", how="left")
STA_COLS = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
# row rule (fixed before modelling): material has a corrected STA residue at >=1 rate AND an MCC char yield
df = df[df[STA_COLS].notna().any(axis=1) & df.mcc_char_yield_pct.notna()].reset_index(drop=True)
y, groups = df.cone_residue_pct.values, df.material.values
print(f"Part A rows: {len(df)} cone replicates / {df.material.nunique()} materials "
      f"(of {cone[cone.cone_residue_pct.notna()].material.nunique()} materials with a cone residue); target mean {y.mean():.1f}%, sd {y.std():.1f}")
print(f"  missing: density {df.density_kgm3.isna().sum()} rows, " +
      ", ".join(f"{c} {df[c].isna().sum()}" for c in STA_COLS))

BASE = ["heat_flux_kWm2", "density_kgm3"]
SETS = {"A0 flux+density (baseline)": BASE,
        "A1 +MCC char yield": BASE + ["mcc_char_yield_pct"],
        "A2 +STA residue x3 rates": BASE + STA_COLS,
        "A3 +MCC +STA (full)": BASE + ["mcc_char_yield_pct"] + STA_COLS}
res = {}
for name, cols in SETS.items():
    print()
    res[name] = report(name, RF, df[cols].values, y, groups, sota=SOTA_CR)

print("\nPaired same-rows same-folds deltas vs A0 (repeated GroupKFold 5x10):")
for name in ["A1 +MCC char yield", "A2 +STA residue x3 rates", "A3 +MCC +STA (full)"]:
    print_paired(f"{name} vs A0", RF, df[SETS["A0 flux+density (baseline)"]].values, df[SETS[name]].values, y, groups)
print_paired("A3 vs A1 (what STA adds on top of MCC)", RF, df[SETS["A1 +MCC char yield"]].values,
             df[SETS["A3 +MCC +STA (full)"]].values, y, groups)

cats = df.categories.fillna("unknown").values
print(f"\nHarder split: GroupKFold by material category ({len(set(cats))} categories), 5-fold x10, same rows:")
for name in ["A0 flux+density (baseline)", "A3 +MCC +STA (full)"]:
    rkf, _ = repeated_group_kfold(RF, df[SETS[name]].values, y, cats)
    print(f"  {name}: pooled OOF R2 mean={rkf.mean():.4f} std={rkf.std():.4f}")

full = df.dropna(subset=SETS["A3 +MCC +STA (full)"]).reset_index(drop=True)
print(f"\nSensitivity, complete-case rows only (no NaN features): {len(full)} rows / {full.material.nunique()} materials; "
      f"target sd {full.cone_residue_pct.std(ddof=0):.1f} (all rows {y.std():.1f}); materials dropped: "
      f"{sorted(set(df.material) - set(full.material))}")
for name in ["A0 flux+density (baseline)", "A3 +MCC +STA (full)"]:
    report(name + " [complete-case]", RF, full[SETS[name]].values, full.cone_residue_pct.values, full.material.values,
           sota=SOTA_CR)

m = RandomForestRegressor(**RF.get_params()).fit(df[SETS["A3 +MCC +STA (full)"]].values, y)
print("\nA3 feature importances (full-data fit, interpretation only):",
      ", ".join(f"{c} {v:.3f}" for c, v in sorted(zip(SETS["A3 +MCC +STA (full)"], m.feature_importances_), key=lambda t: -t[1])))
print("Pearson r, cone residue vs material-mean STA 10K residue (rows with both):",
      round(df[["cone_residue_pct", "sta_res_10K"]].dropna().corr().iloc[0, 1], 3))

# ---------------- Part B: corrected STA residue as target (exp005 successor) ----------------
print("\n================ Part B: corrected STA N2 residue as target (exp005 config) ================")
b = sta.copy()
b["char_yield_pct_floored"] = b.final_residual_mass_pct.clip(lower=0)  # exp005's target definition
b = b.merge(mcc_agg, on="material", how="inner").dropna(subset=["heating_rate_K_min", "mcc_peak_hrr_Wg", "char_yield_pct_floored"])
Xb, yb, gb = b[["heating_rate_K_min", "mcc_peak_hrr_Wg"]].values, b.char_yield_pct_floored.values, b.material.values
print(f"exp005 row set with corrected target: {len(b)} rows / {b.material.nunique()} materials "
      f"(exp005 had 574/96 before the fix: QA-gated rows are now dropped); target median {np.median(yb):.1f}%, "
      f"rows at 0: {(yb == 0).sum()}")
leg = []
for seed in range(10):  # exp005's exact loop (RF random_state=seed)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(Xb, yb, gb))
    mdl = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, min_samples_leaf=2).fit(Xb[tr], yb[tr])
    leg.append(r2_score(yb[te], mdl.predict(Xb[te])))
leg = np.array(leg)
print(f"exp005 exact loop (rs=seed): mean R2={leg.mean():.4f} std={leg.std():.4f} min={leg.min():.4f} max={leg.max():.4f}")
report("B1 exp005 features (heating rate, MCC peak HRR)", RF, Xb, yb, gb, sota=SOTA_CR)
report("B1 same, groups with duplicate materials merged", RF, Xb, yb, np.array([ALIAS.get(g, g) for g in gb]), sota=SOTA_CR)
b2 = b.dropna(subset=["mcc_char_yield_pct"])
report("B2 +MCC char yield (near-tautological: MCC pyrolysis is also anaerobic)", RF,
       b2[["heating_rate_K_min", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]].values, b2.char_yield_pct_floored.values,
       np.array([ALIAS.get(g, g) for g in b2.material.values]), sota=SOTA_CR)
