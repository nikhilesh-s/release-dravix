#!/usr/bin/env python3
"""Dravix V2 -- first FIGRA (fire growth rate) model.

Target: FIGRA = pHRR / time to pHRR from test start [kW/m2/s] (ingest_fsri_figra.py),
averaged over a material x heat flux's spark-ignited replicates that recorded ignition.
Competitor: phan_morgan_2025 FIGRA R2 = 0.9724 (arXiv:2504.00223 Table 5): 15 real polymers,
one condition (50 kW/m2), RF trained only on SDV-synthetic polymers. Their set and protocol
are not ours (see reports/sprint_0929/figra.md); this script just reports our numbers.

Everything below was fixed BEFORE the first run (no tuning, no feature search):
  model (primary)   HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                    max_leaf_nodes=15, min_samples_leaf=5); handles NaN features natively
  model (reference) RandomForestRegressor(300, max_depth=6, min_samples_leaf=2), the exp002 settings
  target            trained on log1p(FIGRA) (right-skewed, 0.2-70), scored in raw kW/m2/s
  features  F0  heat_flux_kWm2
            F1  F0 + material-mean MCC peak HRR and char yield (mcc_hrr.csv, as exp002)
            F2  F1 + density, cone thickness, cone mass per area (material_props_fsri.csv)
                   + material-mean heat release capacity, total heat release, T_peak (mcc_extras_fsri.csv)
  row sets  A   all heat fluxes, materials with MCC data (identical rows for F0/F1/F2)
            B   50 kW/m2 only, same materials as A (the competitor's condition)
            C   all heat fluxes, every material with FIGRA (MCC/props NaN where missing)
  pick      HGB + F2 on A (all fluxes) and B (50 kW/m2). Headline = repeated GroupKFold + CI.
Groups = material. Harder split: groups = material category (material_props_fsri.csv).

Run all:  OMP_NUM_THREADS=1 python train_figra_v1.py        (~10 min single process)
Or split: OMP_NUM_THREADS=1 python train_figra_v1.py A  (and B, C in parallel)
OMP_NUM_THREADS=1 matters on a busy machine: HGB's OpenMP threads thrash otherwise.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from eval_protocol import repeated_group_kfold, report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA = 0.9724

reps = pd.read_csv(f"{DATA_DIR}/figra_fsri.csv")
reps = reps[reps.spark_ignitor & reps.ignition_recorded]
df = (reps.groupby(["material", "heat_flux_kWm2"])
      .agg(figra_kWm2s=("figra_kWm2s", "mean"), n_reps=("figra_kWm2s", "size")).reset_index())

mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv").groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean"))
extras = pd.read_csv(f"{DATA_DIR}/mcc_extras_fsri.csv").groupby("material").agg(
    mcc_hrc_JgK=("hrc_JgK", "mean"), mcc_total_hr_kJg=("total_hr_kJg", "mean"), mcc_t_peak_C=("t_peak_C", "mean"))
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv").set_index("material")[
    ["categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
df = df.join(mcc, on="material").join(extras, on="material").join(props, on="material")
df["categories"] = df["categories"].fillna("unknown")

F0 = ["heat_flux_kWm2"]
F1 = F0 + ["mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
F2 = F1 + ["density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2",
           "mcc_hrc_JgK", "mcc_total_hr_kJg", "mcc_t_peak_C"]
MODELS = {
    "HGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=5, random_state=0),
    "RF": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0),
}

has_mcc = df.mcc_peak_hrr_Wg.notna()
ROW_SETS = {
    "A_allflux_mcc": df[has_mcc],
    "B_50only_mcc": df[has_mcc & (df.heat_flux_kWm2 == 50)],
    "C_allflux_all": df,
}
ONLY = sys.argv[1:] or ["A", "B", "C"]
print(f"FIGRA rows: {len(df)} material x flux / {df.material.nunique()} materials; "
      f"with MCC: {has_mcc.sum()} / {df[has_mcc].material.nunique()}")
print(f"F2 coverage on A: " + ", ".join(f"{c}={ROW_SETS['A_allflux_mcc'][c].notna().mean():.0%}" for c in F2[3:]))

results = []
for rs_name, d in ROW_SETS.items():
    if rs_name[0] not in ONLY:
        continue
    y, groups = d.figra_kWm2s.to_numpy(), d.material.to_numpy()
    for fs_name, feats in [("F0", F0), ("F1", F1), ("F2", F2)]:
        for m_name, model in MODELS.items():
            out = report(f"{rs_name} {fs_name} {m_name}", model, d[feats].to_numpy(float), y, groups,
                         log_target=True, sota=SOTA)
            results.append({"rows": rs_name, "features": fs_name, "model": m_name, **out})

# POST-HOC (added after seeing the F2 result; explanatory only, does not change the pick):
# which half of F2 carries the gain, specimen props or MCC extras?
PROPS = ["density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]
for rs_name in ["A_allflux_mcc", "B_50only_mcc"]:
    if rs_name[0] not in ONLY:
        continue
    d = ROW_SETS[rs_name]
    for fs_name, feats in [("F1+props", F1 + PROPS), ("F1+mcc_extras", [f for f in F2 if f not in PROPS])]:
        out = report(f"{rs_name} {fs_name} HGB [post-hoc]", MODELS["HGB"], d[feats].to_numpy(float),
                     d.figra_kWm2s.to_numpy(), d.material.to_numpy(), log_target=True, sota=SOTA)
        results.append({"rows": rs_name, "features": fs_name, "model": "HGB (post-hoc)", **out})

# Harder split: whole material categories held out (primary model, F2)
for rs_name in ["A_allflux_mcc", "B_50only_mcc"]:
    if rs_name[0] not in ONLY:
        continue
    d = ROW_SETS[rs_name]
    out = report(f"{rs_name} F2 HGB [grouped by CATEGORY, {d.categories.nunique()} categories]",
                 MODELS["HGB"], d[F2].to_numpy(float), d.figra_kWm2s.to_numpy(), d.categories.to_numpy(),
                 log_target=True, sota=SOTA)
    results.append({"rows": rs_name, "features": "F2", "model": "HGB/category-split", **out})

# Closest to the competitor's sample space: score the B (50 kW/m2) OOF predictions on polymers only
if "B" in ONLY:
    d = ROW_SETS["B_50only_mcc"]
    rkf, oof = repeated_group_kfold(MODELS["HGB"], d[F2].to_numpy(float), d.figra_kWm2s.to_numpy(),
                                    d.material.to_numpy(), log_target=True)
    poly = (d.categories == "general polymer").to_numpy()
    poly_r2 = [r2_score(d.figra_kWm2s.to_numpy()[poly], p[poly]) for p in oof]
    print(f"\nB F2 HGB, OOF scored on the {poly.sum()} 'general polymer' materials only "
          f"({', '.join(d.material[poly])}): pooled R2 mean={np.mean(poly_r2):.4f} std={np.std(poly_r2):.4f}")
    print(d.assign(oof_mean=oof.mean(axis=0))[poly][["material", "figra_kWm2s", "oof_mean"]]
          .round(2).to_string(index=False))

print("\nSummary (R2 in raw kW/m2/s; GKF = repeated GroupKFold 5x10 pooled OOF, headline):")
print(pd.DataFrame(results)[["rows", "features", "model", "n_rows", "n_materials", "gss_mean", "gss_std",
                             "rkf_mean", "rkf_std", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
