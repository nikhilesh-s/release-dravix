#!/usr/bin/env python3
"""Dravix V2 -- TTI, take 5: keep all 80 materials, add bench-only ignition physics.

Why: exp003/4/7/9 each added a feature through an inner join, which cut the TTI
set from 80 materials to 16-61. At 16-25 materials the split-to-split noise
(+-0.3-0.7 R2) was bigger than any feature effect being measured, so none of
those comparisons could say anything. HistGradientBoostingRegressor (and
sklearn>=1.4 RandomForest) route NaN natively, so here every feature is
LEFT-joined: all 717 rows / 80 materials stay in every candidate, and every
candidate is scored on the same rows with the same splits.

Physics (classical ignition theory; bench data only, never a cone outcome):
  thermally thick  t_ig ~ (pi/4) * k rho c * (T_ig - T0)^2 / q''^2
  thermally thin   t_ig ~ rho c * L * (T_ig - T0) / q''
- T_ig proxy: MCC 5%-of-heat onset temperature (mcc_extras_fsri.csv, median
  over replicates); T0 = 20 C.
- k rho c = k * Spc_Heat (Spc_Heat is volumetric, J/m3-K). HFM reading at the
  lowest Mean_Temp; Dry if the material has any Dry reading, else Wet.
- L: cone specimen thickness; rho: material.json density; mass/area: cone
  SPECIMEN MASS / SURF AREA (both pre-test, material_props_fsri.csv).

Candidates, declared before any was scored (all hyperparameters fixed a priori,
none tuned):
  C0 exp004 RF (300 trees, depth 8, leaf 2), exp004 features, log1p target
  C1 HGB, exp004 features, log1p target
  C2 HGB, + density, cone thickness, cone mass/area
  C3 HGB, + MCC onset T, peak T, heat release capacity, total HR, log k rho c
  C4 HGB, + physics composites log t_thick, log t_thin, log t_thin(mass/area)
  C5 Ridge(alpha=1) on log1p TTI: log q'', log dT, log k rho c, log L, log rho
  C6 Poisson GLM (log link, raw seconds, alpha=0.01), same 5 physics features
  C7 HGB poisson loss (log link, raw seconds), C4 features
  R0 reference only: HGB on heat flux alone
Pick rule (fixed in advance): highest material-grouped GKF mean R2 on the
primary set; within 0.02 of it, prefer the simpler model.

Data-quality note: 45 rows have TIME TO IGN = 0. 42 of their Scalar POST TEST
CMTs say "No ignition" / "did not ignite" (mineral wool, both fiberglass batts,
gypsum/PC/house wrap at 25 kW/m2, ...); the other 3, PMMA at 15 kW/m2 without
spark, say "MLR did not exceed 1.0 g/min" or are blank. They
are censored non-ignitions stored as 0 s, the wrong end of the scale. Both row
sets run on identical splits: "all 717" (exp003-009's row definition) and
"ignited only" (672 rows / 77 materials). Declared primary was all 717; after
scoring, the lead made "ignited only" the primary for data correctness. The
pick rule is unchanged. See reports/sprint_0929/tti.md.

STA onset (onset_temp_sta.csv) is deliberately not used: it is mostly the scan
start temperature. T_ig comes from MCC t_onset_C.

--onset-bc: the raw MCC t_onset_C / total_hr_kJg integrate end-of-run baseline
drift, which drags the onset early (lead's fix, 2026-09-29 17:19: added
t_onset_bc_C / total_hr_bc_kJg). This flag swaps in the corrected columns and
reruns the candidates that use them (C3-C7) plus chf_check(). C0-C2 and R0 do
not read mcc_extras_fsri.csv and are unaffected.

--replicate-table (round 4, T1 in reports/sprint_0929/tti_replicates.md): read
TTI from the canonical cone_replicates_fsri.csv instead of tti_fsri.csv, whose
regex missed the 18 Frame+Grid replicates and which stores non-ignitions as 0 s.
Drops did_not_ignite and early_stop rows; target tti_s; heat flux, specimen
mass and spark_ignitor come from the same table. Runs C0 and C6 only, on that
one row set. Use with --onset-bc. Without the flag nothing changes.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_protocol import bootstrap_ci, repeated_group_kfold, report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA = 0.58
T0_C = 20.0
TARGET = "time_to_ignition_s"


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def ambient(df, col):
    """Per material: mean reading at the lowest Mean_Temp; Dry if any Dry reading exists, else Wet."""
    dry = set(df.loc[df["condition"] == "Dry", "material"])
    df = df[(df["condition"] == "Dry") | ~df["material"].isin(dry)]
    df = df[df["Mean_Temp"] == df.groupby("material")["Mean_Temp"].transform("min")]
    return df.groupby("material").agg(value=(col, "mean"), cond=("condition", "first"))


def log_pos(x):
    x = np.asarray(x, dtype=float)
    return np.log(np.where(x > 0, x, np.nan))


REPLICATES = "--replicate-table" in sys.argv
if REPLICATES:
    rep = load("cone_replicates_fsri.csv")
    tti = rep[~rep["did_not_ignite"] & ~rep["early_stop"]].rename(columns={"tti_s": TARGET}).reset_index(drop=True)
    assert tti[TARGET].gt(0).all()  # every kept replicate ignited (681 rows / 76 materials on 2026-09-29)
else:
    tti = load("tti_fsri.csv")
mcc = load("mcc_hrr.csv").groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean"))
mcc_x = load("mcc_extras_fsri.csv")
if "--onset-bc" in sys.argv:  # lead's fix: raw onset / total HR integrate end-of-run baseline drift
    mcc_x = mcc_x.drop(columns=["t_onset_C", "total_hr_kJg"]).rename(
        columns={"t_onset_bc_C": "t_onset_C", "total_hr_bc_kJg": "total_hr_kJg"})
mcc_x = mcc_x.groupby("material")[["t_onset_C", "t_peak_C", "hrc_JgK", "total_hr_kJg"]].median()
props = load("material_props_fsri.csv").set_index("material")[
    ["categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
k, c = ambient(load("thermal_conductivity_fsri.csv"), "Average_Cond"), ambient(load("specific_heat_fsri.csv"), "Spc_Heat")
thermal = pd.DataFrame({"k_WmK": k["value"], "rhoc_Jm3K": c["value"], "k_cond": k["cond"], "c_cond": c["cond"]})

df = tti.join(mcc, on="material").join(mcc_x, on="material").join(props, on="material").join(thermal, on="material")
assert len(df) == len(tti)  # LEFT joins never drop or duplicate a row (717 rows / 80 materials on 2026-09-29)
df["spark_ignitor"] = df["spark_ignitor"].astype(int)
df["family"] = df["categories"].fillna("unknown").str.split(";").str[0].str.strip()

q = df["heat_flux_kWm2"] * 1e3                      # W/m2
dT = df["t_onset_C"] - T0_C                          # K
L = df["cone_thickness_mm"] / 1e3                    # m
df["log_q"] = np.log(q)
df["log_dT"] = log_pos(dT)
df["log_kpc"] = log_pos(df["k_WmK"] * df["rhoc_Jm3K"])
df["log_L"] = log_pos(L)
df["log_rho"] = log_pos(df["density_kgm3"])
df["log_t_thick"] = np.log(np.pi / 4) + df["log_kpc"] + 2 * df["log_dT"] - 2 * df["log_q"]
df["log_t_thin"] = log_pos(df["rhoc_Jm3K"] * L * dT / q)
df["log_t_thin_mpa"] = log_pos(1e3 * df["cone_mass_per_area_kgm2"] * dT / q)  # c ~ 1 kJ/kg-K assumed

F0 = ["heat_flux_kWm2", "specimen_mass_g", "spark_ignitor", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
F1 = F0 + ["density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]
F2 = F1 + ["t_onset_C", "t_peak_C", "hrc_JgK", "total_hr_kJg", "log_kpc"]
F3 = F2 + ["log_t_thick", "log_t_thin", "log_t_thin_mpa"]
PHYS = ["log_q", "log_dT", "log_kpc", "log_L", "log_rho"]
# Leakage guard: nothing measured during or after the cone test of the material itself.
assert not {TARGET, "end_of_test_time_s", "test_status", "replicate", "test_date"} & set(F3 + PHYS)

HGB = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
           l2_regularization=1.0, random_state=0)


def linear(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


MODELS = [  # (name, estimator, features, train on log1p?)
    ("C0 RF exp004 feats", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                                 random_state=0, n_jobs=-1), F0, True),
    ("C1 HGB exp004 feats", HistGradientBoostingRegressor(**HGB), F0, True),
    ("C2 HGB +props", HistGradientBoostingRegressor(**HGB), F1, True),
    ("C3 HGB +MCC extras +kpc", HistGradientBoostingRegressor(**HGB), F2, True),
    ("C4 HGB +physics composites", HistGradientBoostingRegressor(**HGB), F3, True),
    ("C5 Ridge physics log1p", linear(Ridge(alpha=1.0)), PHYS, True),
    ("C6 Poisson GLM physics", linear(PoissonRegressor(alpha=0.01, max_iter=3000)), PHYS, False),
    ("C7 HGB poisson C4 feats", HistGradientBoostingRegressor(loss="poisson", **HGB), F3, False),
    ("R0 heat flux only", HistGradientBoostingRegressor(**HGB), ["heat_flux_kWm2"], True),
]
SLOW = ["PET", "Nylon_Carpet_High_Pile", "Overstuffed_Chair_Polyester_Batting", "Polyester_Microfiber_Sheet"]

# Post-hoc, run with --exploratory. Added AFTER C0-C7 were scored, so NOT eligible for the pick. The C0-C7
# diagnosis put the error at 25 kW/m2, near each material's critical heat flux, which the loss-free formulas
# above cannot produce. Classical theory: q''_cr ~ sigma (T_ig^4 - T0^4) + h (T_ig - T0), and
# t_ig ~ (pi/4) k rho c (T_ig - T0)^2 / (q'' - q''_cr)^2. eps = 1 and h = 10 W/m2-K, fixed a priori.
Tk, T0k = df["t_onset_C"] + 273.15, T0_C + 273.15
q_cr = 5.670e-8 * (Tk ** 4 - T0k ** 4) + 10.0 * (Tk - T0k)
df["q_cr_kWm2"] = q_cr / 1e3
df["log_q_excess"] = log_pos(q - q_cr)
df["log_t_thick_loss"] = np.log(np.pi / 4) + df["log_kpc"] + 2 * df["log_dT"] - 2 * df["log_q_excess"]
EXPLORATORY = [
    ("X1 Poisson GLM physics + log(q''-q''_cr)", linear(PoissonRegressor(alpha=0.01, max_iter=3000)),
     PHYS + ["log_q_excess"], False),
    ("X2 HGB poisson C4 feats + CHF terms", HistGradientBoostingRegressor(loss="poisson", **HGB),
     F3 + ["log_q_excess", "log_t_thick_loss"], False),
]


def chf_check():
    """No fitting: does the loss term predict WHICH materials blow up at 25 kW/m2?"""
    ign = df[df[TARGET] > 0]
    mats = ign.drop_duplicates("material").set_index("material")["q_cr_kWm2"].dropna()
    print(f"q''_cr from MCC onset T: {len(mats)} materials, min {mats.min():.1f}, median {mats.median():.1f}, "
          f"max {mats.max():.1f} kW/m2; rows with q'' <= q''_cr: {(q <= q_cr).sum()}")
    med = ign.groupby(["material", "heat_flux_kWm2"])[TARGET].median().unstack()
    obs = (med[25.0] / med[50.0]).dropna()
    obs = obs[obs.index.isin(mats.index)]
    pred = ((50 - mats[obs.index]) / (25 - mats[obs.index])) ** 2
    print(f"TTI(25)/TTI(50), observed vs loss-model prediction ((50-q_cr)/(25-q_cr))^2, {len(obs)} materials: "
          f"Spearman {spearmanr(obs, pred).statistic:+.3f}; predicted median {pred.median():.2f}, observed {obs.median():.2f}")
    s = ign[["log_t_thick", "log_t_thick_loss", TARGET]].dropna()
    print(f"Spearman with TTI on the same {len(s)} ignited rows: log_t_thick {spearmanr(s['log_t_thick'], s[TARGET]).statistic:+.3f}, "
          f"log_t_thick_loss {spearmanr(s['log_t_thick_loss'], s[TARGET]).statistic:+.3f}")


def coverage():
    mats = df.drop_duplicates("material").set_index("material")
    print("Feature coverage, materials with a value (of 80):")
    for f in F3 + PHYS[:1] + ["log_dT", "log_L", "log_rho"]:
        print(f"  {f:26s} {mats[f].notna().sum():3d}")
    for col in ["k_cond", "c_cond"]:
        print(f"  {col}: {mats[col].value_counts().to_dict()} (Wet = no Dry reading, fell back)")
    both = mats["log_kpc"].notna()
    print(f"  k rho c from Dry k and Dry c: {(both & (mats.k_cond == 'Dry') & (mats.c_cond == 'Dry')).sum()}, "
          f"with a Wet fallback: {(both & ((mats.k_cond == 'Wet') | (mats.c_cond == 'Wet'))).sum()}")
    print(f"  families (first material.json category): {mats['family'].value_counts().to_dict()}")
    print("Physical range check (material-level):")
    for f in ["density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2", "t_onset_C", "k_WmK", "rhoc_Jm3K"]:
        s = mats[f].dropna()
        print(f"  {f:24s} min {s.min():10.4g} ({s.idxmin()})  max {s.max():10.4g} ({s.idxmax()})")


def diagnostics():
    """Target-side facts that bound what any bench-feature model can score. No model fitting."""
    zero = df[df[TARGET] <= 0]
    print(f"TTI = 0 rows (non-ignitions per POST TEST CMT, see docstring): {len(zero)} rows / "
          f"{zero['material'].nunique()} materials")
    ign = df[df[TARGET] > 0]
    slopes = [np.polyfit(np.log(g["heat_flux_kWm2"]), np.log(g[TARGET]), 1)[0]
              for _, g in ign.groupby("material") if g["heat_flux_kWm2"].nunique() >= 3]
    print(f"Within-material d log(t_ig) / d log(q''): {len(slopes)} materials, median {np.median(slopes):.2f} "
          f"(IQR {np.percentile(slopes, 25):.2f} to {np.percentile(slopes, 75):.2f}); theory: thick -2, thin -1")
    med = ign.groupby(["material", "heat_flux_kWm2"])[TARGET].median().unstack()
    ratio = (med[25.0] / med[50.0]).dropna()
    near = ratio[ratio > 6].index
    ss = (ign[TARGET] - ign[TARGET].mean()) ** 2
    print(f"TTI(25)/TTI(50) median over {len(ratio)} materials: {ratio.median():.2f} (thick-solid law: 4). "
          f"max {ratio.max():.1f} ({ratio.idxmax()}). {len(near)} materials exceed 6; their 25 kW/m2 rows hold "
          f"{ss[ign['material'].isin(near) & (ign['heat_flux_kWm2'] == 25)].sum() / ss.sum():.2f} of the ignited-row SS")
    for f in ["log_t_thick", "log_t_thin", "log_t_thin_mpa", "log_q"]:
        s = ign[[f, TARGET, "material"]].dropna()
        print(f"  Spearman({f}, TTI), no fitting: {spearmanr(s[f], s[TARGET]).statistic:+.3f} "
              f"({s['material'].nunique()} materials)")


def ceiling(data):
    """R2 of predicting each replicate from its own material x flux x spark cell (the replicate-noise ceiling)."""
    y = data[TARGET]
    cell = data.groupby(["material", "heat_flux_kWm2", "spark_ignitor"])[TARGET]
    n, s = cell.transform("size"), cell.transform("sum")
    m = n >= 2
    print(f"  cell-mean R2 in-sample (upper bound for any cell-constant feature set): "
          f"{r2_score(y, cell.transform('mean')):.4f}")
    print(f"  leave-one-replicate-out cell-mean R2 ({m.sum()} rows in cells with >=2 reps): "
          f"{r2_score(y[m], ((s - y) / (n - 1))[m]):.4f}")


def run(data, label, models):
    y, g, fam = data[TARGET].to_numpy(float), data["material"].to_numpy(), data["family"].to_numpy()
    print(f"\n===== {label}: {len(data)} rows / {data['material'].nunique()} materials / "
          f"{data['family'].nunique()} families =====")
    ceiling(data)
    rows = []
    for name, model, feats, logt in models:
        X = data[feats].to_numpy(float)
        out = report(f"[{label}] {name}", model, X, y, g, log_target=logt, sota=SOTA)
        frk, foof = repeated_group_kfold(model, X, y, fam, log_target=logt)
        flo, fhi = bootstrap_ci(y, foof[0], g)
        print(f"  Family-grouped GKF 5x10 (pooled OOF):          mean R2={frk.mean():.4f} std={frk.std():.4f}; "
              f"material-bootstrap 95% CI [{flo:.3f}, {fhi:.3f}]")
        _, oof = repeated_group_kfold(model, X, y, g, repeats=1, log_target=logt)   # == report()'s repeat-0 OOF
        se = pd.Series((y - oof[0]) ** 2).groupby(g).sum().sort_values(ascending=False)
        share = (se / se.sum()).head(5)
        keep = ~np.isin(g, SLOW)
        print("  SSE share by material (GKF repeat 0): " + ", ".join(f"{m} {v:.2f}" for m, v in share.items()))
        print(f"  R2 on rows outside the 4 slowest-igniting materials (models still trained on them): "
              f"{r2_score(y[keep], oof[0][keep]):.4f}; max OOF prediction {oof[0].max():.0f} s")
        f25 = data["heat_flux_kWm2"].to_numpy() == 25
        print(f"  OOF R2 in log1p space {r2_score(np.log1p(y), np.log1p(np.clip(oof[0], 0, None))):.4f}; "
              f"seconds R2 on 25 kW/m2 rows {r2_score(y[f25], oof[0][f25]):.4f}, other fluxes "
              f"{r2_score(y[~f25], oof[0][~f25]):.4f}")
        rows.append(dict(set=label, candidate=name, rows=out["n_rows"], materials=out["n_materials"],
                         gss=f"{out['gss_mean']:.3f} +- {out['gss_std']:.3f}",
                         gkf=f"{out['rkf_mean']:.3f} +- {out['rkf_std']:.3f}",
                         ci=f"[{out['ci_lo']:.3f}, {out['ci_hi']:.3f}]",
                         fam=f"{frk.mean():.3f} +- {frk.std():.3f}", fam_ci=f"[{flo:.3f}, {fhi:.3f}]",
                         top4_share=f"{se[se.index.isin(SLOW)].sum() / se.sum():.2f}",
                         r2_excl_slow=f"{r2_score(y[keep], oof[0][keep]):.3f}"))
    return rows


if __name__ == "__main__":
    if REPLICATES:  # pre-declared T1: C6 and its same-rows baseline C0, nothing else
        models = [m for m in MODELS if m[0][:2] in ("C0", "C6")]
    elif "--onset-bc" in sys.argv:  # rerun the onset-using models on baseline-corrected MCC columns
        chf_check()
        models = [m for m in MODELS if m[0][:2] in ("C3", "C4", "C5", "C6", "C7")]
    elif "--exploratory" in sys.argv:
        chf_check()
        models = EXPLORATORY
    else:
        coverage()
        diagnostics()
        models = MODELS
    if REPLICATES:  # non-ignitions and early stops were dropped at load
        results = run(df, "replicate table", models)
    else:
        results = run(df, "all rows", models) + run(df[df[TARGET] > 0], "ignited only", models)
    print("\nSummary (R2 in raw seconds):")
    print(pd.DataFrame(results).to_string(index=False))
