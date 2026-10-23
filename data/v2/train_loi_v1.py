#!/usr/bin/env python3
"""Dravix V2 -- first LOI model. Pure polymers, FAA DOT/FAA/AR-05/14 Appendix A.

WHY: LOI is the Tier-1 target and had no real rows. loi_faa_polymers.csv (from
ingest_faa_loi.py) gives, for 32 pure/commercial polymers, LOI next to the MCC/PCFC
heat release capacity eta_c. 26 of them also have char yield mu, PCFC heat of
combustion hc0, and flaming combustion efficiency chi. The report's own physics
(eq. 76, fig. 21) says LOI ~ eta_c^(-1/4), so log(LOI) is linear in log(eta_c).

Configs FIXED BEFORE THE FIRST RUN (no tuning, no feature search):
  A   LinearRegression on log10(eta_c)                    the report's physics, 1 feature
  B   Ridge(alpha=1) on log10(eta_c), mu, hc0             MCC-only (one MCC run gives all three)
  C   Ridge(alpha=1) on B + chi                           adds gas-phase flame inhibition (halogens),
                                                          which eta_c cannot see; chi = HOC/hc0 needs a cone test
  RF  RandomForest(500 trees, min_samples_leaf=2, seed 0) on C's features
All models train on log1p(LOI) and are scored in raw LOI (% O2). Ridge inputs are standardized.
Headline picked a priori: C. Physics says LOI depends on condensed-phase fuel
release (eta_c, mu, hc0) AND gas-phase inhibition (chi), and C is the simplest model
with both. A on the same 26 rows is the same-rows baseline. A on all 32 rows is also reported.

Groups = polymer (one row each). Protocols: eval_protocol.report (legacy GroupShuffleSplit
x10 + repeated GroupKFold 5x10 with material-bootstrap CI), leave-one-out, and a harder
leave-one-FAMILY-out split (families below were assigned by backbone chemistry before running;
they are our labels, not the report's).
Pre-declared sensitivity: rerun without PTFE and FEP (LOI 95 each), which inflate target
variance and so inflate R2.
Reference, not a fitted model: the report's formula LOI = 125 / eta_c^0.25. Its authors fitted
it on these polymers, so its R2 here is in-sample, not held-out.

Rerun: python3 ingest_faa_loi.py && python3 train_loi_v1.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_protocol import report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA = 0.84  # sun_fu_2026_rpuf: random 80/20 split, 435 RPUF formulations (different sample space)
SOTA_MSE = 1.79  # same paper, same split (benchmarks/sota_board.csv)

FAMILY = {
    "HIPS": "styrenic", "PS": "styrenic", "ABS": "styrenic",
    "PP": "polyolefin", "PE": "polyolefin", "PE (chlorinated)": "polyolefin",
    "PET": "aromatic polyester", "PBT": "aromatic polyester", "PEN": "aromatic polyester",
    "PA6": "aliphatic polyamide", "PA66": "aliphatic polyamide",
    "KEVLAR": "aramid", "NOMEX": "aramid",
    "PC": "polycarbonate", "PC-FR": "polycarbonate",
    "PTFE": "fluoropolymer", "FEP": "fluoropolymer", "ETFE": "fluoropolymer",
    "PVC (rigid)": "chloropolymer", "CR": "chloropolymer",
    "PESU": "polysulfone", "PPSU": "polysulfone",
    "PAI": "polyimide", "PEI": "polyimide",
    "PEEK": "polyketone", "PMMA": "acrylic", "POM": "acetal",
    "EP": "thermoset", "CEA": "thermoset",
    "SIR (filled)": "silicone", "PPS": "polysulfide", "PBI": "polybenzimidazole",
}


def logt(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


RIDGE = logt(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
CONFIGS = {  # name -> (features, model)
    "A": (["log10_HRC"], logt(LinearRegression())),
    "B": (["log10_HRC", "char_yield_g_g", "hc0_kJ_g"], RIDGE),
    "C": (["log10_HRC", "char_yield_g_g", "hc0_kJ_g", "chi"], RIDGE),
    "RF": (["log10_HRC", "char_yield_g_g", "hc0_kJ_g", "chi"],
           logt(RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0))),
}


def held_out(model, X, y, groups):
    """Pooled out-of-group R2 and MAE (LOI % units); groups=polymer gives LOO."""
    pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())
    assert np.isfinite(pred).all()
    return r2_score(y, pred), mean_absolute_error(y, pred), pred


def run(df, label):
    print(f"\n=== {label}: {len(df)} polymers, {df['family'].nunique()} families, "
          f"LOI {df['LOI_pct'].min():.0f}-{df['LOI_pct'].max():.0f}, sd {df['LOI_pct'].std(ddof=0):.2f} ===")
    y, poly, fam = df["LOI_pct"].to_numpy(), df["polymer"].to_numpy(), df["family"].to_numpy()
    ref = 125 / df["HRC_J_gK"].to_numpy() ** 0.25
    print(f"  reference LOI=125/eta_c^0.25 (in-sample for its authors): R2={r2_score(y, ref):.3f} "
          f"MAE={mean_absolute_error(y, ref):.2f}")
    rows = []
    for name, (feats, model) in CONFIGS.items():
        if df[feats].isna().any().any():
            continue
        X = df[feats].to_numpy()
        out = report(f"  [{name}] polymer-grouped", model, X, y, poly, sota=SOTA)
        fam_out = report(f"  [{name}] family-grouped ", model, X, y, fam, sota=SOTA)
        loo_r2, loo_mae, pred = held_out(model, X, y, poly)
        lofo_r2, lofo_mae, _ = held_out(model, X, y, fam)
        rows.append(dict(config=name, n=len(y), gss_mean=out["gss_mean"], gkf_mean=out["rkf_mean"],
                         ci=f"[{out['ci_lo']:.2f}, {out['ci_hi']:.2f}]", loo_r2=loo_r2, loo_mae=loo_mae,
                         loo_rmse=np.sqrt(np.mean((pred - y) ** 2)),
                         fam_gkf=fam_out["rkf_mean"], lofo_r2=lofo_r2, lofo_mae=lofo_mae))
        if name == "C":
            resid = pd.DataFrame({"polymer": poly, "LOI": y, "LOO_pred": pred.round(1)})
            resid["abs_err"] = (resid["LOO_pred"] - resid["LOI"]).abs()
            print("  [C] worst leave-one-out errors:\n" +
                  resid.sort_values("abs_err", ascending=False).head(6).to_string(index=False))
    print(f"\n  SUMMARY ({label})\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    df = pd.read_csv(f"{DATA_DIR}/loi_faa_polymers.csv").dropna(subset=["LOI_pct", "HRC_J_gK"])
    assert df["polymer"].is_unique and set(df["polymer"]) <= set(FAMILY)
    df["family"] = df["polymer"].map(FAMILY)
    df["log10_HRC"] = np.log10(df["HRC_J_gK"])
    full = df.dropna(subset=["char_yield_g_g", "hc0_kJ_g", "chi"])
    # R2 = 1 - MSE/Var on one test set, so sun_fu's R2 and MSE imply its test-set LOI spread.
    print(f"sun_fu_2026_rpuf for scale: RMSE {np.sqrt(SOTA_MSE):.2f}, "
          f"implied test-set LOI sd {np.sqrt(SOTA_MSE / (1 - SOTA)):.2f} (assumes R2 and MSE on the same split)")
    run(df, "all polymers with LOI + eta_c (config A only)")
    run(full, "polymers with every feature (same rows for A/B/C/RF)")
    run(full[~full["polymer"].isin(["PTFE", "FEP"])], "sensitivity: same rows minus PTFE, FEP")
