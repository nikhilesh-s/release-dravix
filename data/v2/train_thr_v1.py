#!/usr/bin/env python3
"""Dravix V2 -- first Total Heat Release (THR, MJ/m2) model.

Competitor: pomazi_2026 Table 9 real held-out XGBoost R2=0.67 (NOT their 0.88
full-dataset fit). Target: thr_fsri.csv (trapezoidal integral of FSRI's
replicate-averaged HRRPUA curve, from ignition to the curve's own end).

WHY a physics baseline first: THR is close to an energy balance. The heat a
specimen can release per unit area is its mass per area m'' (kg/m2, weighed
before the test) times the heat of complete combustion of its volatiles per
gram of sample h_c (MCC total heat release, kJ/g = MJ/kg), times a combustion
efficiency chi <= 1 (incomplete flaming combustion, unburnt char, test ended
before burnout). So THR_pred = chi * m'' * h_c needs no fire-test outcome at
all. If that already explains most of the variance, an ML model should be
judged by what it adds on top of it, not against zero.

Leakage check on the inputs: SPECIMEN MASS and SURF AREA are pre-test fields
of each cone test's Scalar file; h_c, HRC, char yield come from separate MCC
runs on milligram samples; density/thickness/category from material.json and
PRE TEST CMT. Nothing is derived from the cone HRR curve or post-test mass.

Pre-declared variants (fixed before any held-out number was computed):
  core rows = THR rows with m''_flux and MCC h_c (needed for the physics product)
  V0  zero-parameter THR_pred = m''_flux * h_c (chi=1): R2 on all core rows and per
      heat flux, no training. Sensitivity: material-level m'' from material_props.
  V1  LinearRegression(fit_intercept=False) on [phys]           (chi fit on train folds)
  V1f LinearRegression(fit_intercept=False) on [phys, phys*q'']  (chi linear in flux)
  V2r RandomForest(300, max_depth=6, min_samples_leaf=2) on FULL features
  V2h HistGradientBoosting(max_iter=300, lr=0.05, max_leaf_nodes=15, min_samples_leaf=10) on FULL
  B_r RandomForest (same settings) on the pHRR (exp002) feature set -- same rows/splits ablation
  B_h HistGradientBoosting (same settings) on the pHRR feature set
  wide rows (all 220 THR rows, NaN features allowed): V2h vs B_h
  family-grouped split (first material.json category): V1, V2h, B_h on core rows
Settings are the exp002 RF settings and sklearn-typical HGB settings, not tuned.
Pick rule (declared before results): highest material-grouped GKF mean on core
rows; if a variant with fewer fitted parameters is within 0.02, pick that one.

Added after the first run, and disclosed as such: the V0 residuals pointed at 6 rows
(Face_Shield, High_Temperature_SCBA_Facepiece at 50/75/100 kW/m2) whose single replicate
was pulled out of the cone ~10 s after ignition, per that test's own POST TEST CMT. Those
rows are not a total heat release. So the same variant list is run twice: "as-declared"
(all rows, printed first) and "clean" (drops rows where EVERY replicate's POST TEST CMT
matches EARLY_STOP). The filter reads test notes only, never residuals or predictions.

Added third, after the coordinator flagged that FSRI's averaged HRRPUA files can end before the
longest replicate: "clean, untruncated target" re-runs the same list on the clean rows with the
target replaced by the mean of per-replicate integrals from the raw Scan files (each replicate from
ignition to its own END OF TEST TIME; scan_check). That target avoids both averaging artifacts: a
file cut at the shortest replicate (THR too low), and a file whose 'Mean' column keeps running on
the surviving replicates only (THR too high).

Added fourth, a data bug fix from the lead: mcc_extras_fsri.csv's total_hr_kJg integrated the MCC's
end-of-run baseline drift, and the new total_hr_bc_kJg column is the corrected value. The last block
re-runs the headline set (and V0/V1 on the as-declared rows) with h_c = total_hr_bc_kJg, then prints
raw -> corrected side by side.

Run: python3 data/v2/train_thr_v1.py  (~5 min)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from eval_protocol import report, repeated_group_kfold
from ingest_fsri_props import read_scalar
from ingest_fsri_sta_mcc import RAW, to_float

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA = 0.67
TRUNC_FRAC = 0.10  # curve still above 10% of its own peak at its last point -> "ended while burning"
# Operator stopped the test on purpose (specimen pulled ~10 s after ignition, or ended at 1-2 min because
# the specimen swelled toward the heater). THR of such a test is not a total heat release.
EARLY_STOP = re.compile(r"removed.{0,40}after ign|removed and extinguished|Test ended at \d+ (?:seconds|minutes?)\.", re.I)


def scalar_table():
    """One row per cone replicate: flux, replicate no., pre-test mass/area, end time after ignition."""
    rows = []
    for p in glob.glob(os.path.join(RAW, "*", "Cone", "*Scalar*.csv")):
        m = re.match(r".+_Cone_HF(\d+)(.*)Scalar_\d+_R(\d+)\.csv", os.path.basename(p))
        if not m or "noSpark" in m.group(2):
            continue
        kv = read_scalar(p)
        mass, area = to_float(kv.get("SPECIMEN MASS")), to_float(kv.get("SURF AREA"))
        end, tti = to_float(kv.get("END OF TEST TIME")), to_float(kv.get("TIME TO IGN"))
        rows.append({"material": os.path.basename(os.path.dirname(os.path.dirname(p))),
                     "heat_flux_kWm2": float(m.group(1)), "rep": int(m.group(3)),
                     "mpa_kgm2": mass / area / 1000 if mass and area else np.nan,
                     "early_stop": bool(EARLY_STOP.search(kv.get("POST TEST CMT", ""))),
                     "end_after_ign_s": end - tti if end is not None and tti is not None else np.nan})
    return pd.DataFrame(rows)


def audit_target(thr, scal):
    """Recompute THR from the raw HRRPUA files with independent code; flag curves that end while burning."""
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*_HRRPUA_*.csv"))):
        mat = os.path.basename(os.path.dirname(os.path.dirname(p)))
        flux = float(re.search(r"_HRRPUA_(\d+)\.csv$", p).group(1))
        df = pd.read_csv(p)
        t, mean = df.iloc[:, 0].to_numpy(float), df["Mean"].to_numpy(float)
        reps = [c for c in df.columns if re.match(r".+_R\d+$", c)]
        rep_thr, rep_tail = [], []
        for c in reps:
            ok = df[c].notna().to_numpy()
            rep_thr.append(trapezoid(df[c].to_numpy(float)[ok], t[ok]) / 1000)
            v = df[c].to_numpy(float)[ok]
            rep_tail.append(v[-1] / v.max() if v.max() > 0 else 0.0)
        s = scal[(scal.material == mat) & (scal.heat_flux_kWm2 == flux)]
        out.append({"material": mat, "heat_flux_kWm2": flux,
                    "thr_recomputed": trapezoid(mean, t) / 1000,
                    "thr_mean_of_reps": np.mean(rep_thr),
                    "curve_end_s": t[-1], "peak_mean": mean.max(), "final_mean": mean[-1],
                    "final_frac": mean[-1] / mean.max() if mean.max() > 0 else 0.0,
                    "max_rep_tail_frac": max(rep_tail),
                    "scalar_end_after_ign_min": s.end_after_ign_s.min(),
                    "scalar_end_after_ign_max": s.end_after_ign_s.max()})
    a = thr.merge(pd.DataFrame(out), on=["material", "heat_flux_kWm2"], how="left")
    diff = (a.thr_recomputed - a.total_heat_release_MJm2).abs()
    print(f"[audit] {a.thr_recomputed.notna().sum()}/{len(a)} rows recomputed from raw HRRPUA files; "
          f"max |recomputed - thr_fsri.csv| = {diff.max():.3f} MJ/m2")
    pm = a[(a.material == "PMMA") & (a.heat_flux_kWm2 == 50)].iloc[0]
    print(f"[audit] PMMA @50: {pm.thr_recomputed:.2f} MJ/m2 (csv {pm.total_heat_release_MJm2}), curve ends "
          f"{pm.curve_end_s}s after ignition at {pm.final_mean:.0f} kW/m2 ({100 * pm.final_frac:.0f}% of peak)")
    rel = (a.thr_mean_of_reps - a.thr_recomputed) / a.thr_recomputed.clip(lower=1)
    print(f"[audit] mean-of-replicate THR vs THR of mean curve: median rel diff {rel.median():+.3f}, "
          f"|rel| > 5% on {(rel.abs() > 0.05).sum()} rows")
    lag = a.curve_end_s - a.scalar_end_after_ign_max
    print(f"[audit] curve end - longest replicate's (END OF TEST TIME - TTI): median {lag.median():+.2f}s, "
          f"|diff| > 5s on {(lag.abs() > 5).sum()} rows")
    a["truncated"] = a.final_frac > TRUNC_FRAC
    print(f"[audit] mean curve ends above {TRUNC_FRAC:.0%} of its peak on {a.truncated.sum()} rows / "
          f"{a[a.truncated].material.nunique()} materials; any single replicate does on "
          f"{(a.max_rep_tail_frac > TRUNC_FRAC).sum()} rows")
    print(f"[audit] THR < 2 MJ/m2 (essentially no flaming heat): {(a.total_heat_release_MJm2 < 2).sum()} rows")
    return a


def scan_check(a):
    """Second, fully independent check from the raw instrument Scan files (O2 analyzer, duct pressure/temp).

    HRR uses FSRI's own O2-only equation (02_Scripts/plot_Cone_data_html.py: 1.10 * 13100 kJ/kg O2 * EDF * ODF)
    divided by the Scalar SURF AREA. Gives (i) per-replicate THR ignition -> END OF TEST TIME, to compare with
    thr_fsri.csv, and (ii) heat still released in the 60 s after END OF TEST TIME (the scan keeps recording),
    which measures how much THR the declared end of test cuts off.
    """
    rows = []
    for scal in glob.glob(os.path.join(RAW, "*", "Cone", "*Scalar*.csv")):
        m = re.match(r".+_Cone_HF(\d+)(.*)Scalar_\d+_R(\d+)\.csv", os.path.basename(scal))
        scan = scal.replace("Scalar", "Scan")
        if not m or "noSpark" in m.group(2) or not os.path.exists(scan):
            continue
        d = pd.read_csv(scan, header=0, skiprows=[1, 2, 3, 4], index_col="Names")
        s = pd.read_csv(scal, index_col=0).squeeze()
        o2 = d["O2 Meter"] / 100
        edf = np.sqrt(d["Exh Press"] / (d["Stack TC"] + 273.15)) * float(s["C FACTOR"])
        q = (1.10 * 13100 * edf * (o2["Baseline"] - o2) / (1.105 - 1.5 * o2)).drop("Baseline")
        t = pd.to_numeric(d.Time.drop("Baseline"), errors="coerce")
        ok = (t.notna() & q.notna()).to_numpy()
        t, q = t.to_numpy(float)[ok], q.to_numpy(float)[ok] / float(s["SURF AREA"])  # kW/m2
        tti, end = float(s["TIME TO IGN"]), float(s["END OF TEST TIME"])
        main, tail = (t >= tti) & (t <= end), (t >= end) & (t <= end + 60)
        rows.append({"material": os.path.basename(os.path.dirname(os.path.dirname(scal))),
                     "heat_flux_kWm2": float(m.group(1)),
                     "thr_scan": trapezoid(q[main], t[main]) / 1000,
                     "tail60": trapezoid(q[tail], t[tail]) / 1000 if tail.sum() > 1 else np.nan})
    s = pd.DataFrame(rows).groupby(["material", "heat_flux_kWm2"]).agg(
        thr_scan=("thr_scan", "mean"), tail60=("tail60", "mean"), n_scan_reps=("thr_scan", "size")).reset_index()
    a = a.merge(s, on=["material", "heat_flux_kWm2"], how="left")
    rel = (a.thr_scan - a.total_heat_release_MJm2) / a.total_heat_release_MJm2.clip(lower=1)
    print(f"[scan] {len(rows)} replicate scans -> {a.thr_scan.notna().sum()}/{len(a)} THR rows; mean-of-replicate "
          f"THR from raw O2 data vs thr_fsri.csv: median rel diff {rel.median():+.4f}, |rel| > 5% on "
          f"{(rel.abs() > 0.05).sum()} rows")
    q = a.tail60.quantile([0.5, 0.9, 1.0]).round(2).tolist()
    frac = a.tail60 / a.total_heat_release_MJm2.clip(lower=1)
    print(f"[scan] heat in the 60 s after END OF TEST TIME: median {q[0]} / p90 {q[1]} / max {q[2]} MJ/m2; "
          f"> 10% of the row's THR on {(frac > 0.10).sum()} rows (of which THR < 20 MJ/m2: "
          f"{((frac > 0.10) & (a.total_heat_release_MJm2 < 20)).sum()})")
    # Does integrating FSRI's averaged curve (which can end before the longest replicate) understate THR?
    d = a.thr_scan - a.total_heat_release_MJm2
    early = (a.curve_end_s - a.scalar_end_after_ign_max) < -5
    print(f"[scan] per-replicate (untruncated) THR minus thr_fsri.csv: median {d.median():+.2f}, p10 {d.quantile(.1):+.2f}, "
          f"p90 {d.quantile(.9):+.2f} MJ/m2; higher by >5% AND >2 MJ/m2 on {((rel > .05) & (d > 2)).sum()} rows, "
          f"lower by >5% AND >2 MJ/m2 on {((rel < -.05) & (d < -2)).sum()} rows")
    print(f"[scan]   rows whose averaged curve ends >5 s before the longest replicate: {early.sum()}; there median "
          f"{d[early].median():+.2f}, max {d[early].max():+.2f} MJ/m2. Scan replicate count != HRRPUA file's on "
          f"{(a.n_scan_reps != a.n_replicates).sum()} rows")
    cols = ["material", "heat_flux_kWm2", "total_heat_release_MJm2", "thr_scan", "curve_end_s",
            "scalar_end_after_ign_min", "scalar_end_after_ign_max", "n_replicates", "n_scan_reps"]
    print("[scan]   largest differences (averaged curve cut at the shortest replicate -> too low; running on with only\n"
          "         the surviving replicates in its 'Mean' -> too high):")
    print(a.assign(diff=d).sort_values("diff")[cols + ["diff"]].iloc[np.r_[0:6, -8:0]].round(2).to_string(index=False))
    return a


def build_features(thr, scal):
    props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")
    mx = pd.read_csv(f"{DATA_DIR}/mcc_extras_fsri.csv").groupby("material").agg(
        hc_kJg=("total_hr_kJg", "mean"), hc_bc_kJg=("total_hr_bc_kJg", "mean"), hrc_JgK=("hrc_JgK", "mean")).reset_index()
    mh = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv").groupby("material").agg(
        mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean")).reset_index()
    mflux = scal.groupby(["material", "heat_flux_kWm2"]).mpa_kgm2.median().rename("mpa_flux_kgm2").reset_index()
    df = (thr.merge(mflux, on=["material", "heat_flux_kWm2"], how="left")
          .merge(props[["material", "categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]],
                 on="material", how="left")
          .merge(mx, on="material", how="left").merge(mh, on="material", how="left"))
    df["phys_MJm2"] = df.mpa_flux_kgm2 * df.hc_kJg  # kg/m2 * MJ/kg
    df["phys_x_flux"] = df.phys_MJm2 * df.heat_flux_kWm2
    df["family"] = df.categories.fillna("unknown").str.split(";").str[0].str.strip()
    return df


FULL = ["heat_flux_kWm2", "mpa_flux_kgm2", "hc_kJg", "mcc_char_yield_pct", "hrc_JgK",
        "density_kgm3", "cone_thickness_mm", "phys_MJm2"]
BASE = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
RF = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0, n_jobs=1)
HGB = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                    min_samples_leaf=10, random_state=0)
LIN = LinearRegression(fit_intercept=False)


def zero_param(df):
    y, p = df.total_heat_release_MJm2.to_numpy(), df.phys_MJm2.to_numpy()
    chi = y / p
    print(f"V0 zero-parameter THR = m''_flux * h_c (chi=1): {len(df)} rows / {df.material.nunique()} materials, "
          f"R2 = {r2_score(y, p):.4f}")
    print(f"   observed chi = THR / (m'' h_c): median {np.median(chi):.3f}, IQR "
          f"[{np.percentile(chi, 25):.3f}, {np.percentile(chi, 75):.3f}], min {chi.min():.3f}, max {chi.max():.3f}")
    for q, g in df.groupby("heat_flux_kWm2"):
        r2 = r2_score(g.total_heat_release_MJm2, g.phys_MJm2) if len(g) > 2 else float("nan")
        print(f"   flux {q:5.0f}: n={len(g):3d}  R2={r2:.4f}  median chi={np.median(g.total_heat_release_MJm2 / g.phys_MJm2):.3f}")
    odd = df.assign(chi=chi, ehc_MJkg=y / df.mpa_flux_kgm2)[(chi > 1.1) | (chi < 0.1)]
    print("   rows with chi > 1.1 (more heat than complete combustion allows) or chi < 0.1:")
    print(odd[["material", "heat_flux_kWm2", "total_heat_release_MJm2", "integrated_to_time_s", "mpa_flux_kgm2",
               "hc_kJg", "ehc_MJkg", "chi"]].sort_values("chi").round(3).to_string(index=False))
    pm = df.cone_mass_per_area_kgm2 * df.hc_kJg
    print(f"   sensitivity, material-level m'' (material_props median): R2 = {r2_score(y, pm):.4f}")
    for flag, g in df.groupby("truncated"):
        print(f"   truncated={flag}: n={len(g)}, median chi={np.median(g.total_heat_release_MJm2 / g.phys_MJm2):.3f}, "
              f"R2={r2_score(g.total_heat_release_MJm2, g.phys_MJm2):.4f}")


def run(tag, df, variants, groups_col="material"):
    results = {}
    y, groups = df.total_heat_release_MJm2.to_numpy(), df[groups_col].to_numpy()
    for name, model, feats in variants:
        results[name] = report(f"{tag} {name} [{', '.join(feats)}]", model, df[feats].to_numpy(float), y, groups,
                               sota=SOTA)
    return results


def evaluate(df, label):
    core = df.dropna(subset=["phys_MJm2"]).reset_index(drop=True)
    print(f"\n[{label}] rows: all {len(df)} / {df.material.nunique()} materials; core (m''_flux and MCC h_c) "
          f"{len(core)} / {core.material.nunique()}\n")
    zero_param(core)

    print("\n== material-grouped, core rows ==")
    res = run("core", core, [("V1", LIN, ["phys_MJm2"]),
                             ("V1f", LIN, ["phys_MJm2", "phys_x_flux"]),
                             ("V2r", RF, FULL), ("V2h", HGB, FULL),
                             ("B_r", RF, BASE), ("B_h", HGB, BASE)])
    print("\n== material-grouped, wide rows (all THR rows, NaN features allowed) ==")
    run("wide", df, [("V2h", HGB, FULL), ("B_h", HGB, BASE)])
    print(f"\n== family-grouped (first category; {core.family.nunique()} families), core rows ==")
    run("family", core, [("V1", LIN, ["phys_MJm2"]), ("V2h", HGB, FULL), ("B_h", HGB, BASE)], groups_col="family")

    # R2 depends on the target's spread. Scoring the same OOF predictions inside one heat flux or one
    # material family (a narrow sample space, closer to Pomazi's single composite family) shows how
    # much of the headline comes from separating very different kinds of material.
    print("\n== stratified R2 of the pooled OOF predictions (repeat 0), core rows, material-grouped ==")
    y, g = core.total_heat_release_MJm2.to_numpy(), core.material.to_numpy()
    for name, model, feats in [("V1", LIN, ["phys_MJm2"]), ("V2h", HGB, FULL)]:
        _, oof = repeated_group_kfold(model, core[feats].to_numpy(float), y, g, repeats=1)
        for col in ["heat_flux_kWm2", "family"]:
            for key, idx in core.groupby(col).indices.items():
                if core.material.iloc[idx].nunique() >= 4:
                    print(f"   {name} {col}={key}: n={len(idx)} rows / {core.material.iloc[idx].nunique()} mats, "
                          f"THR std {y[idx].std():.1f} MJ/m2, R2={r2_score(y[idx], oof[0][idx]):.3f}")
    print("\n   non-truncated core rows only (diagnostic, not a headline):")
    nt = core[~core.truncated].reset_index(drop=True)
    run("core-nontrunc", nt, [("V1", LIN, ["phys_MJm2"]), ("V2h", HGB, FULL)])

    best = max(res, key=lambda k: res[k]["rkf_mean"])
    print(f"\n[{label}] best GKF mean on core rows: {best} {res[best]['rkf_mean']:.4f}")
    return res


if __name__ == "__main__":
    thr = pd.read_csv(f"{DATA_DIR}/thr_fsri.csv")
    scal = scalar_table()
    audit = scan_check(audit_target(thr, scal))
    df = build_features(thr, scal).merge(audit[["material", "heat_flux_kWm2", "truncated", "final_frac", "thr_scan"]],
                                         on=["material", "heat_flux_kWm2"])
    es = scal.groupby(["material", "heat_flux_kWm2"]).early_stop.mean().rename("early_frac").reset_index()
    df = df.merge(es, on=["material", "heat_flux_kWm2"], how="left")
    print("replicates stopped early by the operator (POST TEST CMT):",
          scal[scal.early_stop].groupby("material").heat_flux_kWm2.apply(sorted).to_dict())
    evaluate(df, "as-declared")
    clean = df[df.early_frac != 1].reset_index(drop=True)
    print(f"\n\n######## CLEAN: drop {len(df) - len(clean)} rows whose every replicate was stopped early "
          f"(test notes, not residuals) ########")
    evaluate(clean, "clean")
    # Same clean rows, target = mean of per-replicate integrals from the raw scans, each to its own
    # END OF TEST TIME (not cut where FSRI's averaged curve ends). Added after the coordinator flagged
    # that the averaged files can end before the longest replicate.
    untr = clean.assign(total_heat_release_MJm2=clean.thr_scan).dropna(subset=["total_heat_release_MJm2"])
    print(f"\n\n######## UNTRUNCATED TARGET (per-replicate raw-scan THR), clean rows ########")
    res_raw = evaluate(untr.reset_index(drop=True), "clean, untruncated target")

    # Data bug fix from the lead (~17:19): mcc_extras_fsri.csv total_hr_kJg integrated the MCC's end-of-run
    # baseline drift; total_hr_bc_kJg is the baseline-corrected value. Swap it in for h_c everywhere
    # (phys and FULL). Same rows, same variants: a corrected input, not a new variant.
    def with_bc(d):
        return d.assign(hc_kJg=d.hc_bc_kJg, phys_MJm2=d.mpa_flux_kgm2 * d.hc_bc_kJg,
                        phys_x_flux=d.mpa_flux_kgm2 * d.hc_bc_kJg * d.heat_flux_kWm2)
    print("\n\n######## BASELINE-CORRECTED MCC h_c (total_hr_bc_kJg), clean rows, untruncated target ########")
    res_bc = evaluate(with_bc(untr).reset_index(drop=True), "clean, untruncated, h_c corrected")
    core_ad = with_bc(df).dropna(subset=["phys_MJm2"]).reset_index(drop=True)
    print("\n[as-declared rows + thr_fsri target, h_c corrected]")
    zero_param(core_ad)
    run("as-declared h_c-corrected", core_ad, [("V1", LIN, ["phys_MJm2"])])
    print("\nh_c raw -> corrected, headline set, GKF mean: " +
          ", ".join(f"{k} {res_raw[k]['rkf_mean']:.4f} -> {res_bc[k]['rkf_mean']:.4f}" for k in res_raw))
