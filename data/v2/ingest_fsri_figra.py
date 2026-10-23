#!/usr/bin/env python3
"""Ingest FSRI cone calorimeter curves into a Fire Growth Rate (FIGRA) target table.

Definition, chosen to match the competitor (phan_morgan_2025, arXiv:2504.00223,
"Fire testing" section just before Table 2): fire growth rate is peak HRR divided by
the time to peak HRR.
    FIGRA = pHRR / t_pHRR    [kW/m2/s]
The paper does not say which clock t_pHRR is on. We use time from the start of the
test (specimen exposure, the same clock as TIME TO IGN), the usual cone convention.
All 15 of their Table 2 rows have pHRR/FIGRA > TIG, which fits that clock but does not
prove it. time_ign_to_peak_s is kept so the from-ignition variant can be rebuilt.

Why the raw Scan files and not FSRI's averaged <Mat>_HRRPUA_<flux>.csv alone: the
averaged files put every replicate on the FIRST replicate's time index, so a longer
replicate is cut short (up to ~1,300 s) and a late peak can be lost. The Scan files hold
raw instrument channels, not HRR, so HRR is recomputed here with FSRI's own
oxygen-consumption formula (02_Scripts/plot_Cone_data.py: O2-only depletion factor,
E = 13,100 kJ/kg O2, x1.10), divided by the Scalar file's SURF AREA. Checked on
2026-09-29: that reproduces FSRI's published per-replicate curves to within 0.03% on all
665 replicates found in both places.

pHRR is the raw 0.25 s maximum between ignition and END OF TEST TIME, with no smoothing
(same as FSRI's "Peak HRRPUA" and ASTM E1354 practice). Starting the clock at exposure
makes t_pHRR >= TTI, so an ignition spike cannot shrink it toward zero. A 5 s centred
rolling-mean variant is kept as a sensitivity column; it is not the target.

Outputs:
  figra_fsri.csv            one row per cone replicate (full-length raw Scan)
  figra_fsri_meancurve.csv  one row per material x heat flux from FSRI's averaged curve
                            (same peak as phrr_fsri.csv) + that curve's replicates' mean TTI
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from ingest_fsri import CITATION, RAW, SOURCE, parse_scalar, to_float

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SMOOTH_POINTS = 21  # 5 s at the 0.25 s scan interval, sensitivity column only


def hrrpua_curve(scan_path, c_factor, surf_area_m2):
    """(time_s, HRRPUA kW/m2), as in FSRI plot_Cone_data.py, area from the Scalar file."""
    d = pd.read_csv(scan_path, header=0, skiprows=[1, 2, 3, 4], index_col="Names")
    o2 = d["O2 Meter"] / 100
    edf = np.sqrt(d["Exh Press"] / (d["Stack TC"] + 273.15)) * c_factor  # exhaust mass flow
    odf = (o2.loc["Baseline"] - o2) / (1.105 - 1.5 * o2)  # oxygen depletion factor
    hrr = (1.10 * 13100 * edf * odf / surf_area_m2).drop("Baseline")
    return d["Time"].drop("Baseline").astype(float).to_numpy(), hrr.to_numpy()


def peak_and_time(t, v):
    i = np.nanargmax(v)
    return float(v[i]), float(t[i])


def ingest_replicates():
    rows = []
    for scalar_path in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*Scalar*.csv"))):
        m = re.match(r"(.+)_Cone_(HF[\w+-]+)Scalar_(\d+)_R(\d+)\.csv", os.path.basename(scalar_path))
        if not m:
            continue
        _, hf_tag, test_date, replicate = m.groups()
        kv = parse_scalar(scalar_path)
        tti, eot = to_float(kv.get("TIME TO IGN")), to_float(kv.get("END OF TEST TIME"))
        area, c_factor = to_float(kv.get("SURF AREA")), to_float(kv.get("C FACTOR"))
        scan_path = scalar_path.replace("Scalar", "Scan")
        if None in (tti, eot, area, c_factor) or not os.path.exists(scan_path) or tti >= eot:
            continue
        t, hrr = hrrpua_curve(scan_path, c_factor, area)
        win = (t >= tti) & (t <= eot)  # ignition to end of test
        if win.sum() < 2:
            continue
        phrr, t_peak = peak_and_time(t[win], hrr[win])
        smooth = pd.Series(hrr).rolling(SMOOTH_POINTS, center=True, min_periods=1).mean().to_numpy()
        phrr_s, t_peak_s = peak_and_time(t[win], smooth[win])
        # TIME TO IGN = 0 is FSRI's "no sustained ignition" (fiberglass, mineral wool, PMMA at 15 kW/m2
        # no spark, pHRR at noise level 5-12 kW/m2): FIGRA is undefined there, not huge.
        ignited = tti > 0
        rows.append({
            "material": os.path.basename(os.path.dirname(os.path.dirname(scalar_path))),
            "heat_flux_kWm2": float(re.search(r"HF(\d+)", hf_tag).group(1)),
            "hf_tag": hf_tag,
            "spark_ignitor": "noSpark" not in hf_tag,
            "replicate": int(replicate),
            "test_date": test_date,
            "time_to_ignition_s": tti,
            "peak_hrrpua_kWm2": round(phrr, 2),
            "time_to_peak_s": t_peak,
            "time_ign_to_peak_s": t_peak - tti,
            "ignition_recorded": ignited,
            "figra_kWm2s": round(phrr / t_peak, 4) if ignited else np.nan,
            "figra_smoothed5s_kWm2s": round(phrr_s / t_peak_s, 4) if ignited else np.nan,
            "end_of_test_time_s": eot,
            "surf_area_m2": area,
            "test_status": kv.get("TEST STATUS"),
            "method": "raw Scan -> FSRI O2-consumption HRR / SURF AREA; max over [TTI, END OF TEST]",
            "source": SOURCE,
            "citation": CITATION,
        })
    return pd.DataFrame(rows)


def ingest_mean_curves(reps):
    """FSRI's own averaged curves; t_pHRR = time after ignition of the Mean peak + mean replicate TTI."""
    # one replicate row per (material, flux, replicate): prefer the spark-ignited plain HF tag
    lookup = (reps[reps.spark_ignitor]
              .assign(plain=reps.hf_tag.str.fullmatch(r"HF\d+"))
              .sort_values("plain", ascending=False)
              .drop_duplicates(["material", "heat_flux_kWm2", "replicate"])
              .set_index(["material", "heat_flux_kWm2", "replicate"]))
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*_HRRPUA_*.csv"))):
        material = os.path.basename(os.path.dirname(os.path.dirname(path)))
        flux = float(re.search(r"_HRRPUA_(\d+)", os.path.basename(path)).group(1))
        f = pd.read_csv(path)
        t, mean = f.iloc[:, 0].to_numpy(float), f["Mean"].to_numpy(float)
        phrr, t_after = peak_and_time(t, mean)
        rep_ids = [int(c.rsplit("_R", 1)[1]) for c in f.columns if re.search(r"_R\d+$", c)]
        keys = [(material, flux, r) for r in rep_ids if (material, flux, r) in lookup.index]
        matched = lookup.loc[keys] if keys else lookup.iloc[:0]
        mean_tti = matched.time_to_ignition_s[matched.ignition_recorded].mean()  # NaN if none ignited
        full_len = matched.end_of_test_time_s - matched.time_to_ignition_s
        rows.append({
            "material": material,
            "heat_flux_kWm2": flux,
            "n_replicates": len(rep_ids),
            "n_replicates_matched": len(matched),
            "n_replicates_ignited": int(matched.ignition_recorded.sum()),
            "peak_hrrpua_kWm2": round(phrr, 2),
            "time_ign_to_peak_s": t_after,
            "mean_time_to_ignition_s": round(mean_tti, 3),
            "time_to_peak_s": round(t_after + mean_tti, 3),
            "figra_kWm2s": round(phrr / (t_after + mean_tti), 4),  # NaN when no replicate ignited
            "curve_end_s": t[-1],
            "replicates_truncated": int((full_len > t[-1] + 1).sum()),
            "method": "FSRI averaged <Mat>_HRRPUA_<flux>.csv Mean column; t from ignition + mean replicate TTI",
            "source": SOURCE,
            "citation": CITATION,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    reps = ingest_replicates()
    means = ingest_mean_curves(reps)
    reps.to_csv(os.path.join(DATA_DIR, "figra_fsri.csv"), index=False)
    means.to_csv(os.path.join(DATA_DIR, "figra_fsri_meancurve.csv"), index=False)

    print(f"Replicates: {len(reps)} rows / {reps.material.nunique()} materials -> figra_fsri.csv "
          f"(FIGRA undefined for {reps.figra_kWm2s.isna().sum()})")
    print(f"Mean curves: {len(means)} material x flux rows -> figra_fsri_meancurve.csv; "
          f"{(means.replicates_truncated > 0).sum()} files cut short at least one replicate")

    # sanity: PMMA at 50 kW/m2 (log value for pHRR is ~1,155 kW/m2)
    print(reps.query("material == 'PMMA' and heat_flux_kWm2 == 50")
          [["replicate", "time_to_ignition_s", "peak_hrrpua_kWm2", "time_to_peak_s", "figra_kWm2s"]].to_string(index=False))
    print(means.query("material == 'PMMA'")[["heat_flux_kWm2", "peak_hrrpua_kWm2", "time_to_peak_s", "figra_kWm2s"]]
          .to_string(index=False))

    # cross-check: replicate-mean FIGRA vs FSRI mean-curve FIGRA (spark-ignited replicates only)
    agg = reps[reps.spark_ignitor].groupby(["material", "heat_flux_kWm2"]).figra_kWm2s.mean().rename("rep_mean")
    both = means.set_index(["material", "heat_flux_kWm2"]).join(agg, how="inner").dropna(subset=["figra_kWm2s", "rep_mean"])
    ratio = both.figra_kWm2s / both.rep_mean
    print(f"Mean-curve vs replicate-mean FIGRA on {len(both)} rows: Spearman "
          f"{both.figra_kWm2s.corr(both.rep_mean, method='spearman'):.3f}, ratio median {ratio.median():.3f} "
          f"[5-95%: {ratio.quantile(.05):.3f}-{ratio.quantile(.95):.3f}]")
    sm = (reps.figra_smoothed5s_kWm2s / reps.figra_kWm2s).dropna()
    print(f"5 s smoothing sensitivity (replicates): FIGRA ratio median {sm.median():.3f}, "
          f"{(abs(sm - 1) > 0.2).sum()} of {len(sm)} replicates move >20%")
    ign = reps[reps.ignition_recorded]
    print(f"Ignited replicates: t_pHRR from test start min {ign.time_to_peak_s.min()} s, "
          f"{(ign.time_to_peak_s < 10).sum()} under 10 s (thin fuels: window screen, cotton/hemp sheet, paper facings)")
    assert reps.time_to_peak_s.ge(reps.time_to_ignition_s).all()  # clock sanity
