#!/usr/bin/env python3
"""Dravix V2 -- one row per FSRI cone replicate (Scalar + Scan pair), every outcome + QA flag in one table.

WHY (added 2026-09-29, stream cone_replicates): FSRI's averaged <Mat>_HRRPUA_<flux>.csv files are not
reliable targets. They put all replicates on one time axis, so a file can stop before its longest
replicate ends (THR too low) or keep a 'Mean' running on the surviving replicates (THR too high). They
average peaks that occur at different times, and they include replicates that should not be averaged
(e.g. Vinyl_Tile R4, pulled after 1-2 min). Round 1 had rebuilt per-replicate pieces in three separate
places. This script puts them in one table and IMPORTS the existing logic instead of copying it:
  HRR curve      ingest_fsri_figra.hrrpua_curve / peak_and_time (FSRI O2-consumption formula; within
                 0.03% of FSRI's own per-replicate curves)
  residue        ingest_fsri_cone_residue.ingest (FSRI 'Final Mass' definition + its QA gates)
  early stop     train_thr_v1.EARLY_STOP (regex over POST TEST CMT); TRUNC_FRAC (10% of peak)
  thickness      ingest_fsri_props.parse_thickness_mm (PRE TEST CMT)

Definitions:
  did_not_ignite  TIME TO IGN == 0 (FSRI's convention), or a POST/PRE TEST CMT saying there was no ignition
  tti_s           TIME TO IGN, NaN when did_not_ignite
  phrr_kWm2       raw 0.25 s max of the recomputed HRRPUA over [TTI, END OF TEST TIME] (== figra_fsri.csv)
  t_phrr_s        time of that peak from the start of exposure
  thr_MJm2        trapezoidal integral of HRRPUA over [TTI, END OF TEST TIME] / 1000.
                  For did_not_ignite rows TTI = 0, so pHRR/THR cover the whole test and are noise-level;
                  they are kept (measured), not blanked. Use the flag.
  figra_kWm2s     phrr / t_phrr, NaN when did_not_ignite (== figra_fsri.csv)
  residue_pct     cone_residue_pct from ingest_fsri_cone_residue (blank where its QA gates blank it)
  early_stop      EARLY_STOP matches POST TEST CMT (operator pulled or ended the test on purpose)
  truncated_test  not early_stop, and the test ended before the specimen was done:
                    ignited:        heat released in the 60 s AFTER END OF TEST TIME (the Scan keeps recording)
                                    > TRUNC_FRAC (10%) of the replicate's THR: still burning when the test was
                                    ended, so THR is a lower bound (the THR stream's tail60 measure, per replicate)
                    did_not_ignite: specimen lost > 1% of its mass AND > 0.5 g in the last 60 s before END OF
                                    TEST TIME (time cap hit while still pyrolysing, e.g. PMMA HF15-noSpark,
                                    20 min cap). The 0.5 g floor was added after the first run flagged House_Wrap
                                    (~1 g specimens) on load-cell noise; disclosed in the report.
                  A first rule, "HRR over the last 10 s > 10% of pHRR", flagged 459/690 ignited replicates (char
                  glow and low-peak fuels) and was dropped. thr_tail60_frac and mass_loss_last60s_g are kept so
                  the thresholds can be changed.
  specimen_mass_suspect  SPECIMEN MASS > 3x off its material's median (same rule as the residue gate;
                  e.g. PMMA HF25 R2 866.19 g). Mass itself is kept as recorded.

Also prints a cross-check of per-replicate means against phrr_fsri.csv, thr_fsri.csv, tti_fsri.csv,
figra_fsri.csv and figra_fsri_meancurve.csv (read only; those files are not modified).

Rerun: python3 data/v2/ingest_fsri_cone_replicates.py   (~1 min; writes data/v2/cone_replicates_fsri.csv)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from ingest_fsri import CITATION, RAW, SOURCE, parse_scalar, to_float
from ingest_fsri_cone_residue import ingest as ingest_residue
from ingest_fsri_cone_residue import scan_mass
from ingest_fsri_figra import hrrpua_curve, peak_and_time
from ingest_fsri_props import parse_thickness_mm
from train_thr_v1 import EARLY_STOP, TRUNC_FRAC

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NAME = re.compile(r"(.+)_Cone_(HF[\w+-]+)Scalar_(\d+)_R(\d+)\.csv")
NO_IGN = re.compile(r"no ignition|did not ignite|not ignite|no sustained ignition|never ignited", re.I)
TAIL_S = 60            # post-END OF TEST window for the "still burning" test
MASS_WINDOW_S = 60     # mass-loss window for the "still pyrolysing" test
MASS_LOSS_PCT = 1.0    # % of specimen mass lost in that window ...
MASS_LOSS_G = 0.5      # ... and at least this many grams (load-cell noise floor)


def replicate(scalar_path):
    material = os.path.basename(os.path.dirname(os.path.dirname(scalar_path)))
    _, tag, test_date, rep = NAME.match(os.path.basename(scalar_path)).groups()
    kv = parse_scalar(scalar_path)
    tti, eot = to_float(kv.get("TIME TO IGN")), to_float(kv.get("END OF TEST TIME"))
    area, m0 = to_float(kv.get("SURF AREA")), to_float(kv.get("SPECIMEN MASS"))
    scan_path = scalar_path.replace("Scalar", "Scan")
    t, q = hrrpua_curve(scan_path, to_float(kv.get("C FACTOR")), area)
    ok = np.isfinite(t) & np.isfinite(q)
    t, q = t[ok], q[ok]
    win = (t >= tti) & (t <= eot)
    phrr, t_pk = peak_and_time(t[win], q[win])
    after = (t >= eot) & (t <= eot + TAIL_S)
    thr = trapezoid(q[win], t[win]) / 1000
    mass, mtime = scan_mass(scan_path)
    m = pd.Series({mtime[k]: v for k, v in mass.items() if mtime.get(k) is not None and v is not None}).sort_index()
    m_end, m_before = m.loc[:eot].iloc[-1], m.loc[:max(eot - MASS_WINDOW_S, m.index[0])].iloc[-1]
    post, pre = kv.get("POST TEST CMT", ""), kv.get("PRE TEST CMT", "")
    dni = tti == 0 or bool(NO_IGN.search(post + " " + pre))
    return {
        "material": material, "heat_flux_kWm2": float(re.search(r"HF(\d+)", tag).group(1)), "hf_tag": tag,
        "replicate": int(rep), "test_date": test_date, "orientation": kv.get("ORIENTATION"),
        "spark_ignitor": "noSpark" not in tag, "frame": "Frame" in tag, "grid": "Grid" in tag,
        "specimen_mass_g": m0, "surf_area_m2": area, "thickness_mm": parse_thickness_mm(pre),
        "did_not_ignite": dni, "early_stop": bool(EARLY_STOP.search(post)),
        "tti_s": np.nan if dni else tti,
        "phrr_kWm2": round(phrr, 2), "t_phrr_s": t_pk,
        "thr_MJm2": round(thr, 3),
        "figra_kWm2s": np.nan if dni else round(phrr / t_pk, 4),
        "end_of_test_time_s": eot, "test_status": kv.get("TEST STATUS"),
        "thr_tail60_frac": round(trapezoid(q[after], t[after]) / 1000 / thr, 4) if after.sum() > 1 and thr > 0 else np.nan,
        "mass_loss_last60s_g": round(m_before - m_end, 3),
        "post_test_cmt": post,
    }


def build():
    df = pd.DataFrame([replicate(p) for p in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*Scalar*.csv")))
                       if NAME.match(os.path.basename(p)) and os.path.exists(p.replace("Scalar", "Scan"))])
    res = pd.DataFrame(ingest_residue())[["material", "test_config", "replicate", "test_date", "cone_residue_pct"]]
    res = res.rename(columns={"test_config": "hf_tag", "cone_residue_pct": "residue_pct"}).astype({"replicate": int})
    df = df.merge(res, on=["material", "hf_tag", "replicate", "test_date"], how="left", validate="1:1")
    med = df.groupby("material").specimen_mass_g.transform("median")
    df["specimen_mass_suspect"] = ~df.specimen_mass_g.between(med / 3, med * 3)
    burning = ~df.did_not_ignite & (df.thr_tail60_frac > TRUNC_FRAC)
    pyrolysing = df.did_not_ignite & (df.mass_loss_last60s_g > np.maximum(MASS_LOSS_PCT / 100 * df.specimen_mass_g, MASS_LOSS_G))
    df["truncated_test"] = ~df.early_stop & (burning | pyrolysing)
    df["method"] = ("raw Scan -> FSRI O2-consumption HRRPUA / SURF AREA; pHRR max and THR integral over "
                    "[TTI, END OF TEST TIME]; residue per ingest_fsri_cone_residue")
    df["source"], df["citation"] = SOURCE, CITATION
    cols = ["material", "heat_flux_kWm2", "hf_tag", "replicate", "test_date", "orientation", "spark_ignitor",
            "frame", "grid", "specimen_mass_g", "surf_area_m2", "thickness_mm", "did_not_ignite", "early_stop",
            "truncated_test", "specimen_mass_suspect", "tti_s", "phrr_kWm2", "t_phrr_s", "thr_MJm2", "residue_pct",
            "figra_kWm2s", "end_of_test_time_s", "test_status", "thr_tail60_frac", "mass_loss_last60s_g",
            "post_test_cmt", "method", "source", "citation"]
    return df[cols]


def self_check(df):
    """Expectations stated by the lead before this script existed (round 1 numbers)."""
    pm = df[(df.material == "PMMA") & (df.hf_tag == "HF50")]
    assert len(pm) == 3 and pm.phrr_kWm2.between(1155, 1250).all(), pm.phrr_kWm2.tolist()
    assert abs(pm.thr_MJm2.mean() - 90) < 5, pm.thr_MJm2.tolist()
    hd = df[(df.material == "HDPE") & (df.hf_tag == "HF50")].phrr_kWm2
    assert any(abs(hd - 2712) < 15) and any(abs(hd - 2731) < 15), hd.tolist()
    assert df.did_not_ignite.sum() == 45 and (df.did_not_ignite == (df.tti_s.isna())).all()
    g = df.groupby(["material", "heat_flux_kWm2"]).early_stop.agg(["all", "sum"])
    all_es = sorted(g[g["all"]].index)
    assert all_es == [(m, f) for m in ("Face_Shield", "High_Temperature_SCBA_Facepiece") for f in (50., 75., 100.)], all_es
    # identical to the per-replicate figra_fsri.csv (same imported HRR code), row for row
    fg = pd.read_csv(os.path.join(DATA_DIR, "figra_fsri.csv"), dtype={"test_date": str})
    j = df.merge(fg, on=["material", "hf_tag", "replicate", "test_date"], validate="1:1")
    assert len(j) == len(df) == len(fg)
    assert (j.phrr_kWm2 - j.peak_hrrpua_kWm2).abs().max() < 0.01
    assert np.allclose(j.figra_kWm2s_x, j.figra_kWm2s_y, equal_nan=True)
    print(f"[self-check] OK: PMMA HF50 pHRR {pm.phrr_kWm2.tolist()} THR {pm.thr_MJm2.round(2).tolist()} "
          f"(mean {pm.thr_MJm2.mean():.2f}); HDPE HF50 pHRR {sorted(hd.tolist())}; did_not_ignite "
          f"{df.did_not_ignite.sum()}; early_stop {df.early_stop.sum()} replicates, {len(all_es)} material x flux "
          f"groups where every replicate is early_stop; pHRR == figra_fsri.csv on all {len(j)} rows")
    return pm, hd


def averaged_files():
    """Replicate numbers each FSRI averaged file lists + where its time axis ends."""
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*_HRRPUA_*.csv"))):
        f = pd.read_csv(p)
        out.append({"material": os.path.basename(os.path.dirname(os.path.dirname(p))),
                    "heat_flux_kWm2": float(re.search(r"_HRRPUA_(\d+)", p).group(1)),
                    "file_reps": [int(c.rsplit("_R", 1)[1]) for c in f.columns if re.search(r"_R\d+$", c)],
                    "curve_end_s": float(f.iloc[:, 0].iloc[-1])})
    return pd.DataFrame(out)


def cross_check(df):
    """Per material x flux: mean over the replicates each averaged file lists vs the round-1 tables."""
    av = averaged_files()
    lookup = (df[df.spark_ignitor].assign(plain=df.hf_tag.str.fullmatch(r"HF\d+"))
              .sort_values("plain", ascending=False).drop_duplicates(["material", "heat_flux_kWm2", "replicate"]))
    rows = []
    for a in av.itertuples():
        r = lookup[(lookup.material == a.material) & (lookup.heat_flux_kWm2 == a.heat_flux_kWm2)
                   & lookup.replicate.isin(a.file_reps)]
        dur = r.end_of_test_time_s - r.tti_s.fillna(0)
        rows.append({"material": a.material, "heat_flux_kWm2": a.heat_flux_kWm2, "n_file": len(a.file_reps),
                     "n_matched": len(r), "phrr_rep": r.phrr_kWm2.mean(), "thr_rep": r.thr_MJm2.mean(),
                     "tti_rep": r.tti_s.mean(), "figra_rep": r.figra_kWm2s.mean(),
                     "curve_end_s": a.curve_end_s, "dur_min": dur.min(), "dur_max": dur.max(),
                     "has_early_stop": r.early_stop.any(), "has_dni": r.did_not_ignite.any(),
                     "has_truncated": r.truncated_test.any()})
    c = pd.DataFrame(rows)
    key = ["material", "heat_flux_kWm2"]
    c = (c.merge(pd.read_csv(os.path.join(DATA_DIR, "phrr_fsri.csv"))[key + ["peak_hrrpua_kWm2"]], on=key)
          .merge(pd.read_csv(os.path.join(DATA_DIR, "thr_fsri.csv"))[key + ["total_heat_release_MJm2"]], on=key)
          .merge(pd.read_csv(os.path.join(DATA_DIR, "figra_fsri_meancurve.csv"))[key + ["figra_kWm2s"]]
                 .rename(columns={"figra_kWm2s": "figra_meancurve"}), on=key))
    c["cut_short"] = c.curve_end_s < c.dur_max - 5          # file ends >5 s before its longest replicate
    c["runs_on"] = c.curve_end_s > c.dur_min + 5            # 'Mean' continues past its shortest replicate

    def material_diff(rep, old, floor):
        d = rep - old
        return (d.abs() > floor) & (d.abs() / old.abs().clip(lower=1e-9) > 0.05), d

    def reason(x, d):
        if x.has_early_stop:
            return "early-stopped replicate averaged in"
        if x.cut_short and d < 0:
            return "averaged file cut before longest replicate ends"
        if x.runs_on and d < 0:
            return "Mean runs on past shortest replicate (survivors only)"
        if x.cut_short:
            return "averaged file cut before longest replicate ends"
        if x.n_matched != x.n_file:
            return "replicate set mismatch"
        return "peak of mean < mean of peaks (replicate peaks at different times)"

    out = {}
    for name, rep, old, floor in [("pHRR kW/m2", "phrr_rep", "peak_hrrpua_kWm2", 20.0),
                                  ("THR MJ/m2", "thr_rep", "total_heat_release_MJm2", 2.0),
                                  ("FIGRA kW/m2/s (vs mean curve)", "figra_rep", "figra_meancurve", 0.5)]:
        ok = c[rep].notna() & c[old].notna()
        flag, d = material_diff(c.loc[ok, rep], c.loc[ok, old], floor)
        rel = (d / c.loc[ok, old].abs().clip(lower=1e-9))
        reasons = c.loc[ok][flag].apply(lambda x: reason(x, x[rep] - x[old]), axis=1).value_counts()
        out[name] = (int(ok.sum()), int(flag.sum()), float(rel.median()), reasons)
        print(f"[cross] {name}: {ok.sum()} rows compared, replicate-mean minus old table: median rel "
              f"{rel.median():+.3f}; differ materially (>5% and >{floor:g}) on {flag.sum()} rows, "
              f"{(flag & (d > 0)).sum()} higher / {(flag & (d < 0)).sum()} lower")
        for k, v in reasons.items():
            print(f"[cross]     {v:3d}  {k}")
    # TTI: tti_fsri.csv is per replicate; compare row coverage and the value itself
    tt = pd.read_csv(os.path.join(DATA_DIR, "tti_fsri.csv"), dtype={"test_date": str})
    j = df.merge(tt, on=["material", "test_date", "replicate", "heat_flux_kWm2", "spark_ignitor"], how="left")
    j = j.drop_duplicates(["material", "hf_tag", "replicate", "test_date"])
    miss = j.time_to_ignition_s.isna()
    same = (j.tti_s == j.time_to_ignition_s) | (j.did_not_ignite & (j.time_to_ignition_s == 0))
    miss_tags = j[miss].hf_tag.str.replace(r"HF\d+", "HF*", regex=True).value_counts().to_dict()
    print(f"[cross] TTI: {len(df)} replicates here vs {len(tt)} rows in tti_fsri.csv; {miss.sum()} missing there "
          f"({miss_tags}); on the other "
          f"{(~miss).sum()} the value is identical ({(same & ~miss).sum()}), except that tti_fsri.csv stores the "
          f"{int((j.time_to_ignition_s == 0).sum())} non-ignitions as 0 s where this table has NaN + did_not_ignite")
    fm = c[c.tti_rep.notna()]
    print(f"[cross] TTI groups: {len(c)} material x flux; replicate-mean TTI undefined (no replicate ignited) on "
          f"{c.tti_rep.isna().sum()}; mixed ignite/no-ignite groups {int((c.has_dni & c.tti_rep.notna()).sum())}")
    print(f"[cross] averaged files: {int(c.cut_short.sum())} end >5 s before their longest replicate, "
          f"{int(c.runs_on.sum())} run >5 s past their shortest; {int((c.n_matched != c.n_file).sum())} list a "
          f"replicate with no Scalar/Scan pair; {int(c.has_early_stop.sum())} average in an early-stopped replicate")
    return c, out


if __name__ == "__main__":
    df = build()
    self_check(df)  # before writing: a failed check must not overwrite the CSV
    df.to_csv(os.path.join(DATA_DIR, "cone_replicates_fsri.csv"), index=False)
    print(f"{len(df)} replicates / {df.material.nunique()} materials -> cone_replicates_fsri.csv; "
          f"did_not_ignite {df.did_not_ignite.sum()}, early_stop {df.early_stop.sum()}, truncated_test "
          f"{df.truncated_test.sum()} (still burning {(df.truncated_test & ~df.did_not_ignite).sum()}, "
          f"time-capped while pyrolysing {(df.truncated_test & df.did_not_ignite).sum()}), "
          f"specimen_mass_suspect {df.specimen_mass_suspect.sum()}, residue blank {df.residue_pct.isna().sum()}, "
          f"thickness blank {df.thickness_mm.isna().sum()}")
    tr = df[df.truncated_test]
    print("[truncated] by material:", tr.groupby("material").size().sort_values(ascending=False).to_dict())
    print(df[df.truncated_test].sort_values("thr_tail60_frac", ascending=False)
          [["material", "hf_tag", "replicate", "did_not_ignite", "thr_MJm2", "thr_tail60_frac", "mass_loss_last60s_g",
            "specimen_mass_g", "residue_pct", "end_of_test_time_s"]].to_string(index=False))
    c, _ = cross_check(df)
    if os.environ.get("CROSS_OUT"):  # optional: dump the per material x flux cross-check table
        c.to_csv(os.environ["CROSS_OUT"], index=False)
