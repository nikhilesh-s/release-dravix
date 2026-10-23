#!/usr/bin/env python3
"""Ingest FSRI material properties and full-curve MCC quantities as model features.

Same source/citation as ingest_fsri.py. None of these are fire-test outcomes.
They are material descriptors that are physically upstream of TTI, pHRR, THR,
and FIGRA, which earlier feature-importance runs showed were the weak link
(bulk MCC peak HRR + char yield as proxies for everything):

- material_props_fsri.csv (one row per material):
  - density_kgm3: from each material's own material.json ("--" -> blank).
  - categories: "material category" list from material.json, "; "-joined. This
    enables a family-grouped split, which is harder than material-grouped.
  - cone_thickness_mm: median specimen thickness parsed from the cone Scalar
    files' PRE TEST CMT field ("Dimensions: L x W x T mm", "Thickness: X mm",
    "Nominal thickness: 5/8 inches", "5 cm depth tray"). Blank if unstated.
  - cone_mass_per_area_kgm2: median SPECIMEN MASS / SURF AREA over the
    material's cone replicates.
- mcc_extras_fsri.csv (one row per MCC replicate): quantities that
  mcc_hrr.csv doesn't keep, read off the same full HRR(t)/T(t) curve:
  - total_hr_kJg: integral of HRR over time. The standard MCC "total heat
    release" (heat of complete combustion of the volatiles per g of sample).
  - hrc_JgK: peak HRR / heating rate. Lyon's heat release capacity.
  - t_peak_C: temperature at peak HRR.
  - t_onset_C / t_end_C: temperatures at which 5% / 95% of total heat is
    released. This gives an onset estimate that isn't fooled by early moisture
    loss the way a 5%-mass-loss onset is.
  Onset/end are blank when total_hr_kJg < 0.5 kJ/g (effectively inert
  samples, where the fraction of heat released is just noise).
  - total_hr_bc_kJg, t_onset_bc_C, t_end_bc_C: the same three quantities after
    subtracting a linear baseline drawn between the median HRR of the first
    and last 5% of the run. Use these. The raw integral picks up end-of-run
    baseline drift, which runs from -9 to +22 W/g over a ~1,300 s run: Black_PMMA's
    three replicates read 18.6 / 39.9 / 33.1 kJ/g raw vs 23.2 / 26.5 / 25.7
    corrected (literature ~25). The drift also dragged the raw onset early.
    The raw columns stay only so the 2026-09-29 sprint results reproduce.
  FIXED 2026-09-29 (reports/sprint_0929/mcc_fix.md): MCC files now come from
  ingest_fsri_sta_mcc.mcc_replicate_files(), which also accepts the 9 materials
  whose file names lack '_MCC_' (118 -> 127 materials) and excludes FSRI's
  '_O<n>' outlier replicates explicitly. heating_rate_C_s here was always the
  header's C/s value.
"""
import csv
import glob
import json
import os
import re
from statistics import median

import numpy as np

from ingest_fsri_sta_mcc import CITATION, RAW, SOURCE, mcc_replicate_files, to_float, write_csv

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_thickness_mm(comment):
    c = comment or ""
    m = re.search(r"Dimensions:\s*[\d.]+\s*x\s*[\d.]+\s*x\s*([\d.]+)\s*mm", c)
    if m:
        return float(m.group(1))
    m = re.search(r"[Tt]hickness:\s*([\d.]+)\s*mm", c)
    if m:
        return float(m.group(1))
    m = re.search(r"[Tt]hickness:\s*(\d+)/(\d+)\s*inch", c)
    if m:
        return 25.4 * int(m.group(1)) / int(m.group(2))
    m = re.search(r"[Tt]hickness:\s*([\d.]+)\s*inch", c)
    if m:
        return 25.4 * float(m.group(1))
    m = re.search(r"([\d.]+)\s*cm depth tray", c)
    if m:
        return 10 * float(m.group(1))
    return None


def read_scalar(path):
    kv = {}
    with open(path, newline="", encoding="latin-1", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                kv[row[0].strip()] = row[1].strip()
    return kv


def ingest_material_props():
    rows = []
    for mat_dir in sorted(glob.glob(os.path.join(RAW, "*", ""))):
        material = os.path.basename(os.path.dirname(mat_dir))
        meta_path = os.path.join(mat_dir, "material.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8", errors="replace") as f:
                meta = json.load(f, strict=False)
        dm = re.search(r"([\d.]+)\s*kg/m", str(meta.get("density", "")))

        thicknesses, mass_per_area = [], []
        for scalar in glob.glob(os.path.join(mat_dir, "Cone", "*Scalar*.csv")):
            kv = read_scalar(scalar)
            t = parse_thickness_mm(kv.get("PRE TEST CMT"))
            if t is not None:
                thicknesses.append(t)
            mass, area = to_float(kv.get("SPECIMEN MASS")), to_float(kv.get("SURF AREA"))
            if mass and area:
                mass_per_area.append(mass / area / 1000.0)

        rows.append({
            "material": material,
            "material_name": meta.get("material", ""),
            "categories": "; ".join(meta.get("material category", [])),
            "density_kgm3": float(dm.group(1)) if dm else None,
            "cone_thickness_mm": round(median(thicknesses), 3) if thicknesses else None,
            "n_thickness_readings": len(thicknesses),
            "cone_mass_per_area_kgm2": round(median(mass_per_area), 4) if mass_per_area else None,
            "n_cone_replicates": len(mass_per_area),
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


def ingest_mcc_extras():
    rows = []
    for path, material, test_date, replicate in mcc_replicate_files()[0]:
        with open(path, newline="", encoding="latin-1", errors="replace") as f:
            lines = f.readlines()
        head = {}
        start = None
        for i, line in enumerate(lines):
            if line.strip() == "*":
                start = i + 1
                break
            if ":" in line:
                k, _, v = line.partition(":")
                head[k.strip()] = v.strip()
        if start is None:
            continue
        cols = lines[start].strip().split("\t")
        try:
            it, iT, ih = cols.index("Time (s)"), cols.index("Temperature (C)"), cols.index("HRR (W/g)")
        except ValueError:
            continue
        data = []
        for line in lines[start + 1:]:
            parts = line.strip().split("\t")
            if len(parts) > max(it, iT, ih):
                vals = [to_float(parts[j]) for j in (it, iT, ih)]
                if None not in vals:
                    data.append(vals)
        rate = to_float(head.get("Heating Rate (C/s)"))
        if len(data) < 10 or not rate:
            continue
        t, T, q = np.array(data).T
        ipk = int(np.argmax(q))
        cum = np.concatenate([[0.0], np.cumsum(np.diff(t) * (q[1:] + q[:-1]) / 2)])  # J/g
        total = cum[-1] / 1000.0
        onset = end = None
        if total >= 0.5:
            onset = float(T[np.argmax(cum >= 0.05 * cum[-1])])
            end = float(T[np.argmax(cum >= 0.95 * cum[-1])])
        edge = max(len(q) // 20, 1)
        qb = q - np.interp(t, [t[0], t[-1]], [np.median(q[:edge]), np.median(q[-edge:])])
        cum_bc = np.concatenate([[0.0], np.cumsum(np.diff(t) * (qb[1:] + qb[:-1]) / 2)])
        total_bc = cum_bc[-1] / 1000.0
        onset_bc = end_bc = None
        if total_bc >= 0.5:
            onset_bc = float(T[np.argmax(cum_bc >= 0.05 * cum_bc[-1])])
            end_bc = float(T[np.argmax(cum_bc >= 0.95 * cum_bc[-1])])

        rows.append({
            "material": material,
            "replicate": replicate,
            "test_date": test_date,
            "heating_rate_C_s": rate,
            "sample_weight_mg": to_float(head.get("Sample Weight (mg)")),
            "peak_hrr_Wg": round(float(q[ipk]), 3),
            "hrc_JgK": round(float(q[ipk]) / rate, 2),
            "total_hr_kJg": round(total, 3),
            "t_peak_C": round(float(T[ipk]), 1),
            "t_onset_C": round(onset, 1) if onset is not None else None,
            "t_end_C": round(end, 1) if end is not None else None,
            "total_hr_bc_kJg": round(total_bc, 3),
            "t_onset_bc_C": round(onset_bc, 1) if onset_bc is not None else None,
            "t_end_bc_C": round(end_bc, 1) if end_bc is not None else None,
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


if __name__ == "__main__":
    props = ingest_material_props()
    mcc = ingest_mcc_extras()
    write_csv(os.path.join(OUT_DIR, "material_props_fsri.csv"), props, list(props[0]))
    write_csv(os.path.join(OUT_DIR, "mcc_extras_fsri.csv"), mcc, list(mcc[0]))

    def n(col):
        return sum(r[col] is not None for r in props)

    print(f"material_props_fsri.csv: {len(props)} materials; density {n('density_kgm3')}, "
          f"cone thickness {n('cone_thickness_mm')}, mass/area {n('cone_mass_per_area_kgm2')}")
    print(f"mcc_extras_fsri.csv:     {len(mcc)} replicate rows across {len(set(r['material'] for r in mcc))} materials")

    # self-check: baseline-corrected heat of combustion of well-known polymers (Lyon & Walters ranges)
    hc = {}
    for r in mcc:
        hc.setdefault(r["material"], []).append(r["total_hr_bc_kJg"])
    assert "Vinyl_Siding" in hc  # named without '_MCC_': skipped before the 2026-09-29 fix
    for mat, lo, hi in [("Black_PMMA", 22, 28), ("HDPE", 38, 46), ("ABS", 33, 40)]:
        v = float(np.mean(hc[mat]))
        assert lo <= v <= hi, (mat, v)
    print("self-check OK: corrected total heat release for Black_PMMA, HDPE, ABS in literature ranges")
