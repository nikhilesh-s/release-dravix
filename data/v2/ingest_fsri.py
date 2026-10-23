#!/usr/bin/env python3
"""Ingest FSRI Materials and Products Database cone calorimeter data into
Dravix V2's Tier-2 target tables (TTI, pHRR, THR).

Source: https://github.com/ulfsri/fsri_materials_database (McKinnon, Weinschenk,
Dow, DiDomizio, Madrzykowski — FSRI/UL Research Institutes, 2023). Cite per
their CITATION.cff on any use.

TTI comes straight from each replicate's Scalar file (`TIME TO IGN`), no
derivation. pHRR and THR come from the repo's own derived per-heat-flux
HRRPUA mean-curve files (already averaged across replicates by FSRI's own
processing scripts) — pHRR is the curve's max, THR is its trapezoidal
integral to end-of-test time. Nothing here is synthetic or imputed; every
row traces to a named material + heat-flux + replicate set in the source repo.
"""
import csv
import glob
import os
import re

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "fsri_materials_database", "01_Data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = "fsri_materials_database"
CITATION = ("McKinnon, M., Weinschenk, C., Dow, N., DiDomizio, M., and Madrzykowski, D. "
            "Materials and Products Database (Version 1.0.0), Fire Safety Research "
            "Institute, UL Research Institutes. Columbia, MD 21045, 2023.")


def parse_scalar(path):
    kv = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                kv[row[0].strip()] = row[1].strip()
    return kv


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ingest_tti():
    rows = []
    for scalar_path in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*Scalar*.csv"))):
        fname = os.path.basename(scalar_path)
        material_dir = os.path.basename(os.path.dirname(os.path.dirname(scalar_path)))
        m = re.match(r"(.+)_Cone_(HF[\w-]+)Scalar_(\d+)_R(\d+)\.csv", fname)
        if not m:
            continue
        _, hf_tag, test_date, replicate = m.groups()
        hf_match = re.search(r"HF(\d+)", hf_tag)
        heat_flux = to_float(hf_match.group(1)) if hf_match else None
        no_spark = "noSpark" in hf_tag
        kv = parse_scalar(scalar_path)
        tti = to_float(kv.get("TIME TO IGN"))
        if tti is None or heat_flux is None:
            continue
        rows.append({
            "material": material_dir,
            "material_label": kv.get("MATERIAL", material_dir),
            "heat_flux_kWm2": heat_flux,
            "spark_ignitor": not no_spark,
            "replicate": replicate,
            "test_date": test_date,
            "time_to_ignition_s": tti,
            "orientation": kv.get("ORIENTATION"),
            "specimen_mass_g": to_float(kv.get("SPECIMEN MASS")),
            "end_of_test_time_s": to_float(kv.get("END OF TEST TIME")),
            "test_status": kv.get("TEST STATUS"),
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


def ingest_hrrpua():
    phrr_rows, thr_rows = [], []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*_HRRPUA_*.csv"))):
        fname = os.path.basename(path)
        material_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = re.match(r"(.+)_HRRPUA_([\w-]+)\.csv", fname)
        if not m:
            continue
        _, hf_tag = m.groups()
        hf_match = re.search(r"(\d+)", hf_tag)
        heat_flux = to_float(hf_match.group(1)) if hf_match else None
        if heat_flux is None:
            continue
        times, means = [], []
        n_replicates = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            n_replicates = sum(1 for h in header if re.match(r".+_R\d+$", h.strip()))
            mean_idx = None
            for i, h in enumerate(header):
                if h.strip().lower() == "mean":
                    mean_idx = i
                    break
            time_idx = 0
            if mean_idx is None:
                continue
            for row in reader:
                if len(row) <= max(mean_idx, time_idx):
                    continue
                t = to_float(row[time_idx])
                v = to_float(row[mean_idx])
                if t is None or v is None:
                    continue
                times.append(t)
                means.append(v)
        if not times:
            continue
        peak = max(means)
        peak_t = times[means.index(peak)]
        # trapezoidal integration of mean HRRPUA (kW/m2) over time (s) -> kJ/m2, then MJ/m2
        # (kW/m2 * s = kJ/m2; "Time after Ignition" in this repo's derived
        # curves is in seconds, confirmed against each test's own END OF TEST TIME)
        thr = 0.0
        for i in range(1, len(times)):
            dt_s = times[i] - times[i - 1]
            if dt_s <= 0:
                continue
            thr += 0.5 * (means[i] + means[i - 1]) * dt_s / 1000.0
        common = {
            "material": material_dir,
            "heat_flux_kWm2": heat_flux,
            "n_replicates": n_replicates,
            "curve_points": len(times),
            "source": SOURCE,
            "citation": CITATION,
        }
        phrr_rows.append({**common, "peak_hrrpua_kWm2": round(peak, 2),
                           "time_of_peak_s": peak_t})
        thr_rows.append({**common, "total_heat_release_MJm2": round(thr, 2),
                          "integrated_to_time_s": times[-1]})
    return phrr_rows, thr_rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    tti_rows = ingest_tti()
    phrr_rows, thr_rows = ingest_hrrpua()

    write_csv(os.path.join(OUT_DIR, "tti_fsri.csv"), tti_rows,
              ["material", "material_label", "heat_flux_kWm2", "spark_ignitor",
               "replicate", "test_date", "time_to_ignition_s", "orientation",
               "specimen_mass_g", "end_of_test_time_s", "test_status", "source", "citation"])
    write_csv(os.path.join(OUT_DIR, "phrr_fsri.csv"), phrr_rows,
              ["material", "heat_flux_kWm2", "n_replicates", "curve_points",
               "peak_hrrpua_kWm2", "time_of_peak_s", "source", "citation"])
    write_csv(os.path.join(OUT_DIR, "thr_fsri.csv"), thr_rows,
              ["material", "heat_flux_kWm2", "n_replicates", "curve_points",
               "total_heat_release_MJm2", "integrated_to_time_s", "source", "citation"])

    materials_tti = len(set(r["material"] for r in tti_rows))
    materials_phrr = len(set(r["material"] for r in phrr_rows))
    print(f"TTI:  {len(tti_rows)} replicate-level rows across {materials_tti} materials -> tti_fsri.csv")
    print(f"pHRR: {len(phrr_rows)} material/heat-flux rows across {materials_phrr} materials -> phrr_fsri.csv")
    print(f"THR:  {len(thr_rows)} material/heat-flux rows -> thr_fsri.csv")
