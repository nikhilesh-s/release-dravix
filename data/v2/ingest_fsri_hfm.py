#!/usr/bin/env python3
"""Dravix V2 -- FSRI Heat Flow Meter (HFM) ingestion: thermal conductivity
and specific heat. Same source/citation as ingest_fsri.py.

.tst files are UTF-16 encoded (NETZSCH/Winthermo instrument output).
Each file ends with a "Results Table -- SI Units" section giving real,
multi-temperature-setpoint measurements -- these are read directly, not
derived: FSRI's own instrument software already computed thermal
conductivity (Conductivity files) or specific heat (HeatCapacity files)
per setpoint from the raw plate/transducer signal.
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


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_utf16(path):
    for enc in ("utf-16", "utf-16-le", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            if "\x00" not in text:
                return text
        except (UnicodeError, UnicodeDecodeError):
            continue
    return None


def parse_results_table(text):
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Results Table" in line:
            start = i
            break
    if start is None:
        return None, []
    header_line = None
    for i in range(start + 1, min(start + 5, len(lines))):
        if lines[i].strip():
            header_line = lines[i]
            header_idx = i
            break
    if header_line is None:
        return None, []
    header = [h.strip() for h in header_line.split("\t") if h.strip()]
    data_rows = []
    for line in lines[header_idx + 2:]:  # skip units line
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
        vals = [to_float(p) for p in parts]
        if all(v is not None for v in vals) and len(vals) >= 2:
            data_rows.append(vals)
        else:
            break
    return header, data_rows


def parse_generic(fname):
    """material_dirname_HFM_(Wet|Dry)_<type>_<date>_R<n>.tst -- typo-tolerant."""
    m = re.match(r"(.+)_HFM_(Wet|Dry)_[A-Za-z]+_(\d{6})_R(\d+)\.tst", fname, re.IGNORECASE)
    if not m:
        return None
    return m.groups()  # (name_prefix, condition, test_date, replicate)


def ingest_conductivity():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "HFM", "*.tst"))):
        fname = os.path.basename(path)
        if "conduc" not in fname.lower():
            continue
        m = parse_generic(fname)
        if not m:
            continue
        material_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        _, condition, test_date, replicate = m
        text = read_utf16(path)
        if text is None:
            continue
        header, data_rows = parse_results_table(text)
        if not data_rows:
            continue
        for vals in data_rows:
            row = {"material": material_dir, "condition": condition, "replicate": replicate,
                   "test_date": test_date}
            for h, v in zip(header, vals):
                row[h.replace(" ", "_")] = v
            row.update({"source": SOURCE, "citation": CITATION})
            rows.append(row)
    return rows


def ingest_heat_capacity():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "HFM", "*.tst"))):
        fname = os.path.basename(path)
        if "capacity" not in fname.lower():
            continue
        m = parse_generic(fname)
        if not m:
            continue
        material_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        _, condition, test_date, replicate = m
        text = read_utf16(path)
        if text is None:
            continue
        header, data_rows = parse_results_table(text)
        if not data_rows:
            continue
        for vals in data_rows:
            row = {"material": material_dir, "condition": condition, "replicate": replicate,
                   "test_date": test_date}
            for h, v in zip(header, vals):
                row[h.replace(" ", "_")] = v
            row.update({"source": SOURCE, "citation": CITATION})
            rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    cond_rows = ingest_conductivity()
    cp_rows = ingest_heat_capacity()

    cond_fields = ["material", "condition", "replicate", "test_date",
                   "Mean_Temp", "Upper_Cond", "Lower_Cond", "Average_Cond", "source", "citation"]
    cp_fields = ["material", "condition", "replicate", "test_date",
                 "Mean_Temp", "Enthalpy", "Spc_Heat", "source", "citation"]

    write_csv(os.path.join(OUT_DIR, "thermal_conductivity_fsri.csv"), cond_rows, cond_fields)
    write_csv(os.path.join(OUT_DIR, "specific_heat_fsri.csv"), cp_rows, cp_fields)

    print(f"Thermal conductivity: {len(cond_rows)} setpoint rows across "
          f"{len(set(r['material'] for r in cond_rows))} materials -> thermal_conductivity_fsri.csv")
    print(f"Specific heat:        {len(cp_rows)} setpoint rows across "
          f"{len(set(r['material'] for r in cp_rows))} materials -> specific_heat_fsri.csv")
