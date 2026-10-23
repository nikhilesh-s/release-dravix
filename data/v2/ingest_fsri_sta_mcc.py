#!/usr/bin/env python3
"""Ingest FSRI STA (char yield) and MCC (micro-scale HRR + char yield) data.

Same source/citation as ingest_fsri.py — see that file's docstring and
reports/V2_DATA_ACQUISITION_LOG.md for the full provenance note. This script
adds two more real, replicate-level Tier-2-relevant tables:

- char_yield_sta.csv: final residual mass %, read directly from each STA
  replicate's own data curve (last row = end of the N2 heating ramp, ~640-750 C)
  -- real Simultaneous Thermal Analyzer measurements, N2 atmosphere, 3/10/30 K/min.

  FIXED 2026-09-29 (repro audit, reports/sprint_0929/repro_audit.md 3c/3d; fix in
  reports/sprint_0929/sta_fix.md): 788 of 848 FSRI STA exports have a
  'Mass loss.../mg' column -- mass CHANGE from the start of the run (~0 at t0),
  not absolute mass. Only 60 are 'Mass/%'. Before this fix the last raw value was
  stored as a "percent", so for 93% of rows final_residual_mass_pct was mg lost
  (median -3.18) and exp005's "only 4 materials char" was a unit artifact. The
  Meta-file lookup matched only one of the three spellings of the key
  ('SAMPLE MASS /mg', not '... /mg:' or 'SAMPLE MASS/mg'), so sample_mass_mg was
  blank for 575 of 848 rows. read_sta_percent() below is now the ONE STA
  reader, shared with ingest_sta_onset_temp.py; it converts both export types to
  residual mass % and applies a QA gate (final % outside [-5, 105] -> whole file
  NaN; catches 2 files whose Meta sample mass contradicts the curve).
- mcc_hrr.csv: peak specific HRR (W/g) from each Micro-scale Combustion
  Calorimeter replicate's own HRR(t) curve, plus char yield from the paired
  FINAL_MASS file (final mass / initial sample weight).

  FIXED 2026-09-29 (reports/sprint_0929/mcc_fix.md; found in repro_audit.md 3g):
  - File names: the old regex required '_MCC_', so every MCC file of 9 materials
    named without it ('Vinyl_Siding_30K_min_210609_R1.txt') was silently skipped
    (118 -> 127 materials). mcc_replicate_files() below is the one MCC file
    lister, shared with ingest_fsri_props.py.
  - '_O<n>' files are replicates FSRI flags as outliers (02_Scripts/plot_MCC_data.py:
    'if "O" not in col_name: # to ignore outliers'). They are excluded on purpose now,
    not by accident of the regex.
  - heating_rate_C_s is now the file header's 'Heating Rate (C/s)' (0.5, or 1 for
    Cotton_Shirt R1-R3 and Nylon_Carpet_High_Pile R1-R2). BEFORE 2026-09-29 THIS
    COLUMN HELD K/min (30.0 for every row, from the file name) under a C/s label.
  - peak_specific_hrr_Wg is unchanged, as measured. Peak W/g scales with heating
    rate, so rate_normalized_peak_Wg = peak * (0.5 / rate) puts the 1 C/s rows on
    the 0.5 C/s scale (equal to peak for every 0.5 C/s row). Every training script
    written before 2026-09-29 reads peak_specific_hrr_Wg, and still does.
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


def parse_meta(path):
    kv = {}
    with open(path, newline="", encoding="latin-1", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].startswith("#"):
                kv[row[0].lstrip("#").strip()] = row[1].strip()
    return kv


def read_sta_percent(data_path):
    """One FSRI STA *Data* file -> (temps_C, mass_pct or None, unit, sample_mass_mg).

    mass_pct = residual mass as % of the first exported row, for both export types:
    - '.../%' files: 100 * M / M[0]
    - 'Mass loss.../mg' files: M is the change from run start, so absolute mass is
      SAMPLE MASS (Meta file) + M, normalised by its first row. This is FSRI's own
      convention (02_Scripts/plot_STA_data.py: 'Mass/mg' = m0 + Mass; nMass = Mass/mg / first row).
    mass_pct is None when an mg file has no Meta sample mass, or when the QA gate
    fails (final % outside [-5, 105]: the Meta mass contradicts the curve, e.g.
    HIPS 30K R2 says 1.00 mg but loses 3.93 mg).
    """
    with open(data_path, newline="", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        ti = next(i for i, h in enumerate(header) if "Temp" in h)  # every FSRI variant has exactly one
        mi = next(i for i, h in enumerate(header) if "Mass" in h)
        temps, masses = [], []
        for row in reader:
            if len(row) > max(ti, mi):
                t, m = to_float(row[ti]), to_float(row[mi])
                if t is not None and m is not None:
                    temps.append(t)
                    masses.append(m)
    meta_path = data_path.replace("Data_", "Meta_")
    # Meta key is spelled 'SAMPLE MASS /mg:', 'SAMPLE MASS /mg' or 'SAMPLE MASS/mg' depending on export
    meta = parse_meta(meta_path) if os.path.exists(meta_path) else {}
    sample_mass = to_float(next((v for k, v in meta.items() if k.startswith("SAMPLE MASS")), None))
    unit = "%" if "%" in header[mi] else "mg"
    pct = None
    if masses and unit == "%":
        pct = [100.0 * m / masses[0] for m in masses]
    elif masses and sample_mass:
        pct = [100.0 * (sample_mass + m) / (sample_mass + masses[0]) for m in masses]
    if pct is not None and not -5.0 <= pct[-1] <= 105.0:
        pct = None
    return temps, pct, unit, sample_mass


def ingest_sta():
    rows = []
    for data_path in sorted(glob.glob(os.path.join(RAW, "*", "STA", "N2", "*", "*Data*.csv"))):
        fname = os.path.basename(data_path)
        m = re.match(r"(.+)_STA_N2_(\d+)KData_(\d+)_R(\d+)\.csv", fname)
        if not m:
            continue
        material_dir_full = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_path)))))
        _, heating_rate, test_date, replicate = m.groups()
        temps, pct, unit, sample_mass = read_sta_percent(data_path)
        if not temps:
            continue
        rows.append({
            "material": material_dir_full,
            "material_label": material_dir_full,  # as before; Meta MATERIAL is FSRI's anonymised label ("Plastic A")
            "atmosphere": "N2",
            "heating_rate_K_min": to_float(heating_rate),
            "replicate": replicate,
            "test_date": test_date,
            "sample_mass_mg": sample_mass,
            "final_residual_mass_pct": round(pct[-1], 3) if pct else None,
            "curve_points": len(temps),
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


def sta_self_check(rows):
    """Physical sanity asserts on the corrected residues (material means over replicates)."""
    def mean_res(mat):
        v = [r["final_residual_mass_pct"] for r in rows if r["material"] == mat and r["final_residual_mass_pct"] is not None]
        return sum(v) / len(v)
    assert mean_res("HDPE") < 5, mean_res("HDPE")                 # polyolefin: no char in N2
    assert 15 < mean_res("MDF") < 35, mean_res("MDF")             # wood fibre board: ~20-25% N2 char
    assert 10 < mean_res("Oak_Flooring") < 35, mean_res("Oak_Flooring")  # an mg-export cellulosic
    assert sum(r["final_residual_mass_pct"] is None for r in rows) <= 5  # QA-gated / no Meta mass
    print(f"STA self-check OK: HDPE {mean_res('HDPE'):.2f}%, MDF {mean_res('MDF'):.2f}%, "
          f"Oak_Flooring {mean_res('Oak_Flooring'):.2f}% mean N2 residue")


# '<Mat>_MCC_30K_min_<date>_R<n>.txt'; 9 materials drop the '_MCC' token. '_O<n>' = FSRI-flagged outlier.
MCC_NAME = re.compile(r"(.+?)(?:_MCC)?_(\d+)K_min_(\d+)_([RO])(\d+)\.txt")


def mcc_replicate_files():
    """([(path, material_dir, test_date, replicate), ...], n_outliers) for the FSRI MCC data files.
    R replicates only: O-tagged outliers are counted and excluded; any other file name fails loudly."""
    files, n_outliers = [], 0
    for path in sorted(glob.glob(os.path.join(RAW, "*", "MCC", "*.txt"))):
        if "FINAL_MASS" in path:
            continue
        m = MCC_NAME.fullmatch(os.path.basename(path))
        assert m, f"unrecognised MCC file name: {path}"
        _, _, test_date, kind, replicate = m.groups()
        if kind == "O":
            n_outliers += 1
            continue
        files.append((path, os.path.basename(os.path.dirname(os.path.dirname(path))), test_date, replicate))
    return files, n_outliers


def ingest_mcc():
    rows = []
    files, n_outliers = mcc_replicate_files()
    for path, material_dir, test_date, replicate in files:
        sample_weight = rate = None
        max_hrr = None
        with open(path, newline="", encoding="latin-1", errors="replace") as f:
            lines = f.readlines()
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "*":
                header_line_idx = i + 1
                break
            if line.startswith("Sample Weight"):
                sample_weight = to_float(line.split("\t")[-1].strip())
            if line.startswith("Heating Rate"):
                rate = to_float(line.split("\t")[-1].strip())  # C/s, from the header, not the file name
        if header_line_idx is None:
            continue
        header = lines[header_line_idx].strip().split("\t")
        hrr_idx = None
        for i, h in enumerate(header):
            if "HRR" in h:
                hrr_idx = i
                break
        if hrr_idx is None:
            continue
        vals = []
        for line in lines[header_line_idx + 1:]:
            parts = line.strip().split("\t")
            if len(parts) <= hrr_idx:
                continue
            v = to_float(parts[hrr_idx])
            if v is not None:
                vals.append(v)
        if not vals or sample_weight is None or not rate:
            continue
        max_hrr = max(vals)

        final_mass_path = path.replace(".txt", "_FINAL_MASS.txt")
        final_mass = None
        if os.path.exists(final_mass_path):
            with open(final_mass_path) as f:
                final_mass = to_float(f.read().strip())
        char_yield_pct = (final_mass / sample_weight * 100.0) if (final_mass is not None and sample_weight) else None

        rows.append({
            "material": material_dir,
            "heating_rate_C_s": rate,
            "replicate": replicate,
            "test_date": test_date,
            "sample_weight_mg": sample_weight,
            "peak_specific_hrr_Wg": round(max_hrr, 3),
            "rate_normalized_peak_Wg": round(max_hrr * 0.5 / rate, 3),
            "final_mass_mg": final_mass,
            "char_yield_pct": round(char_yield_pct, 2) if char_yield_pct is not None else None,
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows, n_outliers


def mcc_self_check(rows, n_outliers):
    """The 2026-09-29 fix: recovered materials present, rates from headers, outliers excluded on purpose."""
    mats = {r["material"] for r in rows}
    recovered = {"Composite_Deck_Board", "Corian_Countertop", "Engineered_Flooring", "Eucalyptus_Flooring",
                 "Fiber_Cement_Board", "Masonite_Board", "Pine_Siding", "Pressure_Treated_Deck", "Vinyl_Siding"}
    assert recovered <= mats, recovered - mats
    assert {r["heating_rate_C_s"] for r in rows} <= {0.5, 1.0}  # never the file name's 30 (K/min)
    fast = {r["material"] for r in rows if r["heating_rate_C_s"] == 1.0}
    assert fast == {"Cotton_Shirt", "Nylon_Carpet_High_Pile"}, fast
    assert all(r["rate_normalized_peak_Wg"] == r["peak_specific_hrr_Wg"] for r in rows if r["heating_rate_C_s"] == 0.5)
    assert n_outliers > 0  # the O-exclusion branch actually ran
    print(f"MCC self-check OK: {len(mats)} materials incl. the 9 named without '_MCC_', rates "
          f"{sorted({r['heating_rate_C_s'] for r in rows})} C/s, {n_outliers} O-tagged outlier files excluded")


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sta_rows = ingest_sta()
    mcc_rows, n_outliers = ingest_mcc()
    sta_self_check(sta_rows)  # before writing: a failed physics check must not overwrite either CSV
    mcc_self_check(mcc_rows, n_outliers)

    write_csv(os.path.join(OUT_DIR, "char_yield_sta.csv"), sta_rows,
              ["material", "material_label", "atmosphere", "heating_rate_K_min", "replicate",
               "test_date", "sample_mass_mg", "final_residual_mass_pct", "curve_points",
               "source", "citation"])
    write_csv(os.path.join(OUT_DIR, "mcc_hrr.csv"), mcc_rows,
              ["material", "heating_rate_C_s", "replicate", "test_date", "sample_weight_mg",
               "peak_specific_hrr_Wg", "rate_normalized_peak_Wg", "final_mass_mg", "char_yield_pct",
               "source", "citation"])

    print(f"STA char yield: {len(sta_rows)} replicate rows across "
          f"{len(set(r['material'] for r in sta_rows))} materials -> char_yield_sta.csv")
    print(f"MCC:            {len(mcc_rows)} replicate rows across "
          f"{len(set(r['material'] for r in mcc_rows))} materials -> mcc_hrr.csv")
