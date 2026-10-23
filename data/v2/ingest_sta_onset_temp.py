#!/usr/bin/env python3
"""Dravix V2 -- decomposition onset temperature, a physically real
ignition-relevant feature, extracted from the same STA data files as
char_yield_sta.csv (see ingest_fsri_sta_mcc.py).

Onset temp := the temperature at which cumulative mass loss first reaches
5% of the initial value (a standard TGA/STA onset-of-decomposition
definition), read off each replicate's own Temp/Mass curve. Real materials
that decompose/ignite easily (low activation energy) show a LOW onset
temp; materials that resist ignition show a HIGH one -- this is a much
more direct physical link to Time to Ignition than the bulk MCC peak-HRR
proxy used in exp003/exp004, which measures combustion energy, not
ignition threshold.

FIXED 2026-09-29 (repro audit, reports/sprint_0929/repro_audit.md 3c; fix in
reports/sprint_0929/sta_fix.md): 788 of 848 FSRI STA exports have a
'Mass loss.../mg' column (mass CHANGE from run start, ~0 at t0), not absolute
mass. The old `threshold = masses[0] * 0.95` therefore fired on the first noisy
sample, so most committed onsets were the scan START temperature (~50 C; ABS
110.7 C mean, HDPE 49.5 C). This script now uses read_sta_percent() from
ingest_fsri_sta_mcc.py (the one shared STA reader: Meta SAMPLE MASS, both export
types, QA gate). A replicate that never loses 5% (inert mineral wool) or fails
the QA gate is written with an EMPTY onset instead of being dropped, so
"measured, no onset" is distinguishable from "no STA data".
"""
import csv
import glob
import os
import re

from ingest_fsri_sta_mcc import read_sta_percent

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


def onset_temp(data_path, loss_frac=0.05):
    """First temperature at which residual mass <= (1 - loss_frac) of the start; None if never / QA-gated."""
    temps, pct, _, _ = read_sta_percent(data_path)
    if pct is None:
        return None
    return next((t for t, p in zip(temps, pct) if p <= 100.0 * (1 - loss_frac)), None)


def ingest():
    rows = []
    for data_path in sorted(glob.glob(os.path.join(RAW, "*", "STA", "N2", "*", "*Data*.csv"))):
        fname = os.path.basename(data_path)
        m = re.match(r"(.+)_STA_N2_(\d+)KData_(\d+)_R(\d+)\.csv", fname)
        if not m:
            continue
        material_dir_full = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_path)))))
        _, heating_rate, test_date, replicate = m.groups()
        t_onset = onset_temp(data_path)
        rows.append({
            "material": material_dir_full,
            "heating_rate_K_min": to_float(heating_rate),
            "replicate": replicate,
            "test_date": test_date,
            "onset_temp_5pct_C": round(t_onset, 2) if t_onset is not None else None,
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def self_check(rows):
    """Physical sanity asserts on the corrected onsets (material means over replicates)."""
    def mean_onset(mat):
        v = [r["onset_temp_5pct_C"] for r in rows if r["material"] == mat and r["onset_temp_5pct_C"] is not None]
        return sum(v) / len(v) if v else None
    assert 350 <= mean_onset("ABS") <= 420, mean_onset("ABS")      # literature ~380-400 C in N2
    assert mean_onset("HDPE") > 400, mean_onset("HDPE")             # polyolefin: stable to >400 C
    assert mean_onset("Mineral_Wool_Insulation") is None            # inert: never loses 5%
    n_none = sum(r["onset_temp_5pct_C"] is None for r in rows)
    print(f"Onset self-check OK: ABS {mean_onset('ABS'):.1f} C, HDPE {mean_onset('HDPE'):.1f} C; "
          f"{n_none} replicate rows with no 5% loss or QA-gated (written with empty onset)")


if __name__ == "__main__":
    rows = ingest()
    self_check(rows)  # before writing: a failed physics check must not overwrite the CSV
    write_csv(os.path.join(OUT_DIR, "onset_temp_sta.csv"), rows,
              ["material", "heating_rate_K_min", "replicate", "test_date",
               "onset_temp_5pct_C", "source", "citation"])
    print(f"Onset temp: {len(rows)} replicate rows across "
          f"{len(set(r['material'] for r in rows))} materials -> onset_temp_sta.csv")
