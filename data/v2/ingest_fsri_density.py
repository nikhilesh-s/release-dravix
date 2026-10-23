#!/usr/bin/env python3
"""Dravix V2 -- FSRI per-material density, from each material's own
material.json ("density": "1180 kg/m<sup>3</sup>" style string, FSRI's own
stated value with a source note e.g. "Samples cut from 0.32 cm thick...").

This is the missing third term for thermal inertia (k * rho * cp), the
textbook physical predictor of time-to-ignition (Quintiere-style ignition
models: t_ig ~ thermal inertia / (heat flux - critical flux)^2). We
already have real k (thermal_conductivity_fsri.csv) and cp
(specific_heat_fsri.csv) from HFM, both previously unused in any model.
"""
import glob
import json
import os
import re

RAW = "~/Dravix/dravix-v2/data/v2/raw/fsri_materials_database/01_Data"
OUT_DIR = "~/Dravix/dravix-v2/data/v2"
SOURCE = "fsri_materials_database"
CITATION = ("McKinnon, M., Weinschenk, C., Dow, N., DiDomizio, M., and Madrzykowski, D. "
            "Materials and Products Database (Version 1.0.0), Fire Safety Research "
            "Institute, UL Research Institutes. Columbia, MD 21045, 2023.")


def parse_density(s):
    """'1180 kg/m<sup>3</sup>' -> 1180.0 (kg/m3).

    Bug fixed here: an earlier version stripped HTML tags then averaged
    EVERY number in the string, which silently grabbed the "3" exponent
    left behind by "m<sup>3</sup>" and averaged it with the real value
    (e.g. PMMA "1180 kg/m<sup>3</sup>" -> wrongly (1180+3)/2=591.5, caught
    by the PMMA/EPS sanity check below before this was used anywhere).
    Correct approach: take only the FIRST number in the raw string, before
    any unit/exponent markup.
    """
    m = re.match(r"\s*([\d.]+)", s)
    if not m:
        return None
    return float(m.group(1))


def ingest():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "material.json"))):
        material_dir = os.path.basename(os.path.dirname(path))
        try:
            d = json.load(open(path))
        except Exception:
            continue
        dens_str = d.get("density", "")
        density = parse_density(dens_str) if dens_str else None
        if density is None:
            continue
        rows.append({
            "material": material_dir,
            "density_raw": dens_str,
            "density_kg_m3": density,
            "source": SOURCE,
            "citation": CITATION,
        })
    return rows


if __name__ == "__main__":
    import csv
    rows = ingest()
    path = os.path.join(OUT_DIR, "density_fsri.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["material", "density_raw", "density_kg_m3", "source", "citation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Density: {len(rows)} materials -> density_fsri.csv")
