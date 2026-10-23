#!/usr/bin/env python3
"""Dravix V2 -- MaCFP matl-db PMMA cone calorimeter ingestion.

Source: github.com/MaCFP/matl-db (MIT licensed, cloned to data/v2/raw/matl-db/).
73 real cone-calorimeter tests on PMMA from 12 INDEPENDENT INSTITUTIONS
(Aalto, DBI_Lund, Edinburgh, GIDAZE+, HKPoly, LCPP, NIST, TIFP, UClan,
UDRI, ULSTER, UQ) at various heat fluxes -- a multi-lab replication study,
not a single-source dataset like FSRI.

Known gap (found 2026-09-29), FIXED the same day (stream cone_replicates): the
first version yielded 67 tests from 10 of those institutions. HKPoly's files end
in upper-case `.CSV` and name the column "Heat release" (not "HRR"); UClan's
have no `_<replicate>` suffix (one test per flux, one HRR column per file, per
its README: HRR [kW] scaled to kW/m2 with 0.0088 m2). The glob, regex and column
match below now take all three forms, and UClan's single test gets replicate "1".
To keep exp008/exp020's inputs fixed, this script now writes
phrr_macfp_pmma_all.csv (73 tests / 12 labs) and NO LONGER writes
phrr_macfp_pmma.csv (67 tests / 10 labs, frozen). It asserts that the frozen
file equals the new table restricted to those 10 labs, row for row.

Caveat on the recovered labs: HKPoly's pHRR (1,077-1,141 kW/m2 at 25, 2,909-2,974 at 65) is the
maximum at both fluxes, ~2x the other labs, and its curve integrates to 316 / 427 MJ/m2 vs 167-224
MJ/m2 at every other lab for similar 70-80 g specimens. That is more heat than the specimen holds, so
treat HKPoly as suspect (likely an HRR scaling problem in its export). UClan is in range.

Rerun: python3 data/v2/ingest_macfp_pmma.py

This is NOT added to training data. It is used as an OUT-OF-DISTRIBUTION
validation set for the already-promoted pHRR model (models/v2/phrr_rf_v1.joblib,
exp006): every training/validation split so far has been within FSRI's
single lab/instrument. Testing that model against a different lab's PMMA
measurements is a genuinely harder generalization check than a same-lab
material-grouped split -- if it holds up here, that is much stronger
evidence than another same-source CV number.
"""
import csv
import glob
import os
import re

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "matl-db", "PMMA", "Calibration_Data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = "macfp_matl-db"
CITATION = "MaCFP Condensed-Phase Material Database (matl-db), github.com/MaCFP/matl-db, DOI 10.18434/mds2-2586. MIT licensed."


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ingest():
    rows = []
    for path in sorted(p for p in glob.glob(os.path.join(RAW, "*", "*Cone*")) if p.lower().endswith(".csv")):
        fname = os.path.basename(path)
        institution = os.path.basename(os.path.dirname(path))
        m = re.match(r".+_Cone_(\d+)kW(?:_(\d+))?\.csv", fname, re.I)
        if not m:
            continue
        heat_flux, replicate = m.group(1), m.group(2) or "1"

        hrr_vals = []
        with open(path, newline="", encoding="latin-1", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            units = next(reader, None)  # units row, e.g. "[s],[g],[kW/m2],[K]"
            if not header:
                continue
            hrr_idx = None
            for i, h in enumerate(header):
                if "HRR" in h or "heat release" in h.lower():
                    hrr_idx = i
                    break
            if hrr_idx is None:
                continue
            for row in reader:
                if len(row) <= hrr_idx:
                    continue
                v = to_float(row[hrr_idx])
                if v is not None:
                    hrr_vals.append(v)
        if not hrr_vals:
            continue
        peak_hrr = max(hrr_vals)
        rows.append({
            "material": "PMMA",
            "institution": institution,
            "heat_flux_kWm2": to_float(heat_flux),
            "replicate": replicate,
            "peak_hrr_kWm2": round(peak_hrr, 2),
            "curve_points": len(hrr_vals),
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


if __name__ == "__main__":
    rows = ingest()
    fields = ["material", "institution", "heat_flux_kWm2", "replicate",
              "peak_hrr_kWm2", "curve_points", "source", "citation"]
    # the frozen 67-test file must be exactly this table minus the two recovered labs (read only)
    with open(os.path.join(OUT_DIR, "phrr_macfp_pmma.csv"), newline="", encoding="utf-8") as f:
        frozen = list(csv.DictReader(f))
    old = [{k: str(r[k]) for k in fields} for r in rows if r["institution"] not in ("HKPoly", "UClan")]
    assert old == frozen, "recovered-lab fix changed the 10 original labs' rows"
    assert len(rows) == 73 and len({r["institution"] for r in rows}) == 12, len(rows)
    write_csv(os.path.join(OUT_DIR, "phrr_macfp_pmma_all.csv"), rows, fields)
    print(f"frozen phrr_macfp_pmma.csv == new table minus HKPoly/UClan ({len(frozen)} rows, not rewritten)")
    print(f"MaCFP PMMA pHRR: {len(rows)} tests across {len(set(r['institution'] for r in rows))} institutions")
    for inst in ("HKPoly", "UClan"):
        print(f"  recovered {inst}: " + ", ".join(f"{r['heat_flux_kWm2']:g} kW/m2 R{r['replicate']} {r['peak_hrr_kWm2']}"
                                                   for r in rows if r["institution"] == inst))
    by_hf = {}
    for r in rows:
        by_hf.setdefault(r["heat_flux_kWm2"], []).append(r["peak_hrr_kWm2"])
    for hf in sorted(by_hf):
        vals = by_hf[hf]
        print(f"  {hf} kW/m2: n={len(vals)}, mean pHRR={sum(vals)/len(vals):.1f}, "
              f"min={min(vals):.1f}, max={max(vals):.1f}")
