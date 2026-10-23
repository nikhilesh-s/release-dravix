#!/usr/bin/env python3
"""Dravix V2 -- cone-calorimeter residue per FSRI cone replicate (added 2026-09-29, stream sta_fix).

WHY: Pomazi et al. 2026's Char Residue (CR, real held-out R2 0.82) is the residue
left after a mass-loss-cone test in air. The closest analog in FSRI is the cone
residue itself, not the N2 STA residue that exp005 used: final specimen mass /
initial specimen mass, from each replicate's load cell.

Definition = FSRI's own (02_Scripts/plot_Cone_data.py, 'Final Mass (g)'):
  holder_mass     = SampleMass[scan 1] - SPECIMEN MASS         (load-cell tare offset)
  final_mass_g    = SampleMass[END OF TEST SCAN] - holder_mass
  cone_residue_pct = 100 * final_mass_g / SPECIMEN MASS
Scan files: 'Sample Mass' in g, 'Time' in s (0.25 s scans). The load cell keeps
logging after END OF TEST SCAN (e.g. ABS HF50 R1: test ends at scan 563, file runs
to 1043), so the LAST row is NOT the residue. The load-cell tare at scan 1 varies a lot
between tests (median 2% of specimen mass, Asphalt_Shingle ~ -300 g), which FSRI's
constant-offset correction handles.
QA gates (residue -> empty): (1) residue outside [-5, 105] (28 rows: mostly <5 g fabrics /
foams where load-cell noise exceeds the window, and Frame tests whose frame went on after
scan 1); (2) SPECIMEN MASS more than 3x off its material's median (PMMA HF25 R2 says
866.19 g vs ~33 g for its siblings, a Scalar typo that would read as 96% residue).
'HF15-noSpark' PMMA tests hit the 1200 s time cap before burnout (~55% left): kept, tagged in
test_config, and excluded by the modelling script, since it is a truncated test, not a residue.

Caveats: residue includes inert substrates/fillers (gypsum, cement, glass fibre,
fabric backings), and 'Frame+Grid' configs are kept with their tag in test_config.
"""
import csv
import glob
import os
import re

from ingest_fsri import CITATION, RAW, SOURCE, parse_scalar, to_float

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def scan_mass(path):
    """{scan index str: Sample Mass g} and {scan index str: Time s} from one cone *Scan* file."""
    mass, time = {}, {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        mi, ti = header.index("Sample Mass"), header.index("Time")
        for row in reader:
            if row and row[0].strip().isdigit() and len(row) > mi:
                mass[row[0].strip()] = to_float(row[mi])
                time[row[0].strip()] = to_float(row[ti])
    return mass, time


def mass_consistency_gate(rows):
    """Blank the residue of a replicate whose SPECIMEN MASS is >3x off its material's median (a typo)."""
    for mat in {r["material"] for r in rows}:
        ms = sorted(r["specimen_mass_g"] for r in rows if r["material"] == mat and r["specimen_mass_g"])
        med = ms[len(ms) // 2] if ms else None
        for r in rows:
            if r["material"] == mat and med and r["specimen_mass_g"] and not med / 3 <= r["specimen_mass_g"] <= 3 * med:
                r["cone_residue_pct"] = None
    return rows


def ingest():
    rows = []
    for scan_path in sorted(glob.glob(os.path.join(RAW, "*", "Cone", "*Scan_*.csv"))):
        m = re.match(r"(.+)_Cone_(HF.+)Scan_(\d+)_R(\d+)\.csv", os.path.basename(scan_path))
        scalar_path = scan_path.replace("Scan_", "Scalar_")
        if not m or not os.path.exists(scalar_path):
            continue
        _, tag, test_date, replicate = m.groups()
        kv = parse_scalar(scalar_path)
        m0, eot = to_float(kv.get("SPECIMEN MASS")), to_float(kv.get("END OF TEST SCAN"))
        mass, time = scan_mass(scan_path)
        eot_key = str(int(eot)) if eot is not None else None
        residue = final = None
        if m0 and m0 > 0 and mass.get("1") is not None and mass.get(eot_key) is not None:
            final = mass[eot_key] - (mass["1"] - m0)
            residue = 100.0 * final / m0
            if not -5.0 <= residue <= 105.0:
                residue = None
        rows.append({
            "material": os.path.basename(os.path.dirname(os.path.dirname(scan_path))),
            "heat_flux_kWm2": to_float(re.search(r"HF(\d+)", tag).group(1)),
            "test_config": tag,
            "replicate": replicate,
            "test_date": test_date,
            "specimen_mass_g": m0,
            "load_cell_start_g": mass.get("1"),
            "final_mass_g": round(final, 4) if final is not None else None,
            "cone_residue_pct": round(residue, 3) if residue is not None else None,
            "end_of_test_time_s": to_float(kv.get("END OF TEST TIME")),
            "scan_time_at_eot_s": time.get(eot_key),
            "source": SOURCE,
            "citation": CITATION,
        })
    return mass_consistency_gate(rows)


def self_check(rows):
    """Physical sanity (bounds fixed before the first run): PMMA ~0, gypsum boards high, wood some char."""
    def mean_res(mat):
        v = [r["cone_residue_pct"] for r in rows if r["material"] == mat and r["cone_residue_pct"] is not None
             and "noSpark" not in r["test_config"]]
        return sum(v) / len(v)
    assert mean_res("PMMA") < 3, mean_res("PMMA")
    assert mean_res("Gypsum_Wallboard") > 60, mean_res("Gypsum_Wallboard")
    assert 5 < mean_res("Oak_Flooring") < 40, mean_res("Oak_Flooring")
    # time basis: END OF TEST TIME (scalar) matches the scan clock at END OF TEST SCAN within 1 s
    lag = [abs(r["end_of_test_time_s"] - r["scan_time_at_eot_s"]) for r in rows
           if r["end_of_test_time_s"] is not None and r["scan_time_at_eot_s"] is not None]
    assert sum(x <= 1.0 for x in lag) >= 0.95 * len(lag), "scan clock / END OF TEST TIME mismatch"
    print(f"Cone residue self-check OK: PMMA {mean_res('PMMA'):.2f}%, Gypsum_Wallboard "
          f"{mean_res('Gypsum_Wallboard'):.1f}%, Oak_Flooring {mean_res('Oak_Flooring'):.1f}%; "
          f"EOT time vs scan clock within 1 s for {sum(x <= 1.0 for x in lag)}/{len(lag)} replicates")


if __name__ == "__main__":
    rows = ingest()
    self_check(rows)  # before writing: a failed physics check must not overwrite the CSV
    fields = ["material", "heat_flux_kWm2", "test_config", "replicate", "test_date", "specimen_mass_g", "load_cell_start_g",
              "final_mass_g", "cone_residue_pct", "end_of_test_time_s", "scan_time_at_eot_s", "source", "citation"]
    with open(os.path.join(OUT_DIR, "char_residue_cone_fsri.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["cone_residue_pct"] is not None]
    print(f"Cone residue: {len(ok)}/{len(rows)} replicate rows with a residue, "
          f"{len(set(r['material'] for r in ok))} materials -> char_residue_cone_fsri.csv")
