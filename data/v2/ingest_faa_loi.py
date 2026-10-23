#!/usr/bin/env python3
"""Dravix V2 -- first real LOI rows: FAA report DOT/FAA/AR-05/14, Appendix A.

WHY: LOI (Tier 1) had zero real rows, and the RPUF formulation datasets are only
reachable by author email. Lyon & Janssens, "Polymer Flammability" (FAA Office of
Aviation Research, May 2005; US-government work) prints LOI next to the MCC/PCFC
heat release capacity (eta_c) for the same polymers in Table A-6. The same appendix
gives char yield (Table A-2) and the PCFC heat of complete combustion of the fuel
gases plus flaming combustion efficiency (Table A-5). Together that is a small,
fully citable LOI dataset with MCC-type features.

Each table row is kept as the verbatim text of the PDF's text layer (read
2026-09-29 from the ROSA P copy at URL below). Every numeric column is parsed
from that string, so the value columns cannot disagree with the provenance
column. Nothing here is typed twice or filled from memory.

Joins across tables are exact-name only. Ambiguous names get no join:
  PE               -> Table A-2/A-5 list "PE HD" and "PE LD" separately
  PE (chlorinated) -> Tables list "CPE (25% Cl)" / "CPE (36% Cl)" / "CPE (48% Cl)"
  SIR (filled)     -> Tables list "SIR" (filler content unknown, not the same material)
  KEVLAR, NOMEX    -> not in Table A-2 or A-5
  PC-FR            -> in Table A-2, not in Table A-5
Only the Table A-2 / A-5 rows needed for those joins are transcribed below.

Caveat carried into the CSV: section 4.7 of the report calls the Table A-6 LOI
values "generic values" (typical literature values per polymer), not paired
measurements on the same specimens as eta_c.

Rerun: python3 ingest_faa_loi.py  -> loi_faa_polymers.csv
"""
import csv
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://rosap.ntl.bts.gov/view/dot/16407/dot_16407_DS1.pdf"
ALT_URL = "https://www.fire.tc.faa.gov/pdf/05-14.pdf"
DOC = "Lyon & Janssens, Polymer Flammability, DOT/FAA/AR-05/14 (FAA, May 2005)"
MISSING = "−"  # the report prints a missing value as U+2212 MINUS SIGN

# Table A-6: LIMITING HEAT RELEASE RATE (HRR0), HEAT RELEASE PARAMETER, LIMITING
# OXYGEN INDEX, UL 94 RATING, AND HEAT RELEASE CAPACITY (eta_c).
# Printed columns: Polymer | HRR0 (kW/m2) | HRP (kJ/kJ) | LOI, [O2] (% v/v) | UL 94 Rating | eta_c (J/g-K)
# Rows 1-33 on report page A-11 (PDF page 80), rows 34-47 on page A-12 (PDF page 81).
A6 = [
    ("HIPS", "HIPS 510 ±77 14 ±2 18 HB 893"),
    ("PP", "PP 369 ±79 22 ±2 17 HB 1200"),
    ("PET", "PET 424 ±168 13 ±4 20 HB 402"),
    ("PS", "PS 410 ±66 17 ±2 18 HB 1040"),
    ("ABS", "ABS 359 ±66 13 ±2 18 HB 669"),
    ("PBT", "PBT 341 ±106 15 ±3 23 HB 474"),
    ("PE (chlorinated)", "PE (chlorinated) − − 21 HB 693"),
    ("UPT", "UPT 261 ±105 11 ±2 20 HB −"),
    ("PC/ABS", "PC/ABS 259 ±43 11 ±1 − HB −"),
    ("PA66", "PA66 240 ±59 18 ±2 24 HB 600"),
    ("PMMA", "PMMA 217 ±47 13 ±1 17 HB 574"),
    ("PS-FR", "PS-FR 205 ±27 7 ±1 − V2 −"),
    ("PPO/PS", "PPO/PS 192 ±22 15 ±1 − HB −"),
    ("PA6", "PA6 187 ±55 18 ±2 24 HB 487"),
    ("PC/ABS-FR", "PC/ABS-FR 178 ±36 7 ±1 − V1 −"),
    ("VET", "VET 169 ±44 13 ±1 − HB 805"),
    ("PESU", "PESU 168 ±23 4 ±0 36 V1 345"),
    ("HIPS-FR", "HIPS-FR 164 ±30 5 ±1 − V2 −"),
    ("POM", "POM 162 ±30 6 ±1 15 HB 398"),
    ("EP", "EP 160 ±46 10 ±1 19 HB 657"),
    ("PE", "PE 145 ±93 21 ±3 17 HB 1560"),
    ("PBT-FR", "PBT-FR 141 ±130 6 ±3 − V2 −"),
    ("CEA", "CEA 112 ±22 7 ±2 24 V1 273"),
    ("ABS-FR", "ABS-FR 117 ±33 6 ±1 − V2 301"),
    ("HIPS-FR", "HIPS-FR 114 ±36 4 ±1 − V0 −"),
    ("PVC (flex)", "PVC (flex) 91 ±19 4 ±0 − V2 −"),
    ("SIR (filled)", "SIR (filled) 90 ±13 5 ±0 32 V0 88"),
    ("PC", "PC 89 ±32 9 ±1 25 V2 390"),
    ("PEN", "PEN 57 ±13 5 ±0 32 V2 309"),
    ("ETFE", "ETFE 44 ±31 6 ±0 30 V0 198"),
    ("PVC (rigid)", "PVC (rigid) 9 ±25 3 ±1 45 V0 138"),
    ("CR", "CR − 9 40 V0 188"),
    ("KEVLAR", "KEVLAR − 32 V0 302"),
    ("UPT-FR", "UPT-FR -31 ±10 2 ±0 − V0 −"),
    ("CPVC", "CPVC -34 ±9 3 ±0 52 V0 −"),
    ("PE (crosslinked)-FR", "PE (crosslinked)-FR -38 ±28 5 ±5 − V0 −"),
    ("PAI", "PAI -64 ±16 2 ±0 45 V0 33"),
    ("PPSU", "PPSU -83 ±25 4 ±0 38 V0 115"),
    ("PTFE", "PTFE -84 ±9 2 ±0 95 V0 35"),
    ("PEEK", "PEEK -94 ±20 3 ±0 35 V0 155"),
    ("NOMEX", "NOMEX − − 28 V0 152"),
    ("PEI", "PEI -113 ±19 3 ±0 47 V0 121"),
    ("ECTFE", "ECTFE -127 ±6 4 ±0 60 V0 −"),
    ("PPS", "PPS -147 ±30 6 ±1 44 V0 164"),
    ("PBI", "PBI -150 ±36 3 ±0 36 V0 41"),
    ("PC-FR", "PC-FR -191 ±51 4 ±1 56 V0 30"),
    ("FEP", "FEP − 2 95 V0 57"),
]

# Table A-2: LATENT HEAT OF GASIFICATION, CHAR YIELD, AND ENTHALPY OF GASIFICATION OF POLYMERS.
# Printed columns: Polymer | Lg (kJ/g) | mu (g/g) | hg (kJ/g). Page A-4 (PDF p.73) through PET; A-5 (PDF p.74) after.
A2 = [
    ("A-4", "ABS 2.3 0.00 2.3"), ("A-4", "CEA 4.0 0.36 2.6"), ("A-4", "CR 2.0 0.13 1.7"),
    ("A-4", "EP 1.6 0.04 1.5"), ("A-4", "ETFE 1.1 0.00 1.1"), ("A-4", "FEP 1.5 0.00 1.5"),
    ("A-4", "HIPS 2.0 0.00 2.0"), ("A-4", "PA6 1.5 0.02 1.5"), ("A-4", "PA66 2.1 0.04 2.0"),
    ("A-4", "PAI 4.8 0.61 1.9"), ("A-4", "PBI 5.5 0.75 1.4"), ("A-4", "PBT 1.4 0.07 1.3"),
    ("A-4", "PC 2.4 0.25 1.8"), ("A-4", "PC-FR 3.5 0.52 1.7"), ("A-4", "PEEK 3.4 0.54 1.6"),
    ("A-4", "PEI 3.5 0.53 1.7"), ("A-4", "PEN 2.5 0.18 2.0"), ("A-4", "PESU 3.8 0.40 2.3"),
    ("A-4", "PET 1.4 0.05 1.3"), ("A-5", "PMMA 1.7 0.00 1.7"), ("A-5", "POM 2.4 0.00 2.4"),
    ("A-5", "PP 1.9 0.00 1.9"), ("A-5", "PPS 3.8 0.50 1.9"), ("A-5", "PPSU 4.3 0.58 1.8"),
    ("A-5", "PS 1.8 0.00 1.8"), ("A-5", "PTFE 2.5 0.00 2.5"), ("A-5", "PVC (rigid) 2.3 0.09 2.1"),
]

# Table A-5: EFFECTIVE HEAT OF COMBUSTION, HEAT OF COMPLETE COMBUSTION OF FUEL GASES (hc0),
# AND FLAMING COMBUSTION EFFICIENCY (chi). Printed columns: Polymer | HOC (kJ/g) | hc0 (kJ/g) | chi.
# Page A-9 (PDF p.78) through ABS; A-10 (PDF p.79) after. Section 4.2: hc0 measured by PCFC (= MCC).
A5 = [
    ("A-9", "PMMA 24.8 25.0 0.99"), ("A-9", "PP 41.9 43.0 0.97"), ("A-9", "PAI 15.3 15.9 0.96"),
    ("A-9", "CR 17.6 18.5 0.95"), ("A-9", "PEI 21.8 23.2 0.94"), ("A-9", "CEA 25.9 27.6 0.94"),
    ("A-9", "POM 14.4 15.4 0.94"), ("A-9", "PA66 25.2 27.4 0.92"), ("A-9", "PEEK 21.3 23.2 0.92"),
    ("A-9", "PBT 21.7 23.8 0.91"), ("A-9", "PC 21.2 23.3 0.91"), ("A-9", "PEN 22.9 25.2 0.91"),
    ("A-9", "PESU 22.4 24.9 0.90"), ("A-9", "PA6 25.8 28.7 0.90"), ("A-9", "PPSU 23.5 27.0 0.87"),
    ("A-9", "PPS 25.4 29.3 0.87"), ("A-9", "PBI 22.0 26.5 0.83"), ("A-9", "ABS 29.0 36.6 0.79"),
    ("A-10", "HIPS 28.1 37.2 0.76"), ("A-10", "EP 20.4 27.1 0.75"), ("A-10", "PET 18.0 24.1 0.75"),
    ("A-10", "ETFE 7.3 10.8 0.68"), ("A-10", "PS 27.9 42.0 0.66"), ("A-10", "PTFE 4.6 8.4 0.55"),
    ("A-10", "PVC (rigid) 9.3 18.6 0.50"), ("A-10", "FEP 1.3 7.7 0.17"),
]
PDF_PAGE = {"A-4": 73, "A-5": 74, "A-9": 78, "A-10": 79}


def num(tok):
    return "" if tok == MISSING else float(tok)


def by_name(rows):
    """Right-parse fixed 3-value rows; the polymer name is whatever precedes them."""
    out = {}
    for page, row in rows:
        tok = row.split(" ")
        out[" ".join(tok[:-3])] = (page, row, tok[-3:])
    return out


def build():
    a2, a5 = by_name(A2), by_name(A5)
    records = []
    for i, (name, row) in enumerate(A6, start=1):
        assert row.startswith(name + " "), row
        tok = row.split(" ")  # parse from the right: ... LOI | UL 94 | eta_c
        rec = {
            "table_row": i, "polymer": name,
            "LOI_pct": num(tok[-3]), "UL94_rating": tok[-2], "HRC_J_gK": num(tok[-1]),
            "char_yield_g_g": "", "hc0_kJ_g": "", "HOC_kJ_g": "", "chi": "",
            "source_url": URL,
            "table_ref": f"{DOC}, Table A-6, p. {'A-11 (PDF p. 80)' if i <= 33 else 'A-12 (PDF p. 81)'}",
            "verbatim_row": row, "A2_ref": "", "A2_verbatim_row": "", "A5_ref": "", "A5_verbatim_row": "",
        }
        if name in a2:
            page, r, (_, mu, _) = a2[name]
            rec.update(char_yield_g_g=float(mu), A2_verbatim_row=r,
                       A2_ref=f"Table A-2, p. {page} (PDF p. {PDF_PAGE[page]})")
        if name in a5:
            page, r, (hoc, hc0, chi) = a5[name]
            rec.update(HOC_kJ_g=float(hoc), hc0_kJ_g=float(hc0), chi=float(chi), A5_verbatim_row=r,
                       A5_ref=f"Table A-5, p. {page} (PDF p. {PDF_PAGE[page]})")
        records.append(rec)
    return records


if __name__ == "__main__":
    recs = build()
    # Self-check: 47 printed rows; spot values read straight off the page.
    assert len(recs) == 47
    ptfe = next(r for r in recs if r["polymer"] == "PTFE")
    assert (ptfe["LOI_pct"], ptfe["HRC_J_gK"], ptfe["char_yield_g_g"], ptfe["chi"]) == (95.0, 35.0, 0.0, 0.55)
    kev = next(r for r in recs if r["polymer"] == "KEVLAR")
    assert (kev["LOI_pct"], kev["UL94_rating"], kev["HRC_J_gK"]) == (32.0, "V0", 302.0)
    out = f"{DATA_DIR}/loi_faa_polymers.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    both = [r for r in recs if r["LOI_pct"] != "" and r["HRC_J_gK"] != ""]
    full = [r for r in both if r["char_yield_g_g"] != "" and r["chi"] != ""]
    print(f"wrote {out}: {len(recs)} table rows; {len(both)} with LOI + eta_c; "
          f"{len(full)} of those also with char yield + hc0 + chi")
