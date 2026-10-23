#!/usr/bin/env python3
"""Dravix V2 -- LOI rows from a second FAA report: DOT/FAA/AR-01/53, "A Fire-Resistant Epoxy".

WHY: loi_faa_polymers.csv (AR-05/14) has 32 polymers with LOI + heat release capacity (eta_c).
The follow-up lead AR-01/31 (Walters & Lyon, molar group contributions) turned out to print
LOI only as an unlabeled scatter plot (its Figure 6), so it adds no LOI rows. ROSA P lists
AR-01/53 as a related document. It prints LOI for cured epoxy systems that are not in
loi_faa_polymers.csv, and eta_c for the same systems, measured by the same group on the same
formulations:
  Table 2 "THERMAL ANALYSIS RESULTS"      report p. 6 (PDF p. 16): DSC, TGA, PCFC eta_c / hc / mu
  Table 3 "FIRE AND FLAMMABILITY DATA"    report p. 9 (PDF p. 19): OSU pHRR / THR, LOI, UL 94
Both tables have the same 9 rows (hardener x epoxy DGEBA/DGEBC), so they are joined on the
exact (hardener, epoxy) key. The two tables are on different pages; that is the only join.

Source: Lyon, Castelli & Walters, DOT/FAA/AR-01/53 (FAA, August 2001), US-government work.
Read 2026-09-29 from the ROSA P copy (URL below). The PDF is an NTIS scan with NO text layer,
so the verbatim rows below were transcribed by eye from the page images (pypdf-extracted page
strips, stitched in content-stream order). Each row is one printed table row, cells in column
order separated by single spaces; the merged "Hardener" cell is repeated on both of its rows.
All numeric columns are parsed from these strings. Nothing is filled from memory.

Independent check (no text layer exists, so it cannot be a text-extraction check like
ingest_faa_loi.py's): SECOND_READ holds eta_c and LOI re-read from separate full-resolution
crops of each table, and the script asserts they equal the values parsed from the verbatim rows.
It also asserts three facts stated in the report's prose, independent of the tables:
  p. 6: "The DGEBC systems had consistently lower heat release capacities, eta_c, and total
        heat release, h_c, than did the DGEBA systems"  -> eta_c and h_c: C < A for every hardener
  p. 8: "(T_p = 420 C in table 2)" for DGEBC-EMI
  p. 8: DGEBC-MDA "experimental char yield (mu = 39%)"  -> TGA mu of MDA/C
Same reader both times, so this catches misreads of the page, not a systematic bias.

Choices fixed before any model saw these rows:
  - LOI is printed as a range ("19-20"); LOI_pct is the midpoint. LOI_range keeps the printed text.
  - "—" (em dash) is a blank. 3 of 9 rows have no LOI.
  - The abstract says "The DGEBC cured with methylenedianiline had an LOI of 30-31", but Table 3
    prints "—" for MDA/C, and the same sentence goes on about a "single-ply glass fabric lamina".
    The specimen is ambiguous, so that value is NOT used; it is kept in `note` only.
  - hc is "per unit initial mass of compound" (report p. 7-8), i.e. not the same definition as
    AR-05/14's hc0 (per mass of fuel gases). Kept under a distinct column name, not merged.
  - PCFC here ran at a heating rate "<= 5 K/s" (report p. 4); eta_c values are as printed.

Rerun: python3 ingest_faa_loi_ar0153.py  -> loi_faa_ar0153.csv
"""
import csv
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://rosap.ntl.bts.gov/view/dot/92723/dot_92723_DS1.pdf"
DOC = "Lyon, Castelli & Walters, A Fire-Resistant Epoxy, DOT/FAA/AR-01/53 (FAA, August 2001)"
MISSING = "—"  # the report prints a missing value as an em dash

# Table 2. Columns: Hardener | Epoxy DGEB | DSC dH_polym (kJ/mol) | TGA T_p (C) | TGA mu (%) |
# PCFC eta_c (J/g-K) | PCFC h_c (kJ/g) | PCFC mu (%)
T2 = [
    "EMI-24 A 72 460 11 833 25 5",
    "EMI-24 C 83 420 44 487 10 36",
    "TETA A 92 400 8 655 26 4",
    "TETA C 78 350 30 577 15 23",
    "MDA A 94 425 18 641 26 8",
    "MDA C 111 310 39 261 16 32",
    "BPA A 83 450 10 858 27 1",
    "BPC C 76 330 44 153 11 39",
    "CEBPC C 120 350 47 291 8 42",
]
# Table 3. Columns: Hardener | Epoxy DGEB | Peak Heat Release Rate (kW/m2) |
# Total Heat Release (kW-min/m2) | LOI (% O2) | UL 94     [pHRR/THR are FAR 25.853(a-1), per p. 8]
T3 = [
    "EMI-24 A 109 42 19-20 B",
    "EMI-24 C 50 26 27-28 B",
    "TETA A 144 67 21-22 B",
    "TETA C 70 32 — —",
    "MDA A 82 46 27-28 B",
    "MDA C 44 32 — V-0",
    "BPA A 71 49 20-22 B",
    "BPC C 30 19 35-36 V-0",
    "CEBPC C 34 18 — —",
]
# Second, separate read of eta_c (Table 2) and LOI (Table 3) from full-resolution table crops.
SECOND_READ = {
    ("EMI-24", "A"): (833, "19-20"), ("EMI-24", "C"): (487, "27-28"),
    ("TETA", "A"): (655, "21-22"), ("TETA", "C"): (577, MISSING),
    ("MDA", "A"): (641, "27-28"), ("MDA", "C"): (261, MISSING),
    ("BPA", "A"): (858, "20-22"), ("BPC", "C"): (153, "35-36"), ("CEBPC", "C"): (291, MISSING),
}
NOTE = {("MDA", "C"): "Abstract: 'The DGEBC cured with methylenedianiline had an LOI of 30-31' "
                      "(glass-fabric lamina context; Table 3 prints '—'); not used."}


def num(tok):
    return "" if tok == MISSING else float(tok)


def loi_mid(tok):
    if tok == MISSING:
        return ""
    lo, hi = (float(v) for v in tok.split("-"))
    assert 10 < lo <= hi < 100, tok
    return (lo + hi) / 2


def build():
    t3 = {tuple(r.split(" ")[:2]): r for r in T3}
    assert len(t3) == len(T3) == len(T2) == 9
    recs = []
    for i, row2 in enumerate(T2, start=1):
        hard, epo, dh, tp, mu_tga, eta, hc, mu = row2.split(" ")
        row3 = t3[(hard, epo)]  # exact key; KeyError if the tables disagree
        _, _, phrr, thr, loi, ul94 = row3.split(" ")
        recs.append({
            "table_row": i, "polymer": f"DGEB{epo}-{hard}", "hardener": hard, "epoxy": f"DGEB{epo}",
            "LOI_pct": loi_mid(loi), "LOI_range": "" if loi == MISSING else loi,
            "UL94_rating": "" if ul94 == MISSING else ul94,
            "HRC_J_gK": num(eta), "char_yield_g_g": float(mu) / 100, "hc_kJ_g_per_initial_mass": num(hc),
            "TGA_Tp_C": num(tp), "TGA_char_pct": num(mu_tga), "dH_polym_kJ_mol": num(dh),
            "OSU_pHRR_kW_m2": num(phrr), "OSU_THR_kW_min_m2": num(thr),
            "source_url": URL,
            "table_ref": f"{DOC}, Table 2 p. 6 (PDF p. 16) + Table 3 p. 9 (PDF p. 19), joined on (hardener, epoxy)",
            "T2_verbatim_row": row2, "T3_verbatim_row": row3, "note": NOTE.get((hard, epo), ""),
        })
    return recs


def check(recs):
    by = {(r["hardener"], r["epoxy"][-1]): r for r in recs}
    for key, (eta, loi) in SECOND_READ.items():
        assert by[key]["HRC_J_gK"] == eta and (by[key]["LOI_range"] or MISSING) == loi, key
    for hard in ("EMI-24", "TETA", "MDA"):
        a, c = by[(hard, "A")], by[(hard, "C")]
        assert c["HRC_J_gK"] < a["HRC_J_gK"] and c["hc_kJ_g_per_initial_mass"] < a["hc_kJ_g_per_initial_mass"], hard
    assert by[("EMI-24", "C")]["TGA_Tp_C"] == 420 and by[("MDA", "C")]["TGA_char_pct"] == 39


if __name__ == "__main__":
    recs = build()
    check(recs)
    out = f"{DATA_DIR}/loi_faa_ar0153.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    both = [r for r in recs if r["LOI_pct"] != "" and r["HRC_J_gK"] != ""]
    print(f"wrote {out}: {len(recs)} table rows; {len(both)} with LOI + eta_c: "
          + ", ".join(f"{r['polymer']} (LOI {r['LOI_range']}, eta_c {r['HRC_J_gK']:.0f})" for r in both))
