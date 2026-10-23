#!/usr/bin/env python3
"""Dravix V2 -- LOI from DOT/FAA/AR-99/56 Table 2, joined by exact name to eta_c from DOT/FAA/AR-01/31 Table 1.

WHY: AR-99/56 (Lyon 1999) prints LOI for 34 polymers but no heat release capacity (eta_c);
AR-01/31 (Walters & Lyon 2001) prints eta_c for 86 polymers but no LOI. The lead approved ONE
cross-report join (sprint 0929, round 3) to see whether it adds new chemistries to the 38-row
LOI set (loi_faa_polymers.csv + loi_faa_ar0153.csv). LOI and eta_c here come from different
documents and possibly different specimens, so every joined row carries
    pairing = "cross-report exact-name (AR-99/56 LOI + AR-01/31 HRC)".

Transcription. Both tables are kept as verbatim text-layer strings (read 2026-09-29 from the
ROSA P PDFs below via the fetch tool; nothing saved in the repo). Every numeric column is parsed
from those strings, so the values cannot disagree with the provenance columns.
  - T2 (AR-99/56, PDF p. 40, report p. 26) is an OCR text layer over a scan. Its 34 rows were
    checked against (a) pdfminer's extraction of the same page and (b) the rendered page image:
    all numbers agree. OCR damaged three NAMES, kept verbatim here: "(Pl)" is "(PI)" on the scan,
    "(P A6)" is "(PA6)", "Poly( a-methylstyrene)" is "Poly(alpha-methylstyrene)". A stray "•" (a
    speck on the scan) before BCE's 470 is also verbatim. None of these changes a join.
  - T1 (AR-01/31, PDF pp. 13-24, report pp. 6-17) is a born-digital text layer on rotated pages.
    Row/column cells came from pdfplumber's ruled-table extraction; every "HRC THR char MW" block
    also appears in pypdf's text of the same page (whitespace-insensitive). Name cells are in
    pypdf reading order. Only the name column and the four value columns are kept.

Join rule (fixed before looking at which rows match):
  name = the chemical name; abbreviation = AR-99/56's trailing "(ABBR)" / AR-01/31's trailing
  all-caps token. Join iff the chemical names are identical after removing whitespace and
  case-folding, AND the abbreviations agree the same way where both tables print one, AND
  exactly one AR-01/31 row qualifies. Anything else -> HRC blank and a match_note saying why.
  Spelling variants (sulfone/sulphone), word order, locants ("1,4-") and extra words are NOT
  formatting, so they do not join; NEAR lists the obvious ones for the lead, unjoined.

"New" = not already among the 38 existing LOI polymers. Same polymer under another label counts
as existing: VARIANT maps AR-99/56's label to the existing name (these three were flagged in
reports/sprint_0929/loi_followup.md section 3.5 before this join was run).

Rerun: python3 ingest_faa_loi_ar9956.py  -> loi_faa_ar9956_ar0131.csv
"""
import csv
import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URL_T2 = "https://rosap.ntl.bts.gov/view/dot/92730/dot_92730_DS1.pdf"
URL_T1 = "https://rosap.ntl.bts.gov/view/dot/92724/dot_92724_DS1.pdf"
DOC_T2 = "Lyon, Solid-State Thermochemistry of Flaming Combustion, DOT/FAA/AR-99/56 (1999)"
DOC_T1 = "Walters & Lyon, Calculating Polymer Flammability from Molar Group Contributions, DOT/FAA/AR-01/31 (2001)"
PAIRING = "cross-report exact-name (AR-99/56 LOI + AR-01/31 HRC)"

# AR-99/56 Table 2: PEAK PYROLYSIS TEMPERATURE (TP), PYROLYSIS RESIDUE(u), LIMITING OXYGEN INDEX (LOI),
# AND UL 94 RANKING OF SOME POLYMERS (Values in parentheses are calculated estimates.)
# Printed columns: Polymer | Tp (C) | u (%) | LOI (%) | UL94 Ranking. Report p. 26 = PDF p. 40.
T2 = [
    "Polybenzobisoxazole (PBO) 789 75 56 V-0",
    "Polyparaphenylene 652 75 55 (V-0)",
    "Polybenzimidazole (PBI) 630 70 42 V-0",
    "Polyamideimide (PAI) 628 55 45 V-0",
    "Polyaramide (KEVLAR) 628 43 28 V-0",
    "Polyetherketoneketone (PEKK) 619 62 40 V-0",
    "Polyetherketone (PEK) 614 56 40 V-0",
    "Polytetrafluoroethylene (PTFE) 612 0 95 V-0",
    "Polyetheretherketone (PEEK) 606 50 35 V-0",
    "Polyphenylsulfone (PPSF) 606 44 38 V-0",
    "Polypara(benzoyl)phenylene (PX) 602 66 41 V-0",
    "Fluorinated Cyanate Ester 583 44 (40) V-0",
    "Polyphenylenesulfide (PPS) 578 45 44 V-0",
    "Polyetherimide (PEI) 575 52 47 V-0",
    "Polypyromellitimide (Pl) 567 70 37 V-0",
    "Liquid Crystal Polyester 564 38 40 V-0",
    "Polycarbonate (PC) 546 25 26 V-2",
    "Polysulfone (PSF) 537 30 30 V-1",
    "Polyethylene (PE) 505 0 18 HB",
    "Polyamide 6 (P A6) 497 1 21 HB",
    "Polyethylenenapthalate (PEN) 495 24 32 V-2",
    "Polyphthalamide 488 3 (22) HB",
    "Phenolic Triazine Cyanate Ester (PT) 480 62 30 V-0",
    "Polyethyleneterephthalate (PET) 474 13 21 HB",
    "Cyanate ester of Bisphenol-A (BCE) • 470 33 24 V-1",
    "Polydimethylsiloxane (PDMS) 444 0 30 HB",
    "Acrylonitrile-butadiene-styrene (ABS) 444 0 18 HB",
    "Polyurethane elastomer (PU) 422 3 17 HB",
    "Polymethylmethacrylate (PMMA) 398 2 17 HB",
    "Polychlorotrifluoroethylene 380 0 95 V-0",
    "Polystyrene (PS) 364 0 18 HB",
    "Polyoxymethylene (POM) 361 0 15 HB",
    "Poly( a-methylstyrene) 341 0 18 HB",
    "Polyvinylchloride (PVC) 270 11 50 V-0",
]

# AR-01/31 Table 1: POLYMER STRUCTURE AND VALUES DERIVED FROM PYROLYSIS-COMBUSTION FLOW CALORIMETRY.
# (PDF page, "Material and Abbreviated Name" cell in reading order, "HR Capacity (J/g-K) | Total HR (kJ/g) |
# Char (%) | MW (g/mol)"). Report page = PDF page - 7. All 86 rows, so the join is checked against every name.
T1 = [
    (13, "Polyethylene PE", "1676 41.6 0 28.06"),
    (13, "Polyoxymethylene POM", "169 14 0 30.03"),
    (13, "Polypropylene PP", "1571 41.4 0 42.08"),
    (13, "Polyvinylalcohol (≥99%) PVOH", "533 21.6 3.3 44.03"),
    (13, "Polyethyleneoxide", "652 21.6 1.7 44.05"),
    (13, "Polyisobutylene", "1002 44.4 0 56.11"),
    (13, "Polyvinylchloride", "138 11.3 15.3 62.48"),
    (13, "Polyvinylidene fluoride", "311 9.7 7 64.02"),
    (13, "Polyacrylamide", "104 13.3 8.3 71.08"),
    (13, "Polyacrylic Acid", "165 12.5 6.1 72.06"),
    (14, "Polyvinylacetate PVAc", "313 19.2 1.2 86.09"),
    (14, "Polymethacrylic Acid", "464 18.4 0.5 86.09"),
    (14, "Polychloroprene", "188 16.1 12.9 88.54"),
    (14, "Polytetrafluoro ethylene PTFE", "35 3.7 0 100.02"),
    (14, "Polymethyl methacralate PMMA", "514 24.3 0 100.12"),
    (14, "Polymethyl methacralate PMMA", "461 23.2 0 100.12"),
    (14, "Polyethylacrylate", "323 22.6 0.3 100.12"),
    (15, "Polymethacrylamide", "103 18.7 4.5 101.1"),
    (15, "Polystyrene PS", "927 38.8 0 104.15"),
    (15, "Isotactic Polystyrene", "880 39.9 0 104.15"),
    (15, "Poly-2-vinylpyridene", "612 34.7 0 105.14"),
    (15, "Poly-4-vinylpyridene", "568 31.7 0 105.14"),
    (15, "Poly-1,4-Phenylene Sulfide PPS", "165 17.1 41.6 108.16"),
    (15, "Poly-n- vinylpyrrolidone", "332 25.1 0 111.14"),
    (16, "Polycaprolactam", "487 28.7 0 113.16"),
    (16, "Polycaprolactone", "526 24.4 0 114.14"),
    (16, "Polyethyl methacrylate", "470 26.4 0 114.14"),
    (16, "Polyethyl methacrylate", "380 26.8 0 114.14"),
    (16, "Poly α Methyl styrene", "730 35.5 0 118.18"),
    (16, "Poly-2,6-dimethyl- 1,4-phenyleneoxide PPO", "409 20 25.5 120.15"),
    (17, "Poly-4-vinylphenol", "261 27.6 2.8 120.15"),
    (17, "Polyethylenemaleic anhydride", "138 12.1 2.8 126.11"),
    (17, "Polyvinylbutyral", "806 26.9 0.1 142.1"),
    (17, "Poly-2- vinylnaphthalene", "834 39 0 154.21"),
    (17, "Polybenzoyl-1,4- Phenylene", "41 10.9 65.2 180.21"),
    (17, "Polyethylene Terephthalate PET", "332 15.3 5.1 192.17"),
    (18, "Polyetherketone PEK", "124 10.8 52.9 196.2"),
    (18, "Polylaurolactam", "743 33.2 0 197.32"),
    (18, "Polystyrene maleicanhydride", "279 23.3 2.2 202.21"),
    (18, "Polyacrylonitrile Butadiene-Styrene ABS", "669 36.6 0 211.31"),
    (18, "Poly-1,4-butanediol Terephthalate PBT", "474 20.3 1.5 220.22"),
    (18, "Polyhexamethylene Adiapamide", "615 27.4 0 226.32"),
    (18, "Polyazomethine", "36 8.7 77.8 231.26"),
    (18, "Poly-1,4-Phenylene Ethersulfone PES", "115 11.2 29.3 232.26"),
    (19, "Poly-p-Phenylene Benzobisoxazole PBO", "42 5.4 69.5 234.21"),
    (19, "Poly(p-phenylene Terephthalamide)", "302 14.8 36.1 238.25"),
    (19, "Poly(m-Phenylene Isophthalamide)", "52 11.7 48.4 238.25"),
    (19, "Polyethylene naphthylate PEN", "309 16.8 18.2 242.23"),
    (19, "Dicyclopentadienyl Bisphenol", "493 20.1 27.1 251.32"),
    (19, "Polycarbonate of Bisphenol-A PC", "359 16.3 21.7 254.28"),
    (19, "Polyphosphazine", "204 21.9 20 259.24"),
    (20, "Polydiphenylether Chloral", "16 5.2 57.1 263.12"),
    (20, "Cyano-Substituted Kevlar", "54 9.1 58.3 263.26"),
    (20, "Bisphenol-E Cyanate Ester", "316 14.7 41.9 264.28"),
    (20, "Bisphenol-A Cyanate Ester", "283 17.6 36.3 278.31"),
    (20, "Polyhexamethylene Sebacamide", "878 35.7 0 282.43"),
    (20, "Polyetheretherketone PEEK", "155 12.4 46.5 288.3"),
    (20, "PSA", "119 15.7 60.1 294.42"),
    (20, "Polyetherketone ketone PEKK", "96 8.7 60.7 300.31"),
    (20, "Tetramethylbisphenol- F Cyanate Ester", "280 17.4 35.4 306.36"),
    (21, "Bisphenol-C Polycarbonate", "29 3.0 50.1 307.13"),
    (21, "Polybenzimidazole PBI", "36 8.6 67.5 308.34"),
    (21, "Polyhexamethylene Dodecanediamide", "707 30.8 0 310.48"),
    (21, "Bisphenol-C Cyanate Ester", "24 4.2 53.3 331.16"),
    (21, "Bisphenol-A Epoxy, Catalytic cure Phenoxy-A", "657 26.0 3.9 340.42"),
    (21, "Phenolphthalein Polycarbonate", "28 8 49.8 344.32"),
    (21, "Polyamideimide PAI", "33 7.1 53.6 354.36"),
    (22, "Novolac Cyanate Ester", "122 9.9 51.9 381.39"),
    (22, "Polyimide PI", "25 6.6 51.9 382.33"),
    (22, "Hexafluorobisphenol- A Cyanate Ester", "32 2.3 55.2 386.25"),
    (22, "Bisphenol-C Epoxy", "506 10 36 393.26"),
    (22, "Bisphenol-M Cyanate Ester", "239 22.5 26.4 396.49"),
    (22, "Polyphenylsulphone", "153 11.3 38.4 400.45"),
    (22, "Bisphenol-C Polyarylate", "21 7.6 42.7 411.02"),
    (22, "Biphenol Phthalonitrile", "15 3.5 78.8 438.44"),
    (22, "Polysulfone of Bisphenol-A PSF", "345 19.4 28.1 442.53"),
    (23, "LaRC-1A", "38 6.7 57 474.43"),
    (23, "Epoxy Novolac, Catalytic Cure Phenoxy-N", "246 18.9 15.9 474.55"),
    (23, "Bisphenol-A Phthalonitrile", "40 5.9 73.6 480.52"),
    (23, "Technora", "131 15.3 41.8 568.59"),
    (23, "Bisphenol-A6F Phthalonitrile", "9 2.8 63.8 588.46"),
    (23, "Polyetherimide PEI", "121 11.8 49.2 592.61"),
    (23, "Polyester of Hydroxybenzoic and Hydroxynapthoic Acids", "164 11.1 40.6 650.6"),
    (23, "TOR", "135 11.7 63 692.71"),
    (23, "LaRC-CP2", "14 3.4 57 700.55"),
    (24, "LaRC-CP1", "13 2.9 52 926.66"),
]

VARIANT = {"PVC": "PVC (rigid)", "PPSF": "PPSU", "P A6": "PA6"}  # AR-99/56 label -> existing 38-row name
NEAR = {  # likely the same polymer in AR-01/31 under a different name: recorded, NOT joined
    "Polyphenylsulfone": "Polyphenylsulphone (spelling)",
    "Poly( a-methylstyrene)": "Poly α Methyl styrene (OCR 'a' for alpha; parentheses/hyphen differ)",
    "Polyaramide": "Poly(p-phenylene Terephthalamide) (trade name Kevlar)",
    "Polybenzobisoxazole": "Poly-p-Phenylene Benzobisoxazole PBO (extra words)",
    "Polypara(benzoyl)phenylene": "Polybenzoyl-1,4- Phenylene (locant, parentheses)",
    "Polyphenylenesulfide": "Poly-1,4-Phenylene Sulfide PPS (locant)",
    "Polysulfone": "Polysulfone of Bisphenol-A PSF (extra words)",
    "Polycarbonate": "Polycarbonate of Bisphenol-A PC (extra words)",
    "Cyanate ester of Bisphenol-A": "Bisphenol-A Cyanate Ester (word order)",
    "Fluorinated Cyanate Ester": "Hexafluorobisphenol- A Cyanate Ester (different name)",
    "Polypyromellitimide": "Polyimide PI (generic name)",
    "Polyamide 6": "Polycaprolactam (other name)",
    "Polyethylenenapthalate": "Polyethylene naphthylate PEN (spelling)",
    "Polymethylmethacrylate": "Polymethyl methacralate PMMA x2 (spelling as printed; two rows)",
    "Acrylonitrile-butadiene-styrene": "Polyacrylonitrile Butadiene-Styrene ABS (extra 'Poly')",
    "Liquid Crystal Polyester": "Polyester of Hydroxybenzoic and Hydroxynapthoic Acids (Vectra C LCP)",
}
ROW2 = re.compile(r"^(?P<name>.+?) (?:• )?(?P<tp>\d{3}) (?P<mu>\d+) (?P<loi>\(?\d+\)?) (?P<ul>\(?V-\d\)?|HB)$")
norm = lambda s: re.sub(r"\s+", "", s).casefold() if s else None


def split2(name):
    m = re.match(r"^(.*\S) \(([^()]+)\)$", name)  # "Poly( a-methylstyrene)" has no " (" -> no abbreviation
    return (m.group(1), m.group(2)) if m else (name, None)


def split1(name):
    tok = name.split(" ")
    if len(tok) > 1 and re.fullmatch(r"[A-Z]{2,}[A-Za-z0-9]*", tok[-1]):
        return " ".join(tok[:-1]), tok[-1]
    return name, None


def existing38():
    names = {}
    for f in ("loi_faa_polymers.csv", "loi_faa_ar0153.csv"):
        with open(f"{DATA_DIR}/{f}", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["LOI_pct"] and r["HRC_J_gK"]:
                    names[r["polymer"]] = (f, float(r["LOI_pct"]), float(r["HRC_J_gK"]))
    assert len(names) == 38, len(names)
    return names


def build():
    t1 = [(p, n, v) + split1(n) for p, n, v in T1]
    ex = existing38()
    recs = []
    for i, row in enumerate(T2, start=1):
        m = ROW2.match(row)
        assert m, row
        chem, abbr = split2(m["name"])
        loi = m["loi"]
        hits = [t for t in t1 if norm(t[3]) == norm(chem) and (abbr is None or t[4] is None or norm(t[4]) == norm(abbr))]
        key = abbr or chem
        exist = key if key in ex else VARIANT.get(key) if VARIANT.get(key) in ex else ""
        rec = {"table_row": i, "polymer": m["name"], "abbr": abbr or "",
               "LOI_pct": float(loi.strip("()")), "LOI_is_calc_estimate": loi.startswith("("),
               "UL94_rating": m["ul"], "Tp_C": float(m["tp"]), "char_pct_ar9956": float(m["mu"]),
               "HRC_J_gK": "", "THR_kJ_g": "", "char_pct_ar0131": "", "MW_g_mol": "", "pairing": "",
               "existing_row": exist, "new_vs_existing38": "",
               "match_note": "", "source_url": URL_T2, "table_ref": f"{DOC_T2}, Table 2, p. 26 (PDF p. 40)",
               "verbatim_row": row, "T1_source_url": "", "T1_ref": "", "T1_verbatim_name": "", "T1_verbatim_values": ""}
        if len(hits) == 1:
            p, n, v, _, _ = hits[0]
            hrc, thr, ch, mw = v.split(" ")
            rec.update(HRC_J_gK=float(hrc), THR_kJ_g=float(thr), char_pct_ar0131=float(ch), MW_g_mol=float(mw),
                       pairing=PAIRING, new_vs_existing38=not exist, T1_source_url=URL_T1,
                       T1_ref=f"{DOC_T1}, Table 1, p. {p - 7} (PDF p. {p})", T1_verbatim_name=n, T1_verbatim_values=v,
                       match_note="joined" + ("" if hits[0][4] else "; AR-01/31 name cell prints no abbreviation"))
        else:
            same_abbr = [t[1] for t in t1 if abbr and t[4] and norm(t[4]) == norm(abbr)]
            why = (f"{len(hits)} AR-01/31 rows qualify (ambiguous)" if hits else
                   f"no AR-01/31 name equals '{chem}' after whitespace/case normalisation")
            if same_abbr:
                why += f"; AR-01/31 prints abbreviation {abbr} on: {' | '.join(same_abbr)}"
            if chem in NEAR:
                why += f"; likely same polymer, not joined: {NEAR[chem]}"
            rec["match_note"] = why
        recs.append(rec)
    return recs, ex


if __name__ == "__main__":
    recs, ex = build()
    assert len(recs) == 34 and len(T1) == 86
    j = [r for r in recs if r["pairing"]]
    pbi = next(r for r in recs if r["abbr"] == "PBI")
    assert (pbi["LOI_pct"], pbi["HRC_J_gK"], pbi["existing_row"]) == (42.0, 36.0, "PBI")  # read off both pages
    out = f"{DATA_DIR}/loi_faa_ar9956_ar0131.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    new = [r for r in j if r["new_vs_existing38"]]
    print(f"wrote {out}: {len(recs)} AR-99/56 rows; {len(j)} joined to AR-01/31 eta_c; "
          f"{len(new)} of those new vs the 38 existing: {[r['polymer'] for r in new]}")
    print("\njoined rows:")
    for r in j:
        print(f"  {r['polymer']:32} LOI {r['LOI_pct']:5}  HRC {r['HRC_J_gK']:6}  <- '{r['T1_verbatim_name']}' "
              f"[{r['T1_verbatim_values']}]  existing: {r['existing_row'] or '-'}")
    print("\nconflicts vs the existing 38 (AR-99/56 LOI vs existing LOI; AR-01/31 HRC vs existing HRC):")
    dl = []
    for r in recs:
        if r["existing_row"]:
            f, loi0, hrc0 = ex[r["existing_row"]]
            d = r["LOI_pct"] - loi0
            dl.append(abs(d))
            hrc = (f"HRC {r['HRC_J_gK']:.0f} vs {hrc0:.0f} (x{r['HRC_J_gK'] / hrc0:.2f})" if r["HRC_J_gK"] != ""
                   else "HRC not joined")
            print(f"  {r['polymer']:32} vs {r['existing_row']:12} LOI {r['LOI_pct']:4.0f} vs {loi0:4.0f} "
                  f"(diff {d:+.0f})  {hrc}")
    print(f"  LOI: {len(dl)} overlaps, {sum(x == 0 for x in dl)} identical, mean |diff| {sum(dl) / len(dl):.2f}, "
          f"max {max(dl):.0f}")
    print("\nnot joined:")
    for r in recs:
        if not r["pairing"]:
            print(f"  {r['polymer']:38} {r['match_note']}")
