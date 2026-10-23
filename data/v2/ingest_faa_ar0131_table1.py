#!/usr/bin/env python3
"""Stage FAA DOT/FAA/AR-01/31 (Walters & Lyon, 2001) Table 1 -- polymer repeat-unit formula + PCFC values.

STAGING ONLY: nothing here is in any benchmark or number of record. Output:
  data/sources/faa_staging/ar0131_table1_pcfc.csv      (one row per polymer: a CAS number or the printed "[NA]")
  data/sources/faa_staging/ar0131_table1_unparsed.csv  (anchor rows where a field could not be found)
Each row keeps the text-layer lines it was read from, so every value can be audited against the PDF.

Source PDF (US-government work, public): https://rosap.ntl.bts.gov/view/dot/92724/dot_92724_DS1.pdf
saved by hand to data/sources/pdfs/faa/AR-01-31.pdf (gitignored); text via `pdftotext -layout`.

Parsing: a row is anchored on its CAS cell ("[25030-74-8]" or "[NA]"). The formula and the four numeric columns
(HR capacity, total HR, char %, MW) are taken from the anchor line when present, otherwise from the nearest line
within +-3 that holds no other anchor; `formula_src` / `values_src` record the offset (0 = same line).
Line numbers are 1-based lines of `pdftotext -layout` output split on "\\n" (form feeds stripped).

QA (printed, and written to the CSV): the repeat-unit formula's molar mass is recomputed with RDKit's periodic
table and compared with the table's printed MW column; |diff| > 0.5 g/mol is flagged. Rows hand-checked against
the page image are corrected in HAND_CHECKED below, with the reason, never silently.

Run: python3 data/v2/ingest_faa_ar0131_table1.py
"""
import re
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data/sources/pdfs/faa/AR-01-31.pdf"
OUT = ROOT / "data/sources/faa_staging/ar0131_table1_pcfc.csv"
PT = Chem.GetPeriodicTable()

ANCHOR = re.compile(r"\[(\d{2,7}-\d{2}-\d|NA)\]")
NUM = r"(\d+(?:\.\d+)?)"
# tail of a data row: HR capacity, total HR, char %, MW (all numeric, in this order)
TAIL = re.compile(rf"(?:^|\s){NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$")
FORMULA = re.compile(r"(?<![A-Za-z0-9])(C\d+H\d+(?:[A-Z][a-z]?\d*)*)(?![a-z])")
WINDOW = (1, -1, 2, -2, 3, -3)

# Report errata found by reading the page image (printed page numbers). Keyed by (cas, context fragment).
# `formula` is replaced only where the drawn structure AND the printed MW agree on the corrected formula;
# `formula_printed` always keeps what the report prints.
HAND_CHECKED = {
    ("25030-74-8", "Nylon 12"): dict(formula="C12H23NO", reason="p11: report prints C12H23O; drawn -(CH2)11-C(=O)-NH- and printed MW 197.32 give C12H23NO"),
    ("NA", "BPCCE"): dict(formula="C16H8N2O2Cl2", reason="p14: report prints C16H8O2Cl2; drawn structure has two -O-C#N groups and printed MW 331.16 give C16H8N2O2Cl2"),
    ("42955-03-3", "TORLON"): dict(formula="C22H14N2O3", reason="p14: report prints C15H8O3N2; drawn structure includes -C6H4-CH2-C6H4- and printed MW 354.36 give C22H14N2O3"),
    ("028064-14-4", "DEN-438"): dict(formula="C29H30O6", reason="p16: report prints C10H11O; drawn structure is a 3-ring glycidyl-ether novolac and printed MW 474.55 give C29H30O6"),
    ("NA", "Eypel-A"): dict(reason="p12: printed MW 259.24 equals C14H14PNO2; printed formula and drawn structure give C14H14PNO3 (275.24). "
                                   "Formula kept; MW is the erratum. HR capacity, total HR and char are per gram and unaffected"),
}
# A second CAS printed inside another row's cell (same polymer, same values): recorded as alt_cas, not a new row.
ALIASES = {"30944-92-4": ("173452-35-2", "p15: XU-371 (Ciba) printed in the Primaset PT-30 novolac cyanate ester row")}


def formula_mass(f):
    tot = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        tot += PT.GetAtomicWeight(el) * (int(n) if n else 1)
    return tot


def anchor_formula(line, after):
    """Formula cell right of the CAS cell: text up to the next 2+ space gap, single spaces removed ("C3 H 6")."""
    m = re.match(r"\s*(\S+(?: \S+)*)", line[after:])
    if m:
        f = m.group(1).replace(" ", "")
        toks = re.findall(r"([A-Z][a-z]?)(\d*)", f)
        if f.startswith("C") and "".join(a + b for a, b in toks) == f and all(PT.GetAtomicNumber(a) > 0 for a, _ in toks):
            return f
    return None


def find_near(lines, i, lo, hi, pattern, after=None):
    """Search the anchor line first (right of `after`), then neighbouring lines that hold no other anchor."""
    seg = lines[i][after:] if after is not None else lines[i]
    m = pattern.search(seg)
    if m:
        return m, 0
    for k in WINDOW:
        j = i + k
        if lo <= j < hi and not ANCHOR.search(lines[j]):
            m = pattern.search(lines[j])
            if m:
                return m, k
    return None, None


def main():
    txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    lines = [l.replace("\f", "") for l in txt.split("\n")]
    start = next(i for i, l in enumerate(lines) if "TABLE 1. POLYMER STRUCTURE" in l)
    end = next(i for i, l in enumerate(lines) if i > start and "TABLE 2" in l.upper() and "STRUCTURAL GROUPS" in l.upper())
    rows, unparsed = [], []
    for i in range(start, end):
        a = ANCHOR.search(lines[i])
        if not a:
            continue
        if a.group(1) in ALIASES:
            continue
        f0 = anchor_formula(lines[i], a.end())
        fm, fk = (None, 0) if f0 else find_near(lines, i, start, end, FORMULA, after=a.end())
        tm, tk = find_near(lines, i, start, end, TAIL)
        ctx = " / ".join(re.sub(r"\s{2,}", " | ", lines[j].strip()) for j in range(max(start, i - 2), min(end, i + 3)) if lines[j].strip())
        left = lines[i][:a.start()].strip()
        if not ((f0 or fm) and tm):
            unparsed.append(dict(line_no=i + 1, id=a.group(1), missing=("formula " if not fm else "") + ("values" if not tm else ""), context=ctx))
            continue
        rows.append(dict(
            line_no=i + 1, cas=a.group(1), name_partial=re.sub(r"^\d+\s+", "", re.sub(r"\s{2,}", " | ", left))[:70],
            formula=f0 or fm.group(1), formula_src=fk, hr_capacity_J_gK=float(tm.group(1)), total_hr_kJ_g=float(tm.group(2)),
            char_pct=float(tm.group(3)), mw_printed=float(tm.group(4)), values_src=tk, context=ctx))
    df = pd.DataFrame(rows)
    df.insert(df.columns.get_loc("formula"), "formula_printed", df.formula)
    df["report_erratum"] = ""
    df["alt_cas"] = ""
    for alt, (cas, why) in ALIASES.items():
        hit = df.cas == cas
        assert hit.sum() == 1, (alt, int(hit.sum()))
        df.loc[hit, "alt_cas"] = alt
    for (cas, frag), fix in HAND_CHECKED.items():
        hit = (df.cas == cas) & df.context.str.contains(frag, regex=False)
        assert hit.sum() == 1, (cas, frag, int(hit.sum()))
        for k, v in fix.items():
            if k != "reason":
                df.loc[hit, k] = v
        df.loc[hit, "report_erratum"] = fix["reason"]
    df["mw_from_formula"] = [round(formula_mass(f), 2) for f in df.formula]
    df["mw_diff"] = (df.mw_from_formula - df.mw_printed).round(2)
    df["qa_flag"] = ["report_erratum" if e else ("mw_mismatch" if abs(d) > 0.5 else "") for d, e in zip(df.mw_diff, df.report_erratum)]
    df["source"] = "Walters & Lyon, DOT/FAA/AR-01/31 (2001), Table 1"
    df["source_url"] = "https://rosap.ntl.bts.gov/view/dot/92724/dot_92724_DS1.pdf"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    na = (df.cas == "NA").sum()
    print(f"anchors: {len(df) + len(unparsed)} ({len(df)} parsed: {len(df) - na} with CAS, {na} printed [NA]); unparsed {len(unparsed)}")
    print(f"values from a neighbouring line: {(df.values_src != 0).sum()}; formula from a neighbouring line: {(df.formula_src != 0).sum()}")
    print(df.qa_flag.replace("", "ok").value_counts().to_string())
    print(df[df.qa_flag != ""][["line_no", "cas", "name_partial", "formula_printed", "formula", "mw_printed", "mw_from_formula", "qa_flag"]].to_string(index=False))
    print("values from a neighbouring line (check against the page image):")
    print(df[df.values_src != 0][["line_no", "cas", "name_partial", "values_src", "hr_capacity_J_gK", "total_hr_kJ_g", "char_pct", "mw_printed"]].to_string(index=False))
    print(f"wrote {OUT.relative_to(ROOT)}")
    pd.DataFrame(unparsed, columns=["line_no", "id", "missing", "context"]).to_csv(OUT.with_name("ar0131_table1_unparsed.csv"), index=False)
    for u in unparsed:
        print(f"  UNPARSED line {u['line_no']} [{u['id']}] missing {u['missing']}: {u['context'][:150]}")


if __name__ == "__main__":
    main()
