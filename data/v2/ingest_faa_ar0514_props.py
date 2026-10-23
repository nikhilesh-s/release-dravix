#!/usr/bin/env python3
"""Stage FAA DOT/FAA/AR-05/14 (Lyon & Janssens, Polymer Flammability, 2005) Appendix A tables that Dravix has not used:
  A-1  onset decomposition Td, peak mass-loss-rate Tp and ignition Tign temperatures (deg C, +-10)
  A-3  thermal conductivity k, density rho, heat capacity cp, thermal diffusivity alpha (room temperature)
  A-4  surface absorptivity (1000 K / 1500 K blackbody, flames, solar)

STAGING ONLY: nothing here is in any benchmark or number of record. Output:
  data/sources/faa_staging/ar0514_A1_temperatures.csv
  data/sources/faa_staging/ar0514_A3_thermal.csv
  data/sources/faa_staging/ar0514_A4_absorptivity.csv
Each row keeps its verbatim text-layer line and the printed appendix page.

Source PDF (US-government work, public): https://rosap.ntl.bts.gov/view/dot/16407/dot_16407_DS1.pdf
saved to data/sources/pdfs/faa/dot_16407_DS1.pdf (gitignored); text via `pdftotext -layout` (the .txt beside it).

QA: A-3 alpha is recomputed as k / (rho * cp) and compared with the printed alpha (flag > 5 %).
A-1 values are assigned to Td / Tp / Tign by column position under the page's header, so a blank cell
(e.g. EVA has no Tign) stays blank instead of shifting the next value left.

Run: python3 data/v2/ingest_faa_ar0514_props.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/sources/pdfs/faa/dot_16407_DS1.txt"
OUT = ROOT / "data/sources/faa_staging"
URL = "https://rosap.ntl.bts.gov/view/dot/16407/dot_16407_DS1.pdf"
SRC = "Lyon & Janssens, Polymer Flammability, DOT/FAA/AR-05/14 (FAA, May 2005)"
NUM = r"\d+(?:\.\d+)?"
DASH = "−"


def blocks(lines, title):
    """Yield (page_label, [lines]) for every page of an appendix table (title repeats on continuation pages)."""
    starts = [i for i, l in enumerate(lines) if l.strip().startswith(title)]
    for s in starts:
        e = next(j for j in range(s + 1, len(lines)) if re.fullmatch(r"\s*A-\d+\s*", lines[j]) or lines[j].strip().startswith("TABLE A-"))
        page = lines[e].strip() if re.fullmatch(r"\s*A-\d+\s*", lines[e]) else "?"
        yield page, lines[s:e]


def key(abbr):
    return re.sub(r"[^A-Z0-9/\-]", "", abbr.upper())


def parse_a1(lines):
    rows = []
    for page, blk in blocks(lines, "TABLE A-1."):
        hdr = next(l for l in blk if re.search(r"\bTd\s+Tp\s+Tign\b", l))
        abbr_at = hdr.index("Abbreviation")
        cols = {c: hdr.index(c) + len(c) / 2 for c in ("Td", "Tp", "Tign")}
        for l in blk:
            body = l[:abbr_at].strip()
            tail = l[abbr_at:]
            toks = [(m.group(), abbr_at + m.start() + len(m.group()) / 2) for m in re.finditer(rf"{NUM}|{DASH}", tail)]
            m = re.match(r"\s*(\S+(?: \S+)?)", tail)
            if not body or not toks or not m or l is hdr or "°C" in l or re.search(r"\d", m.group(1).replace("-G", "")) and not re.match(r"[A-Z]", m.group(1)):
                continue
            vals = {}
            for t, x in toks:
                c = min(cols, key=lambda c: abs(cols[c] - x))
                if abs(cols[c] - x) <= 4 and c not in vals:
                    vals[c] = np.nan if t == DASH else float(t)
            if not vals:
                continue
            rows.append(dict(abbr=m.group(1).strip(), name=body, Td_C=vals.get("Td", np.nan), Tp_C=vals.get("Tp", np.nan),
                             Tign_C=vals.get("Tign", np.nan), table_ref=f"{SRC}, Table A-1, p. {page}", source_url=URL,
                             verbatim_row=re.sub(r"\s{2,}", " | ", l.strip())))
    return pd.DataFrame(rows)


def parse_cols(lines, title, names):
    pat = re.compile(rf"^\s*(\S+(?: \S+)?)\s{{2,}}" + r"\s+".join(f"({NUM})" for _ in names) + r"\s*$")
    rows = []
    for page, blk in blocks(lines, title):
        for l in blk:
            m = pat.match(l)
            if m:
                rows.append(dict(abbr=m.group(1), **{n: float(m.group(k + 2)) for k, n in enumerate(names)},
                                 table_ref=f"{SRC}, Table {title.split()[1].rstrip('.')}, p. {page}", source_url=URL,
                                 verbatim_row=re.sub(r"\s{2,}", " | ", l.strip())))
    return pd.DataFrame(rows)


def main():
    lines = TXT.read_text().replace("\f", "\n").split("\n")
    a1 = parse_a1(lines)
    a3 = parse_cols(lines, "TABLE A-3.", ["k_W_mK", "rho_kg_m3", "cp_kJ_kgK", "alpha_m2s_e7"])
    a4 = parse_cols(lines, "TABLE A-4.", ["abs_bb1000K", "abs_bb1500K", "abs_flames", "abs_solar"])
    a3["alpha_from_k_rho_cp"] = (a3.k_W_mK / (a3.rho_kg_m3 * a3.cp_kJ_kgK * 1e3) * 1e7).round(2)
    a3["qa_flag"] = np.where((a3.alpha_from_k_rho_cp / a3.alpha_m2s_e7 - 1).abs() > 0.05, "alpha_mismatch", "")
    a3["krhoc_kJ2_m4K2s"] = (a3.k_W_mK * a3.rho_kg_m3 * a3.cp_kJ_kgK * 1e3 / 1e6).round(4)  # k*rho*c in (kW s^0.5/m2 K)^2 x 1e3
    for df in (a1, a3, a4):
        df.insert(1, "key", df.abbr.map(key))
    OUT.mkdir(parents=True, exist_ok=True)
    a1.to_csv(OUT / "ar0514_A1_temperatures.csv", index=False)
    a3.to_csv(OUT / "ar0514_A3_thermal.csv", index=False)
    a4.to_csv(OUT / "ar0514_A4_absorptivity.csv", index=False)
    print(f"A-1: {len(a1)} rows (Td {a1.Td_C.notna().sum()}, Tp {a1.Tp_C.notna().sum()}, Tign {a1.Tign_C.notna().sum()}); "
          f"duplicate abbreviations: {sorted(a1.key[a1.key.duplicated()].unique())}")
    print(f"A-3: {len(a3)} rows; alpha check: {(a3.qa_flag == '').sum()} ok, {(a3.qa_flag != '').sum()} flagged")
    if (a3.qa_flag != "").any():
        print(a3[a3.qa_flag != ""][["abbr", "k_W_mK", "rho_kg_m3", "cp_kJ_kgK", "alpha_m2s_e7", "alpha_from_k_rho_cp"]].to_string(index=False))
    print(f"A-4: {len(a4)} rows")
    print(f"A-1 keys with A-3 thermal properties: {a1.key.isin(a3.key).sum()}; with Tign and A-3: {(a1.Tign_C.notna() & a1.key.isin(a3.key)).sum()}")
    print("wrote", ", ".join(f"faa_staging/{n}" for n in ("ar0514_A1_temperatures.csv", "ar0514_A3_thermal.csv", "ar0514_A4_absorptivity.csv")))


if __name__ == "__main__":
    main()
