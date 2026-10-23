#!/usr/bin/env python3
"""QA gate for a dataset received from outside (Toldy/BME, Bussemaker/Surrey, ...). Read-only: writes nothing.

  python3 data/v2/check_incoming_dataset.py --kind pomazi   FILE.csv|xlsx [--sheet NAME]
  python3 data/v2/check_incoming_dataset.py --kind loi-rpuf FILE.csv|xlsx [--sheet NAME]
  python3 data/v2/check_incoming_dataset.py --self-test

It reports, it never fixes: schema (missing / unexpected columns, so you can see what needs a mapping),
value ranges and unit sanity, blank group keys, within-file duplicates, and overlap with rows already in the repo
(overlap = the rows the CV groups must be linked on, and the leakage risk if they are not). The exit code is 0 when
no ERROR is found, 1 otherwise. WARN lines are judgement calls for a human.

Schemas are the repo's own: pomazi_recon.csv (Pomazi reconstruction rows) and loi_rpuf_parts rows_wN.csv (LOI RPUF
staging rows; README in loi_rpuf_parts/).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/v2"

POMAZI_REQ = ["row_id", "source_key", "sample_label", "sample_type", "matrix_resin", "heat_flux_kWm2", "thickness_mm"]
POMAZI_TARGETS = {"LOI_pct": (10, 100), "tti_s": (1, 1500), "phrr_kWm2": (20, 3500), "thr_MJm2": (1, 300),
                  "residue_pct": (0, 90), "figra": (0.1, 60)}
POMAZI_INPUTS = {"C_pct": (0, 100), "H_pct": (0, 20), "O_pct": (0, 60), "N_pct": (0, 40), "P_pct": (0, 30),
                 "heat_flux_kWm2": (10, 100), "thickness_mm": (0.5, 30), "fibre_wtpct": (0, 100)}
POMAZI_KEY = ["matrix_resin", "hardener", "fr1_name", "fr1_loading", "fr2_name", "fr2_loading", "fibre_type",
              "fibre_wtpct", "heat_flux_kWm2", "thickness_mm"]

LOI_REQ = ["row_id", "ref_id", "source_doi", "formulation_label", "fr1_name", "fr1_loading", "fr1_unit", "loi_pct", "extraction"]
LOI_RANGES = {"loi_pct": (14, 60), "density_kgm3": (15, 400), "total_fr_wt_pct": (0, 70)}
LOI_CLASSES = {"P-additive", "P-reactive", "N-additive", "P-N", "intumescent(APP-based)", "expandable_graphite",
               "metal_hydroxide", "metal_oxide/salt", "silicon", "boron", "halogen", "nanoclay",
               "carbon_nano(graphene/CNT)", "bio-based", "other", ""}

CENSORED = r'^(>|<|>=|<=|no ignition|n\\.?i\\.?|not ignited|no ign|nb|n/a|na|-)'
errs, warns = [], []


def E(m):
    errs.append(m)
    print("ERROR", m)


def W(m):
    warns.append(m)
    print("WARN ", m)


def load(path, sheet=None):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0)
    return pd.read_csv(p)


def schema(df, req, ref_cols):
    miss = [c for c in req if c not in df.columns]
    extra = [c for c in df.columns if c not in ref_cols]
    if miss:
        E(f"missing required columns: {miss}")
    if extra:
        W(f"{len(extra)} columns not in the repo schema (need a mapping or are new): {extra[:15]}{' ...' if len(extra) > 15 else ''}")


def ranges(df, spec):
    for c, (lo, hi) in spec.items():
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        bad_type = df[c].notna() & x.isna()
        if bad_type.any():
            txt = df.loc[bad_type, c].astype(str).str.strip().str.lower()
            cens = txt.str.match(CENSORED)
            if cens.any():
                W(f"{c}: {int(cens.sum())} censored/qualitative values kept as text (e.g. {txt[cens].head(3).tolist()}); "
                  "they are NOT numbers: decide per target whether they are dropped or modelled as censored")
            if (~cens).any():
                E(f"{c}: {int((~cens).sum())} non-numeric values, e.g. {txt[~cens].head(3).tolist()}")
        out = x.notna() & ((x < lo) | (x > hi))
        if out.any():
            W(f"{c}: {int(out.sum())} values outside [{lo}, {hi}] (unit mix-up? e.g. s vs min, fraction vs %): {x[out].head(5).tolist()}")
        print(f"  {c}: n={int(x.notna().sum())}/{len(df)} min={x.min():.4g} median={x.median():.4g} max={x.max():.4g}")


def check_pomazi(df):
    ref = pd.read_csv(V2 / "pomazi_recon.csv")
    schema(df, POMAZI_REQ, ref.columns)
    ranges(df, {**POMAZI_TARGETS, **POMAZI_INPUTS})
    if "UL94" in df.columns:
        W("UL94 coding: repo uses 1=HB,2=V-2,3=V-1,4=V-0 (thesis mapping); confirm what the sender used")
    if "residue_pct" in df.columns and pd.to_numeric(df["residue_pct"], errors="coerce").max() <= 1.0:
        W("residue_pct looks like a fraction (max <= 1); repo uses percent")
    if "tti_s" in df.columns and "heat_flux_kWm2" in df.columns:
        n = df["tti_s"].isna().sum()
        print(f"  rows without TTI: {n}")
    k = [c for c in POMAZI_KEY if c in df.columns and c in ref.columns]
    if len(k) >= 5:
        dup = df.duplicated(k, keep=False)
        if dup.any():
            W(f"{int(dup.sum())} rows share the same formulation+test key within the file (replicates, or duplicates?)")
        m = df[k].astype(str).merge(ref[k].astype(str).drop_duplicates(), on=k, how="inner")
        print(f"  overlap with pomazi_recon.csv on {len(k)} key columns: {len(m)} rows")
        if len(m):
            W("overlap with the reconstruction: these rows must share a CV group with their reconstruction counterparts")
    else:
        W(f"cannot test overlap: only {len(k)} key columns shared with the repo schema")
    if "source_key" in df.columns and df["source_key"].isna().any():
        E("blank source_key on some rows (needed to build CV groups)")
    print("  NOTE: rows from a different test protocol (flux, thickness, orientation) are a different population; do not pool silently")


def check_loi(df):
    ref_cols = pd.read_csv(sorted(glob.glob(str(V2 / "loi_rpuf_parts/wave4_staging/rows_w*.csv")))[0], nrows=1).columns
    schema(df, LOI_REQ, ref_cols)
    ranges(df, LOI_RANGES)
    for c in ("fr1_class", "fr2_class", "fr3_class"):
        if c in df.columns:
            bad = set(df[c].fillna("")) - LOI_CLASSES
            if bad:
                W(f"{c}: classes outside the repo vocabulary: {sorted(bad)[:8]} (map before use)")
    if "fr1_unit" in df.columns:
        print(f"  fr1_unit values: {df['fr1_unit'].fillna('').value_counts().head(8).to_dict()}")
    if "ref_id" in df.columns and df["ref_id"].isna().any():
        E("blank ref_id (source paper) on some rows; group-by-paper CV needs it on every row")
    if "extraction" in df.columns:
        fig = (df["extraction"] == "figure").sum()
        if fig:
            W(f"{int(fig)} figure-read rows: kept out of the benchmark of record by repo rule")
    if "loi_pct" in df.columns and "row_id" in df.columns and df.duplicated("row_id").any():
        E("duplicate row_id")
    # overlap with the benchmark snapshot, by DOI + label
    lit = pd.read_csv(V2 / "loi_rpuf_literature.csv")
    parts = [pd.read_csv(f) for f in glob.glob(str(V2 / "loi_rpuf_parts/rows_w*.csv"))]
    old = pd.concat(parts) if parts else pd.DataFrame()
    if "source_doi" in df.columns:
        dois = set(df["source_doi"].dropna().str.lower())
        known = set(lit["source_doi"].dropna().str.lower()) | (set(old["source_doi"].dropna().str.lower()) if len(old) else set())
        both = dois & known
        print(f"  papers in file: {len(dois)}; already in the repo's LOI rows: {len(both)}; new: {len(dois - known)}")
        if both:
            W(f"{len(both)} papers already in the benchmark (their rows are duplicates or replicates; the CV group is the paper)")
    if "in_sunfu_refs" in df.columns:
        print(f"  in_sunfu_refs: {df['in_sunfu_refs'].value_counts(dropna=False).to_dict()}")


def self_test():
    print("== self-test 1: pomazi_recon.csv against itself (expect: every row overlaps, no errors)")
    check_pomazi(pd.read_csv(V2 / "pomazi_recon.csv"))
    a = len(errs)
    print("\n== self-test 2: LOI wave-4 staging rows (expect: schema ok, no errors)")
    check_loi(pd.read_csv(sorted(glob.glob(str(V2 / "loi_rpuf_parts/wave4_staging/rows_w*.csv")))[0]))
    print("\n== self-test 3: broken file (expect errors: missing columns, non-numeric, duplicate ids)")
    bad = pd.DataFrame({"row_id": ["a", "a"], "loi_pct": ["22", "twenty"], "ref_id": [None, "X"]})
    n0 = len(errs)
    check_loi(bad)
    ok = a == 0 and len(errs) - n0 >= 3
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'} (errors on good files: {a}; errors on the broken file: {len(errs) - n0})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pomazi", "loi-rpuf"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--sheet")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not (a.kind and a.file):
        ap.error("--kind and FILE are required (or --self-test)")
    df = load(a.file, a.sheet)
    print(f"{a.file}: {len(df)} rows x {len(df.columns)} columns")
    (check_pomazi if a.kind == "pomazi" else check_loi)(df)
    print(f"\n{len(errs)} errors, {len(warns)} warnings")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
