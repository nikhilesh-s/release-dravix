"""Shared data loaders and a vectorised CFRS scorer for the weights analyses (see DECLARATIONS.md).

score_matrix() matches cfrps.cfrps() exactly (same subscore(), same renormalisation over present
inputs); test_weights.py checks that.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from cfrps import PROPERTIES, subscore  # noqa: E402

PROPS = list(PROPERTIES)  # fixed column order: TTI, pHRR, THR, avg_HRR, char_residue, LOI
LIT = ROOT / "data/v2/cone_lit_benchmark_v1.csv"
LIT_COLS = {"tti_s": "TTI", "phrr_kWm2": "pHRR", "thr_MJm2": "THR", "avg_hrr_kWm2": "avg_HRR",
            "char_residue_pct": "char_residue", "loi_pct": "LOI"}

_FAIL = ("HB", "NR", "N.R", "NC", "N.C", "NORATING", "NOTRATED", "FAIL", "BURN", "BC", "F", "NOTCLASS")


def ul94_class(s) -> float:
    """Ordinal UL-94 class: V-0/5VA/5VB = 3, V-1 = 2, V-2 = 1, HB / no rating / fail = 0, else NaN.
    Multi-thickness strings ("V0 (3.2 mm); V2 (1.6 mm)") use the first listed (thickest, closest to
    cone specimens). Per-specimen strings ("F/F/V-2", "V-2/V-2/F") use the worst result."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    first = re.split(r"[;(]", s)[0]
    vals = []
    for part in first.split("/"):
        t = re.sub(r"[\s\-_]", "", part.upper())
        if not t:
            continue
        if t.startswith(("V0", "5VA", "5VB")):
            vals.append(3)
        elif t.startswith("V1"):
            vals.append(2)
        elif t.startswith("V2"):
            vals.append(1)
        elif t.startswith(_FAIL):
            vals.append(0)
        else:
            return np.nan
    return float(min(vals)) if vals else np.nan


def load_lit() -> pd.DataFrame:
    """Literature cone snapshot with CFRS-named property columns, paper id, heat flux, UL-94 class."""
    d = pd.read_csv(LIT)
    out = d[list(LIT_COLS)].rename(columns=LIT_COLS).astype(float)
    out = out.where(out > 0)  # same rule as the reference-range derivation (values > 0)
    out["char_residue"] = d["char_residue_pct"].where(d["char_residue_pct"] >= 0)  # 0 % char is real
    out["paper"] = d["group"].astype(str)
    out["heat_flux"] = d["heat_flux_kWm2"]
    out["ul94"] = d["ul94"].map(ul94_class)
    out["is_control"] = d["is_control"].fillna(0).astype(bool)
    out["label"] = d["formulation_label"].astype(str)
    return out.reset_index(drop=True)


def load_fsri(heat_flux: float = 50.0) -> pd.DataFrame:
    """FSRI materials at one heat flux: per-material means of replicate TTI, pHRR, THR, residue.
    avg_HRR and LOI are not in FSRI (left NaN; the score renormalises over present inputs)."""
    r = pd.read_csv(ROOT / "data/v2/cone_replicates_fsri.csv")
    r = r[(r.heat_flux_kWm2 == heat_flux) & ~r.did_not_ignite.astype(bool)]
    g = r.groupby("material").agg(TTI=("tti_s", "mean"), pHRR=("phrr_kWm2", "mean"),
                                  THR=("thr_MJm2", "mean"), char_residue=("residue_pct", "mean"))
    g["avg_HRR"] = np.nan
    g["LOI"] = np.nan
    return g[PROPS].reset_index()


def subscores(df: pd.DataFrame, props=PROPS) -> np.ndarray:
    """n x len(props) risk-oriented subscores in [0, 1]; NaN where the input is missing."""
    S = np.full((len(df), len(props)), np.nan)
    for j, k in enumerate(props):
        v = df[k].to_numpy(float)
        ok = ~np.isnan(v)
        S[ok, j] = subscore(k, v[ok])
    return S


def score_matrix(S: np.ndarray, w) -> np.ndarray:
    """CFRS (0-100) per row from subscores S (NaN = missing) and weights w (len = S.shape[1]),
    renormalised over present inputs as cfrps() does. Rows with no weighted input -> NaN."""
    w = np.asarray(w, float)
    present = ~np.isnan(S)
    num = np.where(present, S, 0.0) @ w
    den = present.astype(float) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 100.0 * num / den, np.nan)


def weights_vec(wdict: dict, props=PROPS) -> np.ndarray:
    return np.array([wdict.get(k, 0.0) for k in props], float)
