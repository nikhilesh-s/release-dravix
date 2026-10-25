"""D4 external checks (descriptive only, no fitting): see DECLARATIONS.md. Writes results/external_checks.json.
Needs results/calibration.json (run calibrate.py first) for the fitted weights."""
from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from calibrate import EQ, V0, concordance
from common import HERE, PROPS, ROOT, load_lit, score_matrix, subscores, ul94_class, weights_vec

FITTED = weights_vec(json.loads((HERE / "results/calibration.json").read_text())["D1"]["fitted_weights"])
W = {"v0": V0, "eq": EQ, "fitted": FITTED}


def a_indices():
    """Per paper+flux with a neat control: Kendall tau of CFRS vs Vahabi FRI and Petrella FPI."""
    df = load_lit()
    for k, w in W.items():
        df["cfrs_" + k] = score_matrix(subscores(df), w)
    df["prod"] = df.THR * df.pHRR / df.TTI
    df["FPI"] = df.TTI / df.pHRR
    rows = []
    for (paper, hf), g in df.groupby(["paper", "heat_flux"]):
        ctl = g[g.is_control & g["prod"].notna()]
        if ctl.empty:
            continue
        g = g.assign(FRI=ctl["prod"].mean() / g["prod"])
        g = g[g.FRI.notna() & g.FPI.notna() & g.cfrs_v0.notna()]
        if len(g) < 3:
            continue
        r = {"paper": paper, "hf": hf, "n": len(g)}
        for idx in ("FRI", "FPI"):
            for k in W:
                r[f"{idx}_{k}"] = kendalltau(g["cfrs_" + k], g[idx])[0]
        rows.append(r)
    t = pd.DataFrame(rows)
    out = {"n_groups": len(t), "n_rows": int(t.n.sum()),
           "note": "FRI/FPI higher = better, so negative tau = agreement; both are functions of the same cone inputs (consistency check)."}
    for c in t.columns[3:]:
        v = t[c].dropna()
        out[c] = {"n_groups": len(v), "mean_tau": float(v.mean()), "median_tau": float(v.median()),
                  "share_tau_lt_0": float((v < 0).mean()),
                  "row_weighted_mean_tau": float(np.average(v, weights=t.loc[v.index, "n"]))}
    return out


def _cls(s):
    s = re.sub(r"[()]", "", str(s)).strip()
    return 0.0 if s == "B" else ul94_class(s)  # AR-01/53 "B" = burns


def _pairwise(risk, y):
    """C of a risk index vs UL-94 class over all pairs of rows with different class (better = higher class)."""
    ok = ~(np.isnan(risk) | np.isnan(y))
    risk, y = risk[ok], y[ok]
    pr = [(a, b) if y[a] > y[b] else (b, a) for a, b in itertools.combinations(range(len(y)), 2) if y[a] != y[b]]
    return {"C": concordance(risk, np.array(pr)) if pr else None, "n_pairs": len(pr), "n_rows": int(ok.sum())}


def b_faa():
    out = {"note": "Pairs are across polymers (no within-paper control); single small-scale index vs UL-94 class."}
    d = pd.read_csv(ROOT / "data/v2/loi_faa_polymers.csv"); y = d.UL94_rating.map(_cls).to_numpy(float)
    out["polymers_AR0514"] = {
        "LOI_subscore": _pairwise(subscores(d.assign(LOI=d.LOI_pct), ["LOI"])[:, 0], y),
        "char_subscore": _pairwise(subscores(d.assign(char_residue=100 * d.char_yield_g_g), ["char_residue"])[:, 0], y),
        "HRC_raw_lyon": _pairwise(d.HRC_J_gK.to_numpy(float), y)}
    d = pd.read_csv(ROOT / "data/v2/loi_faa_ar9956_ar0131.csv"); y = d.UL94_rating.map(_cls).to_numpy(float)
    out["ar9956_ar0131"] = {
        "LOI_subscore": _pairwise(subscores(d.assign(LOI=d.LOI_pct), ["LOI"])[:, 0], y),
        "char_subscore": _pairwise(subscores(d.assign(char_residue=d.char_pct_ar9956), ["char_residue"])[:, 0], y),
        "HRC_raw_lyon": _pairwise(d.HRC_J_gK.to_numpy(float), y)}
    d = pd.read_csv(ROOT / "data/v2/loi_faa_ar0153.csv"); y = d.UL94_rating.map(_cls).to_numpy(float)
    x = pd.DataFrame({"pHRR": d.OSU_pHRR_kW_m2, "THR": d.OSU_THR_kW_min_m2 * 0.06,  # kW min/m2 -> MJ/m2
                      "char_residue": 100 * d.char_yield_g_g, "LOI": d.LOI_pct}).reindex(columns=PROPS)
    S = subscores(x)
    out["ar0153_epoxies"] = {"n_rows": len(d), "note": "OSU pHRR (30-144 kW/m2) is mostly below the cone pHRR range (75-1300), so its subscore saturates at 0; no TTI/avg_HRR.",
                             **{k: _pairwise(score_matrix(S, w), y) for k, w in W.items()},
                             "LOI_alone": _pairwise(S[:, 5], y), "HRC_raw_lyon": _pairwise(d.HRC_J_gK.to_numpy(float), y)}
    return out


C = {
    "MIL-STD-2031(SH)": {
        "criteria": "Navy submarine interior composites: flame spread (ASTM E162), smoke (ASTM E662), cone calorimeter (ASTM E1354) "
                    "time to ignition, peak and average HRR and gas generation, at several fluxes up to 75-100 kW/m2, plus burn-through "
                    "and larger scale tests. Pass/fail numbers could not be confirmed from an open copy in this search (not recorded here).",
        "sources": ["Sorathia et al., NIST FireDoc https://firedoc.nist.gov/article/OXczXYQBWEcjUZEYCPTo",
                    "CARDIVNSWC-TR-54-95/07 via NIST FireDoc https://firedoc.nist.gov/article/N3czXYQBWEcjUZEYCPTl"],
        "usable_labels_in_repo": False,
        "why": "Needs 75-100 kW/m2 cone data with smoke/gas; repo cone data is 25-65 kW/m2 (mostly 35/50), no smoke column used. No open per-material pass/fail table found."},
    "FAA 14 CFR 25.853 / App. F Part IV": {
        "criteria": "OSU heat release at 35 kW/m2 with pilot flame: peak HRR <= 65 kW/m2 and total heat release in first 2 min <= 65 kW-min/m2 (the '65/65' rule) for large cabin interior panels.",
        "sources": ["Federal Register 60(22), 2 Feb 1995 https://www.govinfo.gov/content/pkg/FR-1995-02-02/html/95-2114.htm"],
        "usable_labels_in_repo": "partial, not cone",
        "why": "OSU pHRR/THR exist in data/v2/loi_faa_ar0153.csv for 9 epoxies, so 65/65 pass/fail could be derived (THR in kW-min/m2 over the full test, not the first 2 min, so only an upper bound on THR). Different apparatus from cone; not usable to score cone-based CFRS."},
}

LEADS = [
    {"name": "DOPO-EP-ML (Huang, Zheng, Shi)", "url": "https://data.mendeley.com/datasets/6wxr5hsd48", "licence": "CC BY 4.0",
     "carries": "599 epoxy records, 130 papers: UL-94 + LOI + cone pHRR (406) / THR (397) at 35 and 50 kW/m2; no TTI. Best candidate to enlarge D1 (check DOI overlap with the lit snapshot)."},
    {"name": "IMDEA bio-FR PA56", "url": "https://zenodo.org/records/15097536", "licence": "CC BY 4.0",
     "carries": "101 PA56 rows: LOI + cone pHRR/THR/TSP; no UL-94 and no TTI in the sheet; one paper (D2-type only)."},
    {"name": "Chen et al. CEJ 2023 / Scikit-FlameRetardancy", "url": "https://github.com/ZevChan/Scikit-FlameRetardancy", "licence": "none stated (contact authors)",
     "carries": "Separate UL-94 (1041 rows) and LOI+TTI+pHRR+THR (383 rows) databases; descriptor-only, not linkable to a paper-level pairing."},
    {"name": "Jafari epoxy / PLA (Song group)", "url": "https://github.com/Pooya1234567/Epoxy_Flame_Retardant", "licence": "none (repo); articles CC BY 4.0",
     "carries": "UL-94 + LOI, pHRR mostly empty; epoxy file has no paper column. Not usable for within-paper pairing."},
    {"name": "Ma et al. JMI 2025", "url": "https://www.oaepublish.com/articles/jmi.2025.09", "licence": "repo none; article CC BY 4.0",
     "carries": "PP LOI only, no source-paper column. No."},
    {"name": "Juelich PMMA cone", "url": "https://zenodo.org/records/6301397", "licence": "CC BY 4.0",
     "carries": "HRR curves at 25-75 kW/m2, no UL-94; useful only for flux-dependence / 75 kW/m2 checks."},
]


if __name__ == "__main__":
    res = {"a_FRI_FPI_vs_CFRS": a_indices(), "b_faa_tables": b_faa(), "c_standards": C, "open_dataset_leads": LEADS}
    (HERE / "results/external_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res["a_FRI_FPI_vs_CFRS"].items() if k != "note"}, indent=1)[:3500])
    print(json.dumps(res["b_faa_tables"], indent=1)[:3500])
