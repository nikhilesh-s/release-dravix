"""A2 robustness analyses R1-R4 for CFRS weights (see DECLARATIONS.md, section A2).

CLI: python robustness.py --weights cfrs_weights_v0.json|eq --out results/robustness_v0.json
Library: sets(), oat(), dirichlet(), stability(), r3(), r4(), concordance().
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROPS, load_fsri, load_lit, score_matrix, subscores, weights_vec  # noqa: E402
from cfrps import DIRECTION, RANGES  # noqa: E402

EQ = {k: 1 / 6 for k in PROPS}

# R3(c) physical reference ranges (lo, hi, source). All values are data-independent round numbers.
# Only definitional/physical items are verified; literature magnitudes could not be checked here
# (no document access; web search returned no numbers), so they are marked "unverified" and no
# citation is invented. Schartel & Hull 2007, Fire Mater. 31:327, doi:10.1002/fam.949 is a real
# cone-data interpretation guide (bibliographic details verified) but its numbers were NOT checked.
PHYS = {
    "TTI": (10.0, 300.0, "unverified. Thermally thick ignition t = (pi/4) k rho c (Tig-T0)^2 / q^2 with assumed "
            "k rho c ~0.5e6 W2 s/m4K2, Tig 350-450 C gives ~17-29 s at 50 kW/m2 for neat polymers (assumed "
            "inputs, textbook formula); 10 s ~ immediate ignition, 300 s = 5 min taken as a long-delay anchor"),
    "pHRR": (100.0, 1500.0, "unverified. 100 kW/m2 recalled as MIL-STD-2031-style pass level (not confirmed); "
             "1500 kW/m2 ~ upper end of neat polyolefin/PS peaks at 50 kW/m2 (recalled, not confirmed)"),
    "THR": (0.0, 130.0, "derived: 3 mm x 1000 kg/m3 x ~43 MJ/kg (polyolefin heat of combustion, recalled, "
            "unverified) = ~129 MJ/m2, the complete-combustion ceiling for a typical 3 mm plaque"),
    "avg_HRR": (0.0, 500.0, "unverified. hi = round number between neat-polymer mean HRR (~300-400) and pHRR; "
                "no source checked"),
    "char_residue": (0.0, 100.0, "definitional: 0-100 wt% of initial mass"),
    "LOI": (21.0, 28.0, "21 vol% O2 = air (burn/no-burn point; ISO 4589-2 / ASTM D2863 define the test); "
            "28 vol% = common 'self-extinguishing' rule of thumb, exact source unverified"),
}
PHYS_RANGES = {k: (v[0], v[1]) for k, v in PHYS.items()}


# ---------------------------------------------------------------- data
@lru_cache(maxsize=1)
def _lit():
    return load_lit()


def sets() -> dict:
    """Ranked sets: name -> {df, S (fixed-range subscores), k (top-k size)}."""
    lit, out = _lit(), {}
    for name, hf in (("lit50", 50.0), ("lit35", 35.0)):
        df = lit[(lit.heat_flux == hf) & (lit[PROPS].notna().sum(axis=1) >= 4)].reset_index(drop=True)
        out[name] = {"df": df, "S": subscores(df), "k": max(1, round(0.1 * len(df)))}
    df = load_fsri(50)
    out["fsri50"] = {"df": df, "S": subscores(df), "k": 10}
    return out


def subs_ranges(df, ranges) -> np.ndarray:
    S = np.full((len(df), len(PROPS)), np.nan)
    for j, p in enumerate(PROPS):
        lo, hi = ranges[p]
        x = (df[p].to_numpy(float) - lo) / (hi - lo)
        S[:, j] = np.clip(1.0 - x if DIRECTION[p] < 0 else x, 0.0, 1.0)
    return S


def batch_ranges(df) -> dict:
    """Min-max within the ranked batch (NaN-safe; a constant column gets a unit span)."""
    r = {}
    for p in PROPS:
        v = df[p].dropna()
        lo, hi = (v.min(), v.max()) if len(v) else (0.0, 1.0)
        r[p] = (lo, hi if hi > lo else lo + 1.0)
    return r


# ---------------------------------------------------------------- D1 pairs and concordance
@lru_cache(maxsize=1)
def d1_pairs():
    """Same paper AND same heat flux, different UL-94 class, >= 1 shared input. (better, worse, paper)."""
    lit = _lit()
    pres = lit[PROPS].notna().to_numpy()
    ul = lit.ul94.to_numpy()
    ok = ~np.isnan(ul)
    b, w, pp = [], [], []
    for (paper, _hf), g in lit[ok].groupby(["paper", "heat_flux"]):
        idx = g.index.to_numpy()
        i, j = np.triu_indices(len(idx), 1)
        i, j = idx[i], idx[j]
        keep = (ul[i] != ul[j]) & (pres[i] & pres[j]).any(axis=1)
        i, j = i[keep], j[keep]
        swap = ul[i] < ul[j]  # make i the better-rated
        i, j = np.where(swap, j, i), np.where(swap, i, j)
        b.append(i)
        w.append(j)
        pp += [paper] * len(i)
    return np.concatenate(b), np.concatenate(w), np.array(pp)


def concordance(score: np.ndarray, pairs=None) -> dict:
    """Pooled and per-paper-averaged C; correct = better-rated row has the lower score, ties 0.5."""
    b, w, paper = pairs if pairs is not None else d1_pairs()
    d = score[w] - score[b]
    ok = ~np.isnan(d)
    c = np.where(np.abs(d) < 1e-9, 0.5, (d > 0).astype(float))[ok]
    paper = paper[ok]
    per = [c[paper == p].mean() for p in np.unique(paper)]
    return {"C": float(c.mean()), "C_paper": float(np.mean(per)), "n_pairs": int(ok.sum())}


def d1_score(w, S=None) -> np.ndarray:
    return score_matrix(subscores(_lit()) if S is None else S, w)


# ---------------------------------------------------------------- rank helpers
def tau(a, b) -> float:
    return float(kendalltau(a, b)[0])  # tau-b (scipy default)


def topk(score, k) -> set:
    return set(np.argsort(-np.asarray(score), kind="stable")[:k])


def overlap(a, b, k) -> float:
    return len(topk(a, k) & topk(b, k)) / k


# ---------------------------------------------------------------- R1
def oat(S, w, k=None) -> dict:
    """Each weight x0.5 and x1.5, renormalised: {prop: {"x0.5": (tau, overlap), "x1.5": ...}}."""
    w = np.asarray(w, float)
    k = k or max(1, round(0.1 * len(S)))
    base = score_matrix(S, w)
    out = {}
    for j, p in enumerate(PROPS):
        out[p] = {}
        for f in (0.5, 1.5):
            v = w.copy()
            v[j] *= f
            s = score_matrix(S, v / v.sum())
            out[p][f"x{f}"] = (tau(base, s), overlap(base, s, k))
    return out


def oat_dc(w) -> dict:
    """Change in pooled D1 C for each one-at-a-time weight change."""
    w = np.asarray(w, float)
    S = subscores(_lit())
    c0 = concordance(score_matrix(S, w))["C"]
    out = {}
    for j, p in enumerate(PROPS):
        out[p] = {}
        for f in (0.5, 1.5):
            v = w.copy()
            v[j] *= f
            out[p][f"x{f}"] = concordance(score_matrix(S, v / v.sum()))["C"] - c0
    return out


# ---------------------------------------------------------------- R2
def dirichlet(S, w, n=2000, conc=50, seed=0, k=None):
    """tau and top-k overlap arrays for w ~ Dir(conc * w) (alpha floored at 1e-3 so zero weights stay valid)."""
    w = np.asarray(w, float)
    k = k or max(1, round(0.1 * len(S)))
    W = np.random.default_rng(seed).dirichlet(np.maximum(conc * w, 1e-3), n)
    P = (~np.isnan(S)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = 100.0 * np.nan_to_num(S) @ W.T / (P @ W.T)
    base = score_matrix(S, w)
    t = np.array([tau(base, M[:, i]) for i in range(n)])
    o = np.array([overlap(base, M[:, i], k) for i in range(n)])
    return t, o


def stability(weights_dict: dict, n=2000, conc=50, seed=0) -> dict:
    """{set: (median_tau, p5_tau)} from R2, for the declared threshold test."""
    w = weights_vec(weights_dict)
    out = {}
    for name, s in sets().items():
        t, _ = dirichlet(s["S"], w, n, conc, seed, s["k"])
        out[name] = (float(np.median(t)), float(np.percentile(t, 5)))
    return out


# ---------------------------------------------------------------- R3
def r3(w) -> dict:
    """tau between fixed ranges (a) and batch-relative (b) / physical (c) rankings, plus D1 C under each.
    D1 C batch-relative: ranges from all lit rows at the same heat flux as the pair."""
    w = np.asarray(w, float)
    lit = _lit()
    out = {"tau": {}}
    for name, s in sets().items():
        a = score_matrix(s["S"], w)
        out["tau"][name] = {
            "batch": tau(a, score_matrix(subs_ranges(s["df"], batch_ranges(s["df"])), w)),
            "physical": tau(a, score_matrix(subs_ranges(s["df"], PHYS_RANGES), w))}
    Sb = np.full((len(lit), len(PROPS)), np.nan)
    for _hf, g in lit.groupby("heat_flux"):
        Sb[g.index.to_numpy()] = subs_ranges(g, batch_ranges(g))
    out["D1"] = {"fixed": concordance(score_matrix(subscores(lit), w)),
                 "batch": concordance(score_matrix(Sb, w)),
                 "physical": concordance(score_matrix(subs_ranges(lit, PHYS_RANGES), w))}
    return out


# ---------------------------------------------------------------- R4
MASKS = {"char+LOI": ("char_residue", "LOI"), "avg_HRR": ("avg_HRR",),
         "char+LOI+avg_HRR": ("char_residue", "LOI", "avg_HRR")}


def _impute(S, how, med):
    if how == "renorm":
        return None
    return np.where(np.isnan(S), 0.5 if how == "impute0.5" else med, S)


def r4(w) -> dict:
    """Masking inputs on rows with all six present (tau vs full-input ranking), and D1 C with the same
    columns masked on every lit row. Methods: renorm (current), impute0.5 (old DFRS), impute_median
    (median subscore of that property over the training snapshot)."""
    w = np.asarray(w, float)
    lit = _lit()
    S_all = subscores(lit)
    med = np.nanmedian(S_all, axis=0)
    full = lit[PROPS].notna().all(axis=1).to_numpy()
    S_full, base = S_all[full], score_matrix(S_all[full], w)
    out = {"n_full": int(full.sum()),
           "D1_unmasked": concordance(score_matrix(S_all, w))["C"], "masks": {}}
    for mname, cols in MASKS.items():
        j = [PROPS.index(c) for c in cols]
        r = {}
        for how in ("renorm", "impute0.5", "impute_median"):
            def run(S):
                M = S.copy()
                M[:, j] = np.nan
                F = _impute(M, how, med)
                return score_matrix(M if F is None else F, w)
            r[how] = {"tau": tau(base, run(S_full)), "D1_C": concordance(run(S_all))["C"]}
        out["masks"][mname] = r
    return out


# ---------------------------------------------------------------- driver
def run(wdict: dict) -> dict:
    w = weights_vec(wdict)
    res = {"weights": wdict, "physical_ranges": {k: {"lo": v[0], "hi": v[1], "source": v[2]} for k, v in PHYS.items()},
           "fixed_ranges": {k: list(v) for k, v in RANGES.items()}, "sets": {}}
    dc = oat_dc(w)
    for name, s in sets().items():
        t, o = dirichlet(s["S"], w, k=s["k"])
        res["sets"][name] = {
            "n": len(s["df"]), "k": s["k"],
            "R1": {p: {f: {"tau": v[0], "topk": v[1]} for f, v in d.items()} for p, d in oat(s["S"], w, s["k"]).items()},
            "R2": {"tau_median": float(np.median(t)), "tau_p5": float(np.percentile(t, 5)),
                   "topk_median": float(np.median(o)), "topk_p5": float(np.percentile(o, 5))}}
    res["R1_dC_D1"] = dc
    res["matters"] = {}
    for p in PROPS:
        tmin = min(v["tau"] for s in res["sets"].values() for v in s["R1"][p].values())
        res["matters"][p] = {"min_tau": tmin, "matters": bool(tmin < 0.95)}
    res["R3"] = r3(w)
    res["R4"] = r4(w)
    return res


def load_weights(arg: str) -> dict:
    if arg == "eq":
        return dict(EQ)
    d = json.loads(Path(arg).read_text())["weights"]
    assert set(d) == set(PROPS) and abs(sum(d.values()) - 1) < 1e-9, "need six weights summing to 1"
    return d


def table(r: dict) -> str:
    L = ["weights: " + " ".join(f"{k}={v:.3f}" for k, v in r["weights"].items())]
    for n, s in r["sets"].items():
        b = s["R2"]
        L.append(f"[{n}] n={s['n']} k={s['k']}  R2 tau med/p5 {b['tau_median']:.3f}/{b['tau_p5']:.3f}  "
                 f"topk med/p5 {b['topk_median']:.2f}/{b['topk_p5']:.2f}")
    L.append("R1 min tau (x0.5 / x1.5) per set: " + " | ".join(PROPS))
    for n, s in r["sets"].items():
        L.append(f"  {n:7s} " + " | ".join(f"{s['R1'][p]['x0.5']['tau']:.3f}/{s['R1'][p]['x1.5']['tau']:.3f}" for p in PROPS))
    L.append("  dC D1  " + " | ".join(f"{r['R1_dC_D1'][p]['x0.5']:+.3f}/{r['R1_dC_D1'][p]['x1.5']:+.3f}" for p in PROPS))
    L.append("  matters: " + ", ".join(f"{p}={'Y' if m['matters'] else 'n'}({m['min_tau']:.3f})" for p, m in r["matters"].items()))
    L.append("R3 tau vs fixed: " + "; ".join(f"{n}: batch {v['batch']:.3f} phys {v['physical']:.3f}" for n, v in r["R3"]["tau"].items()))
    L.append("R3 D1 C: " + "; ".join(f"{k} {v['C']:.3f} (paper {v['C_paper']:.3f}, n={v['n_pairs']})" for k, v in r["R3"]["D1"].items()))
    L.append(f"R4 (n_full={r['R4']['n_full']}, D1 C unmasked {r['R4']['D1_unmasked']:.3f}): tau / D1 C")
    for m, d in r["R4"]["masks"].items():
        L.append(f"  {m:18s} " + "  ".join(f"{h} {v['tau']:.3f}/{v['D1_C']:.3f}" for h, v in d.items()))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="JSON with a 'weights' dict, or the literal 'eq'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(load_weights(a.weights))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(table(res))
