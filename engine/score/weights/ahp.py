"""AHP for the CFRS weights (DECLARATIONS.md A3): 7 criteria, 21 pairwise judgments, Saaty 1-9.

Usage: python ahp.py responses.csv [--out ahp_weights.json]
CSV: one row per expert; a `name` column; one column per pair titled "<LEFT> vs <RIGHT>" (e.g.
"TTI vs pHRR"). Cells are numbers (see matrix()) or Google Forms option text such as
"LEFT strongly more important (5)", "Equal (1)", "RIGHT moderately more important (3)".
The output JSON's "weights" (six CFRS keys, sum 1) is what calibrate.py takes as --prior.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations

import numpy as np

CRITERIA = ("TTI", "pHRR", "THR", "avg_HRR", "LOI", "char_residue", "smoke")
PAIRS = tuple(combinations(range(7), 2))  # 21 (i<j) index pairs, row-major
RI = {3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}  # Saaty random index
CFRS_KEYS = tuple(c for c in CRITERIA if c != "smoke")


def title(p) -> str:
    return f"{CRITERIA[p[0]]} vs {CRITERIA[p[1]]}"


def matrix(judgments) -> np.ndarray:
    """21 signed values (PAIRS order) -> reciprocal 7x7. For pair (i, j):
    v > 1 : LEFT (i) more important by v; v = 1 or -1: equal;
    0 < v < 1 (pass 1/k) or v = -k (k >= 1): RIGHT (j) more important by k. Range 1/9..9."""
    if len(judgments) != len(PAIRS):
        raise ValueError(f"need {len(PAIRS)} judgments, got {len(judgments)}")
    A = np.eye(7)
    for (i, j), v in zip(PAIRS, judgments):
        v = float(v)
        a = 1 / -v if v < 0 else v  # negative integer -k means RIGHT by k, i.e. a_ij = 1/k
        if not 1 / 9 - 1e-9 <= a <= 9 + 1e-9:
            raise ValueError(f"judgment {v} for {title((i, j))} is outside the Saaty range")
        A[i, j], A[j, i] = a, 1 / a
    return A


def weights(A) -> np.ndarray:
    """Principal eigenvector, normalised to sum 1."""
    vals, vecs = np.linalg.eig(A)
    v = np.abs(vecs[:, np.argmax(vals.real)].real)
    return v / v.sum()


def consistency(A):
    """(lambda_max, CI, CR); CR = CI / RI(n), 0 for n < 3."""
    n = len(A)
    lam = float(np.max(np.linalg.eigvals(A).real))
    ci = (lam - n) / (n - 1)
    return lam, ci, (ci / RI[n] if n in RI else 0.0)


def aggregate(mats) -> np.ndarray:
    """AIJ: element-wise geometric mean of judgment matrices (stays reciprocal)."""
    return np.exp(np.mean([np.log(m) for m in mats], axis=0))


def most_inconsistent(A, k=3):
    """The k pairs whose judgment most disagrees with the eigenvector ratio w_i/w_j, i.e. largest
    |log(a_ij * w_j / w_i)|. Each: pair title, what was said, what the rest of the answers imply."""
    w = weights(A)
    out = [(abs(np.log(A[i, j] * w[j] / w[i])), i, j) for i, j in PAIRS]
    return [{"pair": title((i, j)), "judged": round(float(A[i, j]), 3),
             "implied": round(float(w[i] / w[j]), 3), "error": round(float(e), 3)}
            for e, i, j in sorted(out, reverse=True)[:k]]


def cfrs_weights(w7):
    """Drop smoke, renormalise the other six to sum 1. Returns ({key: weight}, smoke_share)."""
    d = dict(zip(CRITERIA, map(float, w7)))
    smoke = d.pop("smoke")
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}, smoke


# ---------------------------------------------------------------- CSV / CLI
def parse_cell(s) -> float:
    """Number, fraction ("1/3"), or Google Forms option text -> signed value for matrix()."""
    s = str(s).strip()
    if s.lower().startswith("equal"):
        return 1.0
    m = re.search(r"\((\d)\)", s)
    if m:
        return -float(m.group(1)) if s.upper().startswith("RIGHT") else float(m.group(1))
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def summarise(A) -> dict:
    w = weights(A)
    return {"weights7": dict(zip(CRITERIA, map(float, w))), "cr": consistency(A)[2]}


def run(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    namecol = next(c for c in rows[0] if c.strip().lower() == "name")
    mats, experts = [], {}
    for r in rows:
        A = matrix([parse_cell(r[title(p)]) for p in PAIRS])
        mats.append(A)
        s = summarise(A)
        s["ok"] = bool(s["cr"] < 0.10)
        s["most_inconsistent"] = [] if s["ok"] else most_inconsistent(A)
        experts[r[namecol]] = s
    agg = aggregate(mats)
    prior, smoke = cfrs_weights(weights(agg))
    return {"version": "ahp", "weights": prior, "smoke_share": smoke, "n_experts": len(rows),
            "experts": experts, "aggregate": summarise(agg),
            "note": ("'weights' = aggregate AHP vector with smoke dropped and the six CFRS keys "
                     "renormalised; use as calibrate.py --prior. smoke_share is recorded only to size "
                     "the gap (engine predicts no smoke). Experts with ok=false (CR >= 0.10) should "
                     "revisit most_inconsistent before their answers are relied on.")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("responses")
    ap.add_argument("--out", default="ahp_weights.json")
    a = ap.parse_args()
    out = run(a.responses)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v:.3f}" for k, v in out["weights"].items())
          + f"; smoke {out['smoke_share']:.3f}; CR " + ", ".join(f"{n} {e['cr']:.3f}" for n, e in out["experts"].items())
          + f"; aggregate CR {out['aggregate']['cr']:.3f}")
