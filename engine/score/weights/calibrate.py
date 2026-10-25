"""A1 outcome calibration of CFRS weights (spec: DECLARATIONS.md, "Data D1/D2", "Metric", "A1").

Pairs are within paper and heat flux; correct = the better-labelled row gets the lower CFRS.
Run: python calibrate.py [--prior path.json] [--out results/calibration.json] [--boot 500]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from common import HERE, LIT, PROPS, load_lit, score_matrix, subscores, weights_vec

K = len(PROPS)
V0 = weights_vec(json.loads((HERE / "cfrs_weights_v0.json").read_text())["weights"])
EQ = np.full(K, 1 / K)
FLOOR = 0.0  # amendment A1-1: minimum weight per active input (CLI --floor)


def build_pairs(df, S, label="ul94", min_gap=None, drop=()):
    """(better_row, worse_row) within same paper AND heat flux, different label (|gap| >= min_gap),
    sharing >= 1 input present in both rows (inputs in `drop`, and LOI when label is LOI, excluded)."""
    if label == "LOI":
        drop = tuple(drop) + ("LOI",)
    use = [j for j, p in enumerate(PROPS) if p not in drop]
    pres = ~np.isnan(S[:, use])
    y = df[label].to_numpy(float)
    out = []
    ok = ~np.isnan(y) & df.heat_flux.notna().to_numpy()
    for _, idx in df[ok].groupby(["paper", "heat_flux"]).indices.items():
        rows = df[ok].index.to_numpy()[idx]
        for a, b in itertools.combinations(rows, 2):
            gap = abs(y[a] - y[b])
            if gap > 0 and gap >= (min_gap or 0) and (pres[a] & pres[b]).any():
                out.append((a, b) if y[a] > y[b] else (b, a))  # higher label = better
    return np.array(out, int).reshape(-1, 2)


def correct(scores, pairs):
    """Per-pair correctness: 1 if better < worse, 0.5 tie (or NaN score), 0 otherwise."""
    d = scores[pairs[:, 1]] - scores[pairs[:, 0]]
    d = np.nan_to_num(d)
    return np.where(np.abs(d) < 1e-12, 0.5, (d > 0).astype(float))


def concordance(scores, pairs):
    return float(correct(scores, pairs).mean())


def concordance_by_paper(scores, pairs, paper):
    c = correct(scores, pairs)
    return float(np.mean([c[paper == p].mean() for p in np.unique(paper)]))


def fit(S, pairs, prior, lam=0.1, pw=None, n_random=3, seed=0):
    """Weights on the simplex minimising mean log(1+exp(-beta*(CFRS_worse-CFRS_better)/100))
    + lam*||w-prior||^2. Columns of S that are all NaN are inactive (weight 0, prior renormalised).
    Each active weight >= FLOOR: w = FLOOR + (1 - n_active * FLOOR) * softmax(t)."""
    act = ~np.isnan(S).all(0)
    fl = FLOOR
    simplex = lambda t: fl + (1 - act.sum() * fl) * np.exp(t - t.max()) / np.exp(t - t.max()).sum()  # noqa: E731
    prior = np.asarray(prior, float)[act] / np.asarray(prior, float)[act].sum()
    Sb, Sw = S[pairs[:, 0]][:, act], S[pairs[:, 1]][:, act]
    Xb, Xw = np.nan_to_num(Sb), np.nan_to_num(Sw)
    Pb, Pw = (~np.isnan(Sb)).astype(float), (~np.isnan(Sw)).astype(float)
    pw = np.ones(len(pairs)) if pw is None else np.asarray(pw, float)

    def loss(t):
        w = simplex(t[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            d = 100 * (Xw @ w) / (Pw @ w) - 100 * (Xb @ w) / (Pb @ w)  # CFRS_worse - CFRS_better
        d = np.nan_to_num(d)
        return (pw * np.logaddexp(0, -np.exp(t[-1]) * d / 100)).sum() / pw.sum() + lam * ((w - prior) ** 2).sum()

    rng = np.random.default_rng(seed)
    starts = [prior, np.full(len(prior), 1 / len(prior))] + list(rng.dirichlet(np.ones(len(prior)), n_random))
    best = None
    for w0 in starts:
        for lb in (np.log(5), np.log(30)):
            r = minimize(loss, np.r_[np.log(np.clip(w0 - fl, 0, None) + 1e-6), lb], method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
    w = simplex(best.x[:-1])
    out = np.zeros(K); out[act] = w
    return out


def cv(S, df, pairs, prior, reps=range(10), drop_one=True, n_random=1):
    """Pooled out-of-fold C per repeat for fitted / v0 / eq, folds = GroupKFold by paper (pairs go
    with their paper). Repeat-0 per-pair correctness is returned for the bootstrap. drop_one: refit
    with each input removed (others renormalised), OOF C per input."""
    paper = df.paper.to_numpy()[pairs[:, 0]]
    cv_v0, cv_eq = correct(score_matrix(S, V0), pairs), correct(score_matrix(S, EQ), pairs)
    res = {"fitted": [], "v0": [], "eq": [], "drop_one": {p: [] for p in PROPS} if drop_one else {}}
    variants = {None: S}
    if drop_one:
        for j, p in enumerate(PROPS):
            if not np.isnan(S[:, j]).all():
                Sj = S.copy(); Sj[:, j] = np.nan; variants[p] = Sj
    for r in reps:
        oof = {v: np.zeros(len(pairs)) for v in variants}
        for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(pairs, groups=paper):
            for v, Sv in variants.items():
                w = fit(Sv, pairs[tr], prior, n_random=n_random)
                oof[v][te] = correct(score_matrix(Sv, w), pairs[te])
        res["fitted"].append(oof[None].mean()); res["v0"].append(cv_v0.mean()); res["eq"].append(cv_eq.mean())
        for p in res["drop_one"]:
            if p in oof:
                res["drop_one"][p].append(oof[p].mean())
        if r == 0:
            res["pair_correct_r0"] = {"fitted": oof[None], "v0": cv_v0, "eq": cv_eq}
    return res


def boot_diff(c_a, c_b, paper, n=2000, seed=0):
    """Paper-bootstrap 95% CI of pooled C(a) - C(b) from per-pair correctness."""
    up, inv = np.unique(paper, return_inverse=True)
    d = np.bincount(inv, c_a - c_b, len(up)); m = np.bincount(inv, minlength=len(up))
    cnt = np.random.default_rng(seed).multinomial(len(up), np.full(len(up), 1 / len(up)), n)
    est = (cnt @ d) / (cnt @ m)
    return [float(x) for x in np.percentile(est, [2.5, 97.5])]


def boot_weights(S, df, pairs, prior, n=500, seed=1):
    paper = df.paper.to_numpy()[pairs[:, 0]]
    up, inv = np.unique(paper, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.array([fit(S, pairs, prior, pw=rng.multinomial(len(up), np.full(len(up), 1 / len(up)))[inv], n_random=0)
                  for _ in range(n)])
    q = np.percentile(W, [5, 50, 95], axis=0)
    out = {}
    for j, p in enumerate(PROPS):
        wd = float(q[2, j] - q[0, j])
        out[p] = {"p5": float(q[0, j]), "p50": float(q[1, j]), "p95": float(q[2, j]), "width": wd,
                  "label": "constrained" if wd < 0.15 else "weak" if wd <= 0.30 else "unconstrained"}
    return out


def adopt(res, ci):
    f, v0, eq = (np.array(res[k]) for k in ("fitted", "v0", "eq"))
    wins = int((f > v0).sum())
    ok = wins >= 8 and ci[0] > 0
    v1 = "fitted" if ok else ("v0" if v0.mean() >= eq.mean() else "eq")
    return {"fitted_beats_v0_repeats": f"{wins}/{len(f)}", "ci_lower_gt_0": bool(ci[0] > 0),
            "adopt_fitted": bool(ok), "v1": v1}


def single_input(S, pairs):
    """C of each input alone on pairs where it is present in both rows."""
    out = {}
    for j, p in enumerate(PROPS):
        m = ~np.isnan(S[pairs[:, 0], j]) & ~np.isnan(S[pairs[:, 1], j])
        out[p] = {"C": concordance(S[:, j], pairs[m]) if m.any() else None, "n_pairs": int(m.sum())}
    return out


def run(df, S, label, prior, min_gap, n_boot, reps=range(10)):
    t0 = time.time()
    drop = ("LOI",) if label == "LOI" else ()
    S = S.copy()
    for p in drop:
        S[:, PROPS.index(p)] = np.nan
    pairs = build_pairs(df, S, label, min_gap)
    paper = df.paper.to_numpy()[pairs[:, 0]]
    rows = np.unique(pairs)
    w = fit(S, pairs, prior, n_random=5)
    res = cv(S, df, pairs, prior, reps)
    pc = res.pop("pair_correct_r0")
    ci = boot_diff(pc["fitted"], pc["v0"], paper)
    sc = {"fitted": score_matrix(S, w), "v0": score_matrix(S, V0), "eq": score_matrix(S, EQ)}
    return {
        "label": label, "n_rows_in_pairs": int(len(rows)), "n_pairs": int(len(pairs)),
        "n_papers": int(len(np.unique(paper))),
        "coverage_among_paired_rows": {p: int((~np.isnan(S[rows, j])).sum()) for j, p in enumerate(PROPS)},
        "fitted_weights": dict(zip(PROPS, map(float, w))), "v0_weights": dict(zip(PROPS, map(float, V0))),
        "full_data_C": {k: concordance(v, pairs) for k, v in sc.items()},
        "full_data_C_paper_avg": {k: concordance_by_paper(v, pairs, paper) for k, v in sc.items()},
        "cv_oof_C": {k: res[k] for k in ("fitted", "v0", "eq")},
        "cv_oof_C_mean": {k: float(np.mean(res[k])) for k in ("fitted", "v0", "eq")},
        "cv_oof_C_paper_avg_r0": {k: concordance_by_paper_pairs(pc[k], paper) for k in pc},
        "ci95_fitted_minus_v0_r0": ci,
        "drop_one_oof_C_mean": {p: float(np.mean(v)) for p, v in res["drop_one"].items() if v},
        "bootstrap_weights": boot_weights(S, df, pairs, prior, n_boot),
        "n_boot_refits": n_boot,
        "adoption": adopt(res, ci),
        "single_input_C": single_input(S, pairs),
        "runtime_s": round(time.time() - t0, 1),
    }


def concordance_by_paper_pairs(c, paper):
    return float(np.mean([c[paper == p].mean() for p in np.unique(paper)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior"); ap.add_argument("--out", default=str(HERE / "results/calibration.json"))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--floor", type=float, default=0.0)
    a = ap.parse_args()
    global FLOOR
    FLOOR = a.floor
    prior = weights_vec(json.loads(open(a.prior).read())["weights"]) if a.prior else EQ
    df = load_lit(); S = subscores(df)
    out = {"snapshot_sha256": hashlib.sha256(LIT.read_bytes()).hexdigest(), "prior": dict(zip(PROPS, map(float, prior))),
           "lambda": 0.1, "floor": FLOOR, "n_rows_total": len(df),
           "D1": run(df, S, "ul94", prior, None, a.boot), "D2": run(df, S, "LOI", prior, 3, a.boot)}
    p = HERE / a.out if not a.out.startswith("/") else __import__("pathlib").Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for d in ("D1", "D2"):
        r = out[d]
        print(d, r["n_pairs"], "pairs", r["n_papers"], "papers", r["cv_oof_C_mean"], r["ci95_fitted_minus_v0_r0"],
              r["adoption"], r["runtime_s"], "s")


if __name__ == "__main__":
    main()
