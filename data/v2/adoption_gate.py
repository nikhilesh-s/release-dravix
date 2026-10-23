#!/usr/bin/env python3
"""Adoption gate v2 -- experiments/configs/adoption_gate_v2.json (approved D1, 2026-10-07), pseudo_code verbatim.

  python data/v2/adoption_gate.py                         # label-free self-test on synthetic data (must pass first)
  python data/v2/adoption_gate.py --npz RUNS/<round>_<t>_oof.npz --out RUNS/<round>_<t>_gate_v2.json \
         --result c1=experiments/results/<slug>.json [--result c2=...]   # one gate run per target per round

The --npz run first does implementation.consistency_checks_in_the_round (1e-9 abs, full-precision fields only), then
calls adoption_gate_v2 with exactly the config constants. Never run with python -O (the asserts are the checks).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import bootstrap_ci  # noqa: E402

N_BOOT, SEED, LEVEL, REPEATS, MIN_WINS = 2000, 0, 0.95, 10, 8


def gate_draws(groups, n_boot=N_BOOT, seed=SEED):
    # same rng, same group order, same call sequence as eval_protocol.bootstrap_ci
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    return [np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))]) for _ in range(n_boot)]


def repeat_mean_r2(y, oof, rows):  # oof shape (REPEATS, n)
    return np.mean([r2_score(y[rows], oof[r, rows]) for r in range(oof.shape[0])])


def adoption_gate_v2(y, groups, oof_inc, oof_cands, order, selectable):
    # oof_cands: {cand_id: (REPEATS, n) array} for EVERY family member; order: config order; selectable: subset of order
    n = len(y)
    assert set(oof_cands) == set(order) and set(selectable) <= set(order) and len(selectable) >= 1
    for P in [oof_inc, *oof_cands.values()]:
        assert P.shape == (REPEATS, n) and np.isfinite(P).all()   # else BLOCKED: no adoption decision
    r2_inc = np.array([r2_score(y, oof_inc[r]) for r in range(REPEATS)])
    r2 = {c: np.array([r2_score(y, oof_cands[c][r]) for r in range(REPEATS)]) for c in order}
    theta = {c: float((r2[c] - r2_inc).mean()) for c in order}
    wins = {c: int((r2[c] > r2_inc).sum()) for c in order}
    ci_inc = bootstrap_ci(y, oof_inc[0], groups)
    ci = {c: bootstrap_ci(y, oof_cands[c][0], groups) for c in order}

    draws = gate_draws(groups)
    # identity check: the gate's draws ARE bootstrap_ci's draws
    assert np.array_equal(np.percentile([r2_score(y[rw], oof_inc[0, rw]) for rw in draws], [2.5, 97.5]), ci_inc)
    inc_b = np.array([repeat_mean_r2(y, oof_inc, rw) for rw in draws])
    d = {c: np.array([repeat_mean_r2(y, oof_cands[c], rw) for rw in draws]) - inc_b for c in order}
    se = {c: float(d[c].std(ddof=1)) for c in order}
    T = np.column_stack([(d[c] - theta[c]) / se[c] if se[c] > 0 else np.zeros(N_BOOT) for c in order])
    q = float(np.quantile(T.max(axis=1), LEVEL))
    lb = {c: theta[c] - q * se[c] for c in order}

    best = max((c for c in order if c in selectable), key=lambda c: r2[c].mean())   # ties -> first listed
    g1 = wins[best] >= MIN_WINS
    g2 = bool(ci[best][0] >= ci_inc[0])   # plain bool, so json.dump works
    g3 = lb[best] > 0
    adopt = g1 and g2 and g3   # AND round-specific conditions AND critic pass (outside this function)

    # report-only quantities (never used to adopt, select or demote)
    q_raw = float(np.quantile(np.max(np.column_stack([d[c] - theta[c] for c in order]), axis=1), LEVEL))
    report = {c: dict(theta=theta[c], wins=wins[c], ci_lo=float(ci[c][0]), ci_hi=float(ci[c][1]), se=se[c],
                      lb_maxT=lb[c],
                      lb_unadjusted=2 * theta[c] - float(np.quantile(d[c], LEVEL)),
                      lb_raw_maxT=theta[c] - q_raw,
                      p_boot_gain_le_0=float(np.mean(d[c] <= 0))) for c in order}
    return dict(K=len(order), q=q, q_raw=q_raw, best=best, G1=g1, G2=g2, G3=g3, adopt_if_critic_passes=adopt,
                ci_inc=[float(v) for v in ci_inc], n_rows=n, n_groups=len(np.unique(groups)), n_boot=N_BOOT, seed=SEED,
                per_candidate=report)


def consistency_checks(y, groups, oof_inc, inc_per_repeat_r2, inc_ci, oof_cands, results, tol=1e-9):
    """implementation.consistency_checks_in_the_round: OOF arrays vs full-precision fields, all to 1e-9 absolute."""
    def chk(name, oof, per_rep, mean_full, ci_full):
        r2 = np.array([r2_score(y, oof[r]) for r in range(REPEATS)])
        assert len(per_rep) == REPEATS and np.all(np.abs(r2 - np.asarray(per_rep)) <= tol), f"{name}: per-repeat R2"
        if mean_full is not None:
            assert abs(r2.mean() - mean_full) <= tol, f"{name}: rkf_mean_full"
        ci = bootstrap_ci(y, oof[0], groups)
        assert np.all(np.abs(ci - np.asarray(ci_full)) <= tol), f"{name}: ci"
    chk("incumbent", oof_inc, inc_per_repeat_r2, float(np.mean(inc_per_repeat_r2)), inc_ci)
    for c, P in oof_cands.items():
        R = results[c]
        chk(c, P, R["per_repeat_r2"], R["rkf_mean_full"], (R["ci_lo_full"], R["ci_hi_full"]))


def run_npz(npz_path, result_paths, out_path):
    Z = np.load(npz_path, allow_pickle=False)
    order = [str(s) for s in Z["order"]]
    selectable = [str(s) for s in Z["selectable"]]
    y, groups = Z["y"], Z["groups"]
    oof_inc = Z["oof_incumbent"]
    oof_cands = {c: Z[f"oof_{c}"] for c in order}
    assert set(result_paths) == set(order), (sorted(result_paths), order)
    results = {c: json.load(open(p)) for c, p in result_paths.items()}
    consistency_checks(y, groups, oof_inc, Z["inc_per_repeat_r2"], Z["inc_ci"], oof_cands, results)
    print(f"consistency checks passed (1e-9) for incumbent + {order}")
    out = adoption_gate_v2(y, groups, oof_inc, oof_cands, order, selectable)
    out["gate_config"] = "experiments/configs/adoption_gate_v2.json"
    out["order"], out["selectable"], out["npz"] = order, selectable, os.path.relpath(npz_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_candidate"}, indent=1))
    for c in order:
        r = out["per_candidate"][c]
        print(f"  {c:3s} theta {r['theta']:+.4f} wins {r['wins']}/10 ci_lo {r['ci_lo']:.4f} se {r['se']:.4f} "
              f"lb_maxT {r['lb_maxT']:+.4f} lb_unadj {r['lb_unadjusted']:+.4f} p(d<=0) {r['p_boot_gain_le_0']:.3f}")
    print(f"wrote {out_path}")
    return out


def self_test():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 5)
    y = 3 * rng.normal(size=40)[groups] + 10 + 0.3 * rng.normal(size=200)
    inc = np.stack([y + rng.normal(scale=2, size=200) for _ in range(10)])
    # (a) gate_draws identity with bootstrap_ci
    pct = np.percentile([r2_score(y[rw], inc[0, rw]) for rw in gate_draws(groups)], [2.5, 97.5])
    assert np.array_equal(pct, bootstrap_ci(y, inc[0], groups)), "(a) draw identity"
    # (b) three candidates equal to the incumbent
    o = adoption_gate_v2(y, groups, inc, {"a": inc.copy(), "b": inc.copy(), "c": inc.copy()}, ["a", "b", "c"], ["a", "b", "c"])
    assert all(r["theta"] == 0 and r["lb_maxT"] <= 0 for r in o["per_candidate"].values()) and not o["G3"], "(b)"
    # (c) a perfect candidate vs the noisy incumbent
    perfect = np.stack([y] * 10)
    o = adoption_gate_v2(y, groups, inc, {"p": perfect, "a": inc.copy()}, ["p", "a"], ["p", "a"])
    assert o["best"] == "p" and o["per_candidate"]["p"]["lb_maxT"] > 0 and o["G1"], "(c)"
    # (d) K = 1: studentised max-T bound == basic bootstrap bound
    noisy = np.stack([y + rng.normal(scale=1.5, size=200) for _ in range(10)])
    o = adoption_gate_v2(y, groups, inc, {"n": noisy}, ["n"], ["n"])
    r = o["per_candidate"]["n"]
    assert abs(r["lb_maxT"] - r["lb_unadjusted"]) <= 1e-12, "(d)"
    print("adoption_gate self-test OK (a draw identity, b null family, c perfect candidate, d K=1)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        self_test()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", action="append", required=True, help="cand_id=path/to/result.json")
    a = ap.parse_args()
    run_npz(a.npz, dict(s.split("=", 1) for s in a.result), a.out)
