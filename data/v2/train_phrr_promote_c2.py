#!/usr/bin/env python3
"""Dravix V2 -- pHRR promotion rescoring (pre-declared in experiments/configs/phrr_round1_promote.json, 2026-10-02).

Decision (Arnav 2026-10-02): PlasticC and HIPS are one material group; promote option B = c2 (HGB on the exp043 features,
log1p target). Data = train_phrr_v3_robust.load(replicate_target=True) with the --bc extras, has_mcc rows (191 rows / 65
materials as labelled, 64 groups once PlasticC is aliased to HIPS FOR GROUPING ONLY; no row is dropped or changed).

1. Sanity: unmerged c2 must still give rkf_mean 0.661 (round 1, exp056).
2. Score inc_merged (exp043 config: rf(), log target) and c2_merged (hgb(), log target) on merged groups:
   repeated GroupKFold 5x10 mean/std, repeat-0 material-bootstrap CI (the rule's CI), pooled-over-10-repeats CI
   (material-bootstrap of the repeat-MEAN OOF prediction per row), GroupShuffleSplit x10 mean, category-grouped GKF
   mean + CI (bootstrap over merged material groups), paired per-repeat wins/delta of c2_merged vs inc_merged.
3. Fit c2 on all 191 rows -> models/v2/phrr_hgb_v4.joblib (TransformedTargetRegressor, predicts kW/m2), same bundle keys as
   phrr_rf_v3.joblib. phrr_rf_v3.joblib is never touched.

Rerun: python3 data/v2/train_phrr_promote_c2.py   (writes experiments/results/phrr_promote_*.json, models/v2/phrr_hgb_v4.joblib)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_phrr_v3_robust as T  # sets OMP_NUM_THREADS=1 before numpy

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score

from eval_protocol import bootstrap_ci, grouped_shuffle_r2, repeated_group_kfold

T.EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")  # exactly what --bc does

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")
MODEL_PATH = os.path.join(ROOT, "models", "v2", "phrr_hgb_v4.joblib")
V3_PATH = os.path.join(ROOT, "models", "v2", "phrr_rf_v3.joblib")
CONFIG = "experiments/configs/phrr_round1_promote.json"
TARGET, SOTA, FEATS = T.TARGET, T.SOTA, T.BASE + T.PHYS
H5 = ("ahead of the published number on a material-grouped protocol (harder than the competitor's random split), "
      "different population; not ahead when whole material families are held out")


def merged(mat):
    return np.where(mat == "PlasticC", "HIPS", mat)


def pooled_ci(y, oof, groups, n_boot=2000, seed=0):
    """Material-bootstrap CI (same draws as eval_protocol.bootstrap_ci) of the R2 of the repeat-MEAN OOF prediction."""
    return bootstrap_ci(y, oof.mean(axis=0), groups, n_boot=n_boot, seed=seed)


def score(model, df, grp):
    X, y, cat = df[FEATS].to_numpy(float), df[TARGET].to_numpy(float), df["category"].to_numpy()
    gss = grouped_shuffle_r2(model, X, y, grp, log_target=True)
    rkf, oof = repeated_group_kfold(model, X, y, grp, log_target=True)
    lo, hi = bootstrap_ci(y, oof[0], grp)
    plo, phi = pooled_ci(y, oof, grp)
    crkf, coof = repeated_group_kfold(model, X, y, cat, log_target=True)
    clo, chi = bootstrap_ci(y, coof[0], grp)  # CI resamples (merged) materials; the splits hold out whole categories
    return dict(gss_mean=gss.mean(), gss_std=gss.std(), rkf=rkf, rkf_mean=rkf.mean(), rkf_std=rkf.std(), ci_lo=lo, ci_hi=hi,
                pool_r2=r2_score(y, oof.mean(axis=0)), pool_lo=plo, pool_hi=phi, cat_mean=crkf.mean(), cat_std=crkf.std(),
                cat_lo=clo, cat_hi=chi, oof=oof, n_rows=len(y), n_groups=len(np.unique(grp)), n_cats=len(np.unique(cat)))


def line(name, r):
    return (f"{name:12s} rkf {r['rkf_mean']:.4f} +/- {r['rkf_std']:.4f}  rep0 CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
            f"pooled(mean-OOF) R2 {r['pool_r2']:.4f} CI [{r['pool_lo']:.3f}, {r['pool_hi']:.3f}]  gss {r['gss_mean']:.4f}  "
            f"category-GKF {r['cat_mean']:.4f} +/- {r['cat_std']:.4f} CI [{r['cat_lo']:.3f}, {r['cat_hi']:.3f}]  "
            f"n={r['n_rows']}/{r['n_groups']} groups, {r['n_cats']} categories")


def main():
    t0 = time.time()
    full = T.load(replicate_target=True)
    df = full[full["has_mcc"]].reset_index(drop=True)
    mat = df["material"].to_numpy()
    grp = merged(mat)
    n_alias = int((mat == "PlasticC").sum())
    print(f"Config: {CONFIG}\nload(replicate_target=True), has_mcc: {len(df)} rows / {len(np.unique(mat))} materials; "
          f"merged groups: {len(np.unique(grp))}; PlasticC rows aliased to HIPS for grouping: {n_alias}")
    assert (len(df), len(np.unique(mat)), len(np.unique(grp))) == (191, 65, 64), "unexpected row/group counts"
    assert n_alias > 0 and "HIPS" in set(mat)
    print(f"features: {FEATS}\nEXTRAS: {T.EXTRAS}")

    # 1. sanity: round-1 unmerged c2 -------------------------------------------------------------------------------
    X, y = df[FEATS].to_numpy(float), df[TARGET].to_numpy(float)
    rkf0, _ = repeated_group_kfold(T.hgb(), X, y, mat, log_target=True)
    print(f"\n=== Sanity: unmerged c2 (round 1) rkf_mean {rkf0.mean():.4f} (declared 0.661) ===")
    if abs(round(rkf0.mean(), 3) - 0.661) > 0.002:
        print("BLOCKED: round-1 c2 not reproduced; no rescoring.")
        sys.exit(2)

    # 2. rescoring on merged groups --------------------------------------------------------------------------------
    print("\n=== Merged-group scoring (eval_protocol seeds, groups = material with PlasticC -> HIPS) ===")
    inc = score(T.rf(), df, grp)
    c2 = score(T.hgb(), df, grp)
    print(line("inc_merged", inc))
    print(line("c2_merged", c2))
    d = c2["rkf"] - inc["rkf"]
    wins, delta = int((d > 0).sum()), float(d.mean())
    print(f"\npaired c2_merged vs inc_merged (same merged splits): wins {wins}/10, mean delta {delta:+.4f}")
    print(f"  inc per-repeat R2: {np.round(inc['rkf'], 3).tolist()}\n  c2  per-repeat R2: {np.round(c2['rkf'], 3).tolist()}")
    print("critic expected: c2 0.611 [0.516, 0.838]; exp043 0.546 [0.322, 0.830]")
    print(f"measured:        c2 {c2['rkf_mean']:.3f} [{c2['ci_lo']:.3f}, {c2['ci_hi']:.3f}]; "
          f"exp043 {inc['rkf_mean']:.3f} [{inc['ci_lo']:.3f}, {inc['ci_hi']:.3f}]")
    print(f"c2_merged CI lower bound {c2['ci_lo']:.3f} > SOTA {SOTA}: {c2['ci_lo'] > SOTA}; "
          f"category-grouped {c2['cat_mean']:.3f} (CI [{c2['cat_lo']:.3f}, {c2['cat_hi']:.3f}]) vs SOTA {SOTA}")

    # 3. artifact --------------------------------------------------------------------------------------------------
    v3_before = os.path.getmtime(V3_PATH)
    model = TransformedTargetRegressor(regressor=clone(T.hgb()), func=np.log1p, inverse_func=np.expm1).fit(X, y)
    bundle = dict(
        model=model, features=FEATS, target=TARGET,
        trained_on=(f"cone_replicates_fsri.csv replicate-mean pHRR (no did_not_ignite / early_stop) + fixed MCC "
                    f"({T.EXTRAS['hr']}, {T.EXTRAS['onset']}) + props + HFM, all {len(df)} rows / {len(np.unique(mat))} "
                    f"materials (64 groups: PlasticC aliased to HIPS for grouping only)"),
        validation=(f"train_phrr_promote_c2.py (c2 = HistGradientBoostingRegressor, log1p target, exp043 features): repeated "
                    f"GroupKFold 5x10 on merged material groups, pooled OOF R2={c2['rkf_mean']:.4f} (repeat-0 material-bootstrap "
                    f"95% CI [{c2['ci_lo']:.3f}, {c2['ci_hi']:.3f}]; CI over repeat-mean OOF [{c2['pool_lo']:.3f}, "
                    f"{c2['pool_hi']:.3f}]); paired vs exp043 on merged groups {wins}/10, delta {delta:+.4f}; "
                    f"category-grouped (family held out) GKF {c2['cat_mean']:.3f} [{c2['cat_lo']:.3f}, {c2['cat_hi']:.3f}]. "
                    f"Label (H5): {H5}. Comparable with pomazi_2026: pending-human."),
        sota_reference="pomazi_2026 Peak HRR real held-out RFR R2=0.30 (benchmarks/sota_board.csv); " + H5)
    joblib.dump(bundle, MODEL_PATH)
    chk = joblib.load(MODEL_PATH)
    assert set(chk) == set(joblib.load(V3_PATH)), "bundle keys differ from phrr_rf_v3"
    p = chk["model"].predict(df[chk["features"]].to_numpy(float))
    assert p.max() > 100, "saved model must predict kW/m2, not log units"
    assert os.path.getmtime(V3_PATH) == v3_before, "phrr_rf_v3.joblib was modified"
    print(f"\nSaved {os.path.relpath(MODEL_PATH, ROOT)} (keys {sorted(chk)}); in-sample R2 {r2_score(y, p):.4f} (sanity only)")
    print("predict on 3 rows (kW/m2):", np.round(p[:3], 1).tolist(), "actual:", np.round(y[:3], 1).tolist())

    # 4. result JSONs ----------------------------------------------------------------------------------------------
    def write(slug, rr, hp, model_name, paired_vs, pdelta, pwins, note):
        res = dict(slug=slug, target="Peak HRR (kW/m2)", round_config=CONFIG,
                   dataset="data/v2 load(replicate_target=True), --bc extras, has_mcc rows (cone_replicates_fsri replicate-mean pHRR); "
                           "groups = material with PlasticC -> HIPS (64 groups)",
                   features=FEATS, model=model_name, hyperparameters=hp, code="data/v2/train_phrr_promote_c2.py",
                   n_rows=int(rr["n_rows"]), n_groups=int(rr["n_groups"]), protocol="rgkf5x10",
                   rkf_mean=round(float(rr["rkf_mean"]), 4), rkf_std=round(float(rr["rkf_std"]), 4),
                   ci_lo=round(float(rr["ci_lo"]), 4), ci_hi=round(float(rr["ci_hi"]), 4), gss_mean=round(float(rr["gss_mean"]), 4),
                   paired_vs=paired_vs, paired_delta=round(pdelta, 4), paired_wins=pwins,
                   sota=SOTA, sota_source="pomazi_2026", status="ok", blocked_reason="", note=note,
                   pooled_ci_lo=round(float(rr["pool_lo"]), 4), pooled_ci_hi=round(float(rr["pool_hi"]), 4),
                   category_grouped_mean=round(float(rr["cat_mean"]), 4), category_grouped_ci_lo=round(float(rr["cat_lo"]), 4),
                   category_grouped_ci_hi=round(float(rr["cat_hi"]), 4))
        assert len(note) <= 300
        path = os.path.join(RESULTS, f"{slug}.json")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {os.path.relpath(path, ROOT)}")

    os.makedirs(RESULTS, exist_ok=True)
    write("phrr_promote_inc_merged", inc, "rf(): n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0",
          "RandomForestRegressor", "exp043 (reference, own merged-group rescoring)", 0.0, "n/a",
          f"exp043 config (RF, log target) rescored with PlasticC->HIPS merged groups; reference. Category-grouped "
          f"{inc['cat_mean']:.3f} [{inc['cat_lo']:.3f}, {inc['cat_hi']:.3f}]. Pooled CI [{inc['pool_lo']:.3f}, {inc['pool_hi']:.3f}].")
    write("phrr_promote_c2_merged", c2,
          "hgb(): learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, random_state=0",
          "HistGradientBoostingRegressor (log1p target)", "exp043 (merged groups)", delta, f"{wins}/10",
          f"c2 on merged groups, new pHRR number of record. Category-grouped {c2['cat_mean']:.3f} "
          f"[{c2['cat_lo']:.3f}, {c2['cat_hi']:.3f}] (CI spans 0.30: not ahead). Pooled CI [{c2['pool_lo']:.3f}, {c2['pool_hi']:.3f}]. "
          f"Comparable: pending-human.")
    print(f"\nruntime {time.time() - t0:.0f}s")
    return c2, inc, wins, delta


if __name__ == "__main__":
    main()
