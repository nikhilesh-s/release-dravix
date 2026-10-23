#!/usr/bin/env python3
"""Dravix V2 -- pHRR round 1 (pre-declared in experiments/configs/phrr_round1.json, 2026-10-01).

ONE round, five candidates, no tuning, then pHRR (FSRI) is parked. Data = train_phrr_v3_robust.load(replicate_target=True)
with the --bc extras, rows where has_mcc (191 rows / 65 materials). Protocol = eval_protocol (repeated GroupKFold 5x10 by
material, pooled OOF; GroupShuffleSplit x10; material-bootstrap CI on repeat-0 OOF).

Step 1 reproduces the incumbent exp043 A1 (rf(), BASE+PHYS, log1p target): rkf_mean 0.567, CI [0.283, 0.836]. If that fails
the script stops (blocked) and runs no candidate.

  c1  incumbent features minus MCC extras (mcc_total_hr_kJg, mcc_t_peak_C, mcc_t_onset_C); rf(); log target
  c2  hgb() (fixed settings) on the incumbent features; log target
  c3  equal-weight average of incumbent RF and c2 HGB predictions (raw units after expm1), per fold
  c4  c2 with monotonic_cst +1 on heat_flux_kWm2 and mcc_peak_hrr_Wg, 0 elsewhere
  c5  two-stage: in-fold imputation of the five MCC features (one hgb() per feature, fit on the training-fold materials that
      have the feature, predictors = density, thickness, mass/area, category one-hot), then incumbent RF on all 76 materials.
      Scored on all 221 rows / 76 materials and on the 191 incumbent rows. For identical splits on the 191 rows, each repeat
      uses the incumbent's own GroupKFold(5, shuffle, random_state=r) on the 191 rows; the 11 extra (no-MCC) materials are
      folded by a second GroupKFold with the same seed and their fold k is added to the incumbent's test fold k. So the
      191-row score is paired with exp043 on identical test materials per fold; extra materials only add training rows.

Selection rule (config, applied mechanically): highest repeated-GKF mean among c1-c4 and c5 (191 rows). Promote only if it
beats exp043 in >= 8/10 paired repeats AND its CI lower bound >= exp043's. Supplementary only (never promote): CI pooled over
all 10 repeats, share of bootstrap resamples with R2 > 0.30.

Rerun: python3 data/v2/train_phrr_round1.py   (writes experiments/results/phrr_r1_*.json)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_phrr_v3_robust as T  # sets OMP_NUM_THREADS=1 before numpy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

from eval_protocol import bootstrap_ci, grouped_shuffle_r2, repeated_group_kfold

T.EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")  # exactly what --bc does

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")
CONFIG = "experiments/configs/phrr_round1.json"
TARGET = T.TARGET
SOTA = T.SOTA
FEATS = T.BASE + T.PHYS
MCC_EXTRAS = ["mcc_total_hr_kJg", "mcc_t_peak_C", "mcc_t_onset_C"]
FEATS_C1 = [f for f in FEATS if f not in MCC_EXTRAS]
MCC_COLS = ["mcc_peak_hrr_Wg", "mcc_char_yield_pct"] + MCC_EXTRAS
IMP_NUM = ["density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]
INC = dict(rkf_mean=0.567, ci_lo=0.283, ci_hi=0.836)


class AvgRfHgb(BaseEstimator, RegressorMixin):
    """c3: both models fit on log1p(y); prediction = mean of the two expm1'd predictions (raw units)."""

    def fit(self, X, y):
        self.m_ = [clone(T.rf()).fit(X, np.log1p(y)), clone(T.hgb()).fit(X, np.log1p(y))]
        return self

    def predict(self, X):
        return np.mean([np.expm1(m.predict(X)) for m in self.m_], axis=0)


def hgb_mono():
    m = T.hgb()
    m.set_params(monotonic_cst=[1 if f in ("heat_flux_kWm2", "mcc_peak_hrr_Wg") else 0 for f in FEATS])
    return m


def r2_rows(y, P):
    return T.r2_rows(y, P)


def score(model, df, feats, log_target):
    """eval_protocol components (same as report()/T.evaluate), without the category-grouped extra."""
    X, y, mat = df[feats].to_numpy(float), df[TARGET].to_numpy(float), df["material"].to_numpy()
    gss = grouped_shuffle_r2(model, X, y, mat, log_target=log_target)
    rkf, oof = repeated_group_kfold(model, X, y, mat, log_target=log_target)
    lo, hi = bootstrap_ci(y, oof[0], mat)
    return dict(gss_mean=gss.mean(), gss_std=gss.std(), rkf=rkf, rkf_mean=rkf.mean(), rkf_std=rkf.std(),
                ci_lo=lo, ci_hi=hi, oof=oof, y=y, mat=mat, n_rows=len(y), n_groups=len(np.unique(mat)))


# ---- c5: in-fold imputation ---------------------------------------------------------------------------------------
def impute_fit_predict(dfa, tr, te):
    """Fit MCC imputers on the training-fold materials only, fill NaN MCC cells in train and test, fit the incumbent RF
    (log1p target) on all training rows, predict the test rows (raw units). Test targets are never touched."""
    tr_mat = dfa.iloc[tr].drop_duplicates("material")  # one row per training material (MCC is material-level)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr_mat[["category"]])

    def design(d):
        return np.hstack([d[IMP_NUM].to_numpy(float), enc.transform(d[["category"]])])

    imputers = {}
    for c in MCC_COLS:
        seen = tr_mat[tr_mat[c].notna()]
        imputers[c] = clone(T.hgb()).fit(design(seen), seen[c].to_numpy(float))
    Xs = {}
    for name, idx in (("tr", tr), ("te", te)):
        d = dfa.iloc[idx]
        X = d[FEATS].copy()
        D = design(d)
        for c in MCC_COLS:
            miss = X[c].isna().to_numpy()
            if miss.any():
                X.loc[miss, c] = imputers[c].predict(D[miss])
        Xs[name] = X.to_numpy(float)
    m = clone(T.rf()).fit(Xs["tr"], np.log1p(dfa.iloc[tr][TARGET].to_numpy(float)))
    return np.expm1(m.predict(Xs["te"]))


def run_c5(dfa, n191):
    """dfa = the 191 incumbent rows first, then the no-MCC rows. Returns dict with 191-row and all-row results."""
    y, mat = dfa[TARGET].to_numpy(float), dfa["material"].to_numpy()
    n = len(dfa)
    assert set(mat[:n191]).isdisjoint(mat[n191:])
    X191 = dfa.iloc[:n191][FEATS].to_numpy(float)
    ex = np.arange(n191, n)
    oof = np.empty((10, n))
    for r in range(10):
        f_inc = list(GroupKFold(5, shuffle=True, random_state=r).split(X191, y[:n191], mat[:n191]))
        f_ex = list(GroupKFold(5, shuffle=True, random_state=r).split(ex, y[ex], mat[ex]))
        for k in range(5):
            te = np.concatenate([f_inc[k][1], ex[f_ex[k][1]]])
            tr = np.setdiff1d(np.arange(n), te)
            assert set(mat[tr]).isdisjoint(mat[te])
            oof[r, te] = impute_fit_predict(dfa, tr, te)
    assert np.isfinite(oof).all()
    # GroupShuffleSplit x10 on the 191 rows (same seeds/test materials as eval_protocol); extra materials always train
    gss = []
    for seed in range(10):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X191, y[:n191], mat[:n191]))
        p = impute_fit_predict(dfa, np.concatenate([tr, ex]), te)
        gss.append(r2_score(y[te], p))
    gss = np.array(gss)
    sub = slice(0, n191)
    out = {}
    for tag, s in (("191", sub), ("all", slice(0, n))):
        yy, mm, oo = y[s], mat[s], oof[:, s]
        rkf = np.array([r2_score(yy, p) for p in oo])
        lo, hi = bootstrap_ci(yy, oo[0], mm)
        out[tag] = dict(rkf=rkf, rkf_mean=rkf.mean(), rkf_std=rkf.std(), ci_lo=lo, ci_hi=hi, oof=oo, y=yy, mat=mm,
                        n_rows=len(yy), n_groups=len(np.unique(mm)), gss_mean=gss.mean(), gss_std=gss.std())
    return out


# ---- supplementary --------------------------------------------------------------------------------------------------
def supplementary(y, oof, mat, n_boot=2000, seed=0):
    """Same material-bootstrap draws as eval_protocol.bootstrap_ci, but scored on all 10 repeats' OOF."""
    rng = np.random.default_rng(seed)
    mats = np.unique(mat)
    rows_of = {m: np.flatnonzero(mat == m) for m in mats}
    pooled, rep0 = [], []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))])
        r2 = r2_rows(y[rows], oof[:, rows])
        pooled.append(r2.mean())
        rep0.append(r2[0])
    pooled, rep0 = np.array(pooled), np.array(rep0)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    return dict(pooled_lo=lo, pooled_hi=hi, share_gt_sota_pooled=float((pooled > SOTA).mean()),
                share_gt_sota_rep0=float((rep0 > SOTA).mean()), rep0_lo=np.percentile(rep0, 2.5),
                rep0_hi=np.percentile(rep0, 97.5))


def row(name, r):
    return (f"{name:34s} {r['rkf_mean']:7.4f} {r['rkf_std']:7.4f}  [{r['ci_lo']:6.3f}, {r['ci_hi']:6.3f}] "
            f"{r['gss_mean']:7.4f}  n={r['n_rows']}/{r['n_groups']}")


def main():
    t0 = time.time()
    full = T.load(replicate_target=True)
    df = full[full["has_mcc"]].reset_index(drop=True)
    extra = full[~full["has_mcc"]].reset_index(drop=True)
    print(f"Round config: {CONFIG}")
    print(f"load(replicate_target=True): {len(full)} rows / {full['material'].nunique()} materials; has_mcc: {len(df)} rows / "
          f"{df['material'].nunique()} materials; no-MCC rows: {len(extra)} / {extra['material'].nunique()} materials")
    assert (len(df), df["material"].nunique()) == (191, 65), "incumbent rows not 191/65"
    print(f"features (incumbent): {FEATS}\nEXTRAS: {T.EXTRAS}")

    # 1. reproduce the incumbent -------------------------------------------------------------------------------------
    inc = score(T.rf(), df, FEATS, True)
    print("\n=== Step 1: reproduce exp043 A1 (V3 rf_phys_log) ===")
    print(row("exp043 A1 (measured)", inc))
    ok = (abs(round(inc["rkf_mean"], 3) - INC["rkf_mean"]) <= 0.002 and abs(round(inc["ci_lo"], 3) - INC["ci_lo"]) <= 0.002
          and abs(round(inc["ci_hi"], 3) - INC["ci_hi"]) <= 0.002)
    print(f"declared: rkf_mean {INC['rkf_mean']}, CI [{INC['ci_lo']}, {INC['ci_hi']}] -> {'REPRODUCED' if ok else 'NOT REPRODUCED'}")
    if not ok:
        print("BLOCKED: incumbent not reproduced; no candidate run.")
        sys.exit(2)

    # 2. candidates --------------------------------------------------------------------------------------------------
    cands = {}
    print("\n=== Step 2: candidates (same splits: eval_protocol seeds, material groups) ===")
    cands["c1"] = dict(short="nomccextras", model="RandomForestRegressor", hp="rf(): n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0",
                       feats=FEATS_C1, r=score(T.rf(), df, FEATS_C1, True))
    cands["c2"] = dict(short="hgb", model="HistGradientBoostingRegressor", hp="hgb(): learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, random_state=0",
                       feats=FEATS, r=score(T.hgb(), df, FEATS, True))
    cands["c3"] = dict(short="rfhgb_avg", model="mean of rf() and hgb() predictions (expm1 raw units)", hp="rf() and hgb() as incumbent / c2, both log1p target",
                       feats=FEATS, r=score(AvgRfHgb(), df, FEATS, False))
    cands["c4"] = dict(short="hgb_mono", model="HistGradientBoostingRegressor monotonic", hp="hgb() + monotonic_cst +1 on heat_flux_kWm2, mcc_peak_hrr_Wg, 0 elsewhere",
                       feats=FEATS, r=score(hgb_mono(), df, FEATS, True))
    dfa = pd.concat([df, extra], ignore_index=True)
    c5 = run_c5(dfa, len(df))
    cands["c5"] = dict(short="impute", model="in-fold MCC imputation (hgb per MCC feature) then RandomForestRegressor",
                       hp="imputers: hgb(); predictors density, thickness, mass/area, category one-hot; then rf(), log1p target",
                       feats=FEATS, r=c5["191"], r_all=c5["all"])

    # paired vs incumbent (repeat r vs repeat r, 191 rows)
    for cid, c in cands.items():
        d = c["r"]["rkf"] - inc["rkf"]
        c["wins"], c["delta"] = int((d > 0).sum()), float(d.mean())
    print(f"\n{'':34s} {'rkf_mean':>7s} {'std':>7s}  {'95% CI (rep 0)':>16s} {'gss':>7s}   paired vs exp043")
    print(row("exp043 incumbent", inc))
    for cid, c in cands.items():
        print(row(f"{cid} {c['short']}", c["r"]) + f"   wins {c['wins']}/10, delta {c['delta']:+.4f}")
    r = c5["all"]
    print(row("c5 impute, ALL rows", r) + "   (not paired; different row set)")
    print(f"  c5 per-repeat R2 on 191 rows: {np.round(cands['c5']['r']['rkf'], 3).tolist()}")
    print(f"  exp043 per-repeat R2:         {np.round(inc['rkf'], 3).tolist()}")

    # 3. selection rule ---------------------------------------------------------------------------------------------
    best = max(cands, key=lambda k: cands[k]["r"]["rkf_mean"])
    b = cands[best]
    beats = b["r"]["rkf_mean"] > inc["rkf_mean"]
    cond_w, cond_ci = b["wins"] >= 8, b["r"]["ci_lo"] >= inc["ci_lo"]
    promote = beats and cond_w and cond_ci
    print("\n=== Step 3: selection rule (config, mechanical) ===")
    print(f"Highest repeated-GKF mean among c1-c5(191 rows): {best} {b['short']} ({b['r']['rkf_mean']:.4f} vs exp043 {inc['rkf_mean']:.4f})")
    print(f"  beats exp043 mean: {beats}; paired wins {b['wins']}/10 (need >= 8): {cond_w}; "
          f"ci_lo {b['r']['ci_lo']:.4f} >= exp043 ci_lo {inc['ci_lo']:.4f} (declared 0.283): {cond_ci}")
    verdict = f"PROMOTE {best} {b['short']}" if promote else "NOT PROMOTED: exp043 stays"
    print(f"VERDICT: {verdict}")
    hi_ci = [k for k, c in cands.items() if c["r"]["ci_lo"] > SOTA]
    print(f"Candidates with ci_lo > {SOTA}: {hi_ci if hi_ci else 'none'}"
          + (f"  -> EXIT condition 5 (claim ambiguity); wording per H5 only" if hi_ci else ""))
    print(f"Incumbent ci_lo {inc['ci_lo']:.4f} > {SOTA}: {inc['ci_lo'] > SOTA}")

    # 4. supplementary (not rules) ----------------------------------------------------------------------------------
    print("\n=== Supplementary (NOT rules; never promote) ===")
    sup_inc = supplementary(inc["y"], inc["oof"], inc["mat"])
    sup_b = supplementary(b["r"]["y"], b["r"]["oof"], b["r"]["mat"])
    for nm, s in (("exp043 incumbent", sup_inc), (f"{best} {b['short']} (selected)", sup_b)):
        print(f"  {nm}: CI pooled over all 10 repeats [{s['pooled_lo']:.3f}, {s['pooled_hi']:.3f}]; share of bootstrap resamples "
              f"with R2 > {SOTA}: {100 * s['share_gt_sota_pooled']:.1f}% (repeat-averaged R2 per resample), "
              f"{100 * s['share_gt_sota_rep0']:.1f}% (repeat-0 R2 per resample, the rule's CI basis; its CI [{s['rep0_lo']:.3f}, {s['rep0_hi']:.3f}])")

    # 5. result JSONs -----------------------------------------------------------------------------------------------
    notes = {
        "c1": "Incumbent minus MCC extras (total HR, T_peak, T_onset); RF, log target. Paired vs exp043 on identical splits.",
        "c2": "HGB (fixed settings) on exp043 features, log target. Paired vs exp043 on identical splits.",
        "c3": "Equal-weight mean of exp043 RF and c2 HGB predictions per fold (raw units). Paired vs exp043 on identical splits.",
        "c4": "c2 HGB with monotonic_cst +1 on heat flux and MCC peak HRR. Paired vs exp043 on identical splits.",
        "c5": (f"In-fold MCC imputation, RF on 221 rows/76 mats; scored on the 191 incumbent rows (paired, identical test mats). "
               f"All-rows R2 {r['rkf_mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]."),
    }
    os.makedirs(RESULTS, exist_ok=True)
    for cid, c in cands.items():
        rr = c["r"]
        dataset = ("data/v2 load(replicate_target=True), --bc extras, has_mcc rows (cone_replicates_fsri replicate-mean pHRR)"
                   if cid != "c5" else
                   "data/v2 load(replicate_target=True), --bc extras, all 221 rows/76 materials for training; scored on the 191 has_mcc rows")
        res = dict(slug=f"phrr_round1_{cid}_{c['short']}", target="Peak HRR (kW/m2)", round_config=CONFIG, dataset=dataset,
                   features=c["feats"], model=c["model"], hyperparameters=c["hp"], code="data/v2/train_phrr_round1.py",
                   n_rows=int(rr["n_rows"]), n_groups=int(rr["n_groups"]), protocol="rgkf5x10",
                   rkf_mean=round(float(rr["rkf_mean"]), 4), rkf_std=round(float(rr["rkf_std"]), 4),
                   ci_lo=round(float(rr["ci_lo"]), 4), ci_hi=round(float(rr["ci_hi"]), 4), gss_mean=round(float(rr["gss_mean"]), 4),
                   paired_vs="exp043", paired_delta=round(c["delta"], 4), paired_wins=f"{c['wins']}/10",
                   sota=SOTA, sota_source="pomazi_2026", status="ok", blocked_reason="", note=notes[cid][:300])
        assert len(res["note"]) <= 300
        path = os.path.join(RESULTS, f"phrr_r1_{cid}_{c['short']}.json")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {os.path.relpath(path, ROOT)}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
