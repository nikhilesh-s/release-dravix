#!/usr/bin/env python3
"""wave2_exam_round1 -- ONE-TIME exam of the current engine on the sealed wave-2 rows.
Follows experiments/configs/wave2_exam_round1.json exactly (nothing tuned). Run once.

  uv run python data/v2/wave2_exam_round1.py --population   # counts only; no prediction, no score
  uv run python data/v2/wave2_exam_round1.py | tee experiments/runs/wave2_exam_round1.log
"""
import argparse, glob, hashlib, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/v2"))
sys.path.insert(0, str(ROOT / "engine"))
import train_litcone_gt_round1 as T  # noqa: E402

SNAP_SHA = "0b2af8e1817bdda1a9b1980e481b44d75ac62763f032696090aacd5545e05f99"
TRAIN_N = {"TTI": 3368, "pHRR": 4017, "THR": 3355, "avg_HRR": 759}  # litcone_gt_round1_FINAL.log
SEALED = ["TTI", "THR", "avg_HRR"]
OUT = ROOT / "experiments/results/wave2_exam_round1.json"
N_BOOT, SEED = 2000, 0


def population():
    snap = pd.read_csv(T.SNAP, low_memory=False)
    assert hashlib.sha256(T.SNAP.read_bytes()).hexdigest() == SNAP_SHA, "training snapshot changed"
    w2_ids = set(pd.concat(pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(T.PARTS / "oa_staging/rows_oa_w*.csv"))))["pmcid"])
    d = T.load_rows()  # same merge, figure exclusion, numeric parsing, paper-level + key dedupe as the snapshot build
    w = d[d["pmcid"].isin(w2_ids)]
    c = {"wave2_after_dedupe": len(w), "wave2_papers_after_dedupe": w["group"].nunique()}
    w = w[~w["material_form"].fillna("").isin(T.EXCLUDED_FORMS)]
    c["after_form_filter"], c["papers_after_form_filter"] = len(w), w["group"].nunique()
    sd = set(snap["doi"].dropna().str.lower().str.strip()) - {""}
    pairs = set(zip(snap["doi"].fillna("").str.lower().str.strip(), snap["formulation_label"].fillna("")))
    leak = (w["group"].isin(set(snap["group"])) | w["doi"].isin(sd) | w["pmcid"].isin(set(snap["pmcid"].dropna()))
            | pd.Series([p in pairs for p in zip(w["doi"], w["formulation_label"].fillna(""))], index=w.index))
    c["dropped_paper_leak_rows"], c["dropped_paper_leak_papers"] = int(leak.sum()), w.loc[leak, "group"].nunique()
    w = w[~leak].reset_index(drop=True)
    c["exam_rows"], c["exam_papers"] = len(w), w["group"].nunique()
    for t, col in T.TARGETS.items():
        m = (w[col] > 0) & w["heat_flux_kWm2"].notna()
        c[f"n_{t}"], c[f"papers_{t}"] = int(m.sum()), w.loc[m, "group"].nunique()
    return snap, w, c


def control_value(w, col):
    """Same paper's is_control row, same target, same heat flux (thickness-matched if several); NaN on control rows."""
    out = np.full(len(w), np.nan)
    ctrl = w[(w["is_control"] == 1) & (w[col] > 0)]
    for i, (g, q, th, ic) in enumerate(zip(w["group"], w["heat_flux_kWm2"], w["thickness_mm"], w["is_control"])):
        if ic == 1:
            continue
        c = ctrl[(ctrl["group"] == g) & (ctrl["heat_flux_kWm2"] == q)]
        if len(c) > 1 and not np.isnan(th):
            c2 = c[c["thickness_mm"] == th]
            c = c2 if len(c2) else c
        if len(c):
            out[i] = c[col].astype(float).mean()
    return out


def metrics(y, preds, g):
    """Point R2/MAE/Spearman for each named prediction + paper-bootstrap 95% CI; paired dR2 vs engine."""
    groups = np.unique(g)
    rows_of = {k: np.flatnonzero(g == k) for k in groups}
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([rows_of[k] for k in rng.choice(groups, len(groups))]) for _ in range(N_BOOT)]
    f = {"R2": r2_score, "MAE": mean_absolute_error, "Spearman": lambda a, b: spearmanr(a, b).statistic}
    out = {}
    for name, p in preds.items():
        r = {"n_rows": len(y), "n_papers": len(groups)}
        for k, fn in f.items():
            bs = [fn(y[i], p[i]) for i in boots]
            r[k], r[k + "_ci"] = float(fn(y, p)), [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
        if name != "engine":
            d = [r2_score(y[i], preds["engine"][i]) - r2_score(y[i], p[i]) for i in boots]
            r["engine_minus_this_R2"] = float(out["engine"]["R2"] - r["R2"])
            r["engine_minus_this_R2_ci"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    a = ap.parse_args()
    snap, w, c = population()
    print("population:", json.dumps(c))
    if a.population:
        return
    assert not OUT.exists(), f"{OUT} exists: the exam runs once"
    assert not (set(w["doi"]) - {""}) & set(snap["doi"].dropna()), "leak"
    t0 = time.time()
    import dravix_engine as E
    E.BUNDLE = Path(tempfile.mkdtemp()) / "dravix_engine.joblib"  # fresh fit from the snapshot; repo bundle untouched
    E.train()
    E._B = None
    import joblib
    b = joblib.load(E.BUNDLE)
    assert {k: v["n"] for k, v in b["models"].items()} == TRAIN_N
    pred = pd.DataFrame(E.predict(w.to_dict("records")))
    res = {}
    for t, col in T.TARGETS.items():
        m = ((w[col] > 0) & w["heat_flux_kWm2"].notna()).to_numpy()
        tr = snap[(snap[col] > 0) & snap["heat_flux_kWm2"].notna()]
        med = tr.groupby("matrix_canonical")[col].median()
        y, g, sub = w.loc[m, col].to_numpy(float), w.loc[m, "group"].to_numpy(), w[m]
        pe = pred.loc[m, t].to_numpy(float)
        pm = sub["matrix_canonical"].map(med).fillna(tr[col].median()).to_numpy(float)
        res[t] = {"sealed": t in SEALED, "matrices_unseen_in_training": int((~sub["matrix_canonical"].isin(med.index)).sum()),
                  "all": metrics(y, {"engine": pe, "matrix_median": pm}, g)}
        cv = control_value(w, col)[m]
        k = ~np.isnan(cv)
        if k.sum() >= 10 and len(np.unique(g[k])) >= 5:
            res[t]["noncontrol_with_control"] = metrics(y[k], {"engine": pe[k], "matrix_median": pm[k], "neat_control": cv[k]}, g[k])
        print(f"\n=== {t} ({'SEALED' if t in SEALED else 'not sealed; engine unseen'}) ===")
        for blk, r in res[t].items():
            if isinstance(r, dict):
                for name, v in r.items():
                    print(f"  {blk:24s} {name:13s} n={v['n_rows']:4d}/{v['n_papers']:3d}p  R2 {v['R2']:7.3f} "
                          f"[{v['R2_ci'][0]:.3f}, {v['R2_ci'][1]:.3f}]  MAE {v['MAE']:8.2f} [{v['MAE_ci'][0]:.1f}, {v['MAE_ci'][1]:.1f}]  "
                          f"rho {v['Spearman']:.3f} [{v['Spearman_ci'][0]:.3f}, {v['Spearman_ci'][1]:.3f}]"
                          + (f"  dR2(engine-this) {v['engine_minus_this_R2']:.3f} [{v['engine_minus_this_R2_ci'][0]:.3f}, "
                             f"{v['engine_minus_this_R2_ci'][1]:.3f}]" if "engine_minus_this_R2" in v else ""))
    OUT.write_text(json.dumps({
        "slug": "wave2_exam_round1", "target": "TTI, THR, avg HRR (sealed); pHRR (engine-unseen, reported separately)",
        "round_config": "experiments/configs/wave2_exam_round1.json", "code": "data/v2/wave2_exam_round1.py",
        "dataset": "wave-2 OA rows (0fcca80) after load_rows dedupe, textile/coated excluded, papers in the training snapshot dropped",
        "model": "engine/dravix_engine.py primary blend, fit on cone_lit_benchmark_v1.csv (sha256 0b2af8e1...)",
        "protocol": "single held-out exam, collected after the model was built; paper bootstrap 2000",
        "population": c, "results": res, "status": "ok", "runtime_s": round(time.time() - t0),
        "note": "Run once. Quote as 'held-out, collected after the model was built'. Site wording pending Nik."}, indent=2))
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
