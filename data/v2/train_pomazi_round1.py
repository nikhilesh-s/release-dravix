#!/usr/bin/env python3
"""Pomazi-reconstruction round 1 (config: experiments/configs/pomazi_round1.json). Pre-declared; no tuning."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import eval_protocol as ep

CFG = json.load(open(ROOT / "experiments/configs/pomazi_round1.json"))
FEATS = ["epoxy_ratio_pct", "hardener_ratio_pct", "fr1_ratio_pct", "fr2_ratio_pct", "C_pct", "H_pct", "O_pct",
         "N_pct", "P_pct", "aliphatic_pct", "cycloaliphatic_pct", "aromatic_pct", "heat_flux_kWm2", "fibre_wtpct_input"]
# The "8 composition inputs": the config counts (46/38 formulations) are reproduced only by the C,H,O,N,P,aliphatic,
# cycloaliphatic,aromatic columns (the first-8 FEATS incl. ratios give 41/33).
COMP = ["C_pct", "H_pct", "O_pct", "N_pct", "P_pct", "aliphatic_pct", "cycloaliphatic_pct", "aromatic_pct"]
TARGETS = {  # col: (bar key, result-slug name, label, expected rows, expected groups, Table-1 GBR settings)
    "phrr_kWm2": ("phrr", "Peak HRR (kW/m2)", 75, 46, dict(n_estimators=500, learning_rate=0.2, max_depth=3, subsample=0.8, max_features=0.7)),
    "tti_s": ("tti", "TTI (s)", 63, 38, dict(n_estimators=500, learning_rate=0.2, max_depth=3, subsample=0.8, max_features=0.9)),
    "thr_MJm2": ("thr", "THR (MJ/m2)", 63, 38, dict(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.9, max_features=0.9)),
    "residue_pct": ("cr", "Char residue (%)", 63, 38, dict(n_estimators=200, learning_rate=0.2, max_depth=3, subsample=0.7, max_features=0.9)),
}
BARS = CFG["targets"]
CODE = "data/v2/train_pomazi_round1.py"


def make(cid, tcol):
    if cid == "rf":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0)
    if cid == "gbr_theirs":
        return make_pipeline(SimpleImputer(strategy="median"),
                             GradientBoostingRegressor(random_state=0, **TARGETS[tcol][4]))
    if cid == "hgb":
        return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=7, min_samples_leaf=5, random_state=0)
    if cid == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def hp(cid, tcol):
    return {"rf": "RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0), NaN native",
            "gbr_theirs": f"SimpleImputer(median, in-fold)+sklearn GradientBoostingRegressor {TARGETS[tcol][4]}, random_state=0 (stand-in for XGBoost Table 1; xgboost not installed)",
            "hgb": "HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=7, min_samples_leaf=5, random_state=0)",
            "ridge": "SimpleImputer(median, in-fold)+StandardScaler+Ridge(alpha=1.0)"}[cid]


def build(tcol):
    d = pd.read_csv(ROOT / "data/v2/pomazi_recon_inputs.csv")
    d = d[(d.has_inputs == True) & d[tcol].notna() & d.dup_of.isna()].reset_index(drop=True)
    form = d[COMP].round(1).astype(str).agg("|".join, axis=1).values
    system = d.system.fillna("other").values
    return d[FEATS].values.astype(float), d[tcol].values.astype(float), form, system


def run(cid, tcol, X, y, form, system):
    m = make(cid, tcol)
    ra = []
    for s in range(50):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s)
        ra.append(r2_score(y[te], clone(m).fit(X[tr], y[tr]).predict(X[te])))
    ra = np.array(ra)
    rkf, oof = ep.repeated_group_kfold(m, X, y, form)
    lo, hi = ep.bootstrap_ci(y, oof[0], form)
    oc = np.empty(len(y))
    for sy in np.unique(system):
        te = system == sy
        oc[te] = clone(m).fit(X[~te], y[~te]).predict(X[te])
    return dict(target=tcol, cid=cid, A_mean=ra.mean(), A_p025=np.percentile(ra, 2.5), A_p975=np.percentile(ra, 97.5),
                B_mean=rkf.mean(), B_std=rkf.std(), ci_lo=lo, ci_hi=hi, C_r2=r2_score(y, oc), bar=BARS[tcol],
                n_rows=len(y), n_groups=len(np.unique(form)))


def main():
    t0 = time.time()
    rows = []
    for tcol, (_, _, nr, ng, _) in TARGETS.items():
        X, y, form, system = build(tcol)
        assert (len(y), len(np.unique(form))) == (nr, ng), (tcol, len(y), len(np.unique(form)))
        print(f"{tcol}: {len(y)} rows / {len(np.unique(form))} formulations OK", flush=True)
        for cid in ["rf", "gbr_theirs", "hgb", "ridge"]:
            r = run(cid, tcol, X, y, form, system)
            rows.append(r)
            print(f"  {tcol} {cid}: A {r['A_mean']:.3f} [{r['A_p025']:.3f},{r['A_p975']:.3f}] B {r['B_mean']:.3f}±{r['B_std']:.3f} "
                  f"CI [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] C {r['C_r2']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "experiments/runs/pomazi_round1_all.csv", index=False)
    sel = df.loc[df.groupby("target").B_mean.idxmax()]
    trig = []
    print("\nSELECTED (highest protocol-B mean per target)")
    print(f"{'target':12s} {'cand':11s} {'A mean [p2.5, p97.5]':24s} {'B mean±std':14s} {'B CI':16s} {'C R2':>7s} {'bar':>5s}")
    for _, r in sel.iterrows():
        print(f"{r.target:12s} {r.cid:11s} {r.A_mean:6.3f} [{r.A_p025:6.3f},{r.A_p975:6.3f}]   "
              f"{r.B_mean:6.3f}±{r.B_std:5.3f}  [{r.ci_lo:6.3f},{r.ci_hi:6.3f}] {r.C_r2:7.3f} {r.bar:5.2f}")
    print()
    for _, r in sel.iterrows():
        b, a = r.ci_lo > r.bar, r.A_mean > r.bar
        print(f"{r.target}: protocol-B ci_lo ({r.ci_lo:.3f}) > bar ({r.bar}): {b}; protocol-A mean ({r.A_mean:.3f}) > bar: {a}")
        if b: trig.append(f"TRIGGER: {r.target} B_ci_lo {r.ci_lo:.3f} vs bar {r.bar}")
        if a: trig.append(f"TRIGGER: {r.target} A_mean {r.A_mean:.3f} vs bar {r.bar}")
    print("\n" + ("\n".join(trig) if trig else "No exit triggers fired."))
    print("Round-2 eligibility (best B mean within 0.10 of bar): " +
          ", ".join(f"{r.target}={r.B_mean >= r.bar - 0.10 and r.B_mean < r.bar + 1e9}" for _, r in sel.iterrows()))
    for _, r in sel.iterrows():
        tcol = r.target
        _, label, nr, ng, _ = TARGETS[tcol]
        slug = f"pomazi_r1_{tcol}_{r.cid}"
        note = (f"Independent reconstruction ({nr} rows, {ng} formulations). A (50x80/20 random) {r.A_mean:.3f} "
                f"[{r.A_p025:.3f},{r.A_p975:.3f}]; C (leave-system-out) {r.C_r2:.3f}. B is primary. "
                f"{'gbr_theirs=sklearn stand-in for XGBoost.' if r.cid == 'gbr_theirs' else ''}")[:300]
        out = dict(slug=slug, target=label, round_config="experiments/configs/pomazi_round1.json",
                   dataset="data/v2/pomazi_recon_inputs.csv; has_inputs & target notna & dup_of empty",
                   features=FEATS, model=dict(rf="RandomForestRegressor", gbr_theirs="GradientBoostingRegressor",
                                              hgb="HistGradientBoostingRegressor", ridge="Ridge")[r.cid],
                   hyperparameters=hp(r.cid, tcol), code=CODE, n_rows=int(r.n_rows), n_groups=int(r.n_groups),
                   protocol="gkf_formulation5x10", rkf_mean=round(r.B_mean, 4), rkf_std=round(r.B_std, 4),
                   ci_lo=round(r.ci_lo, 4), ci_hi=round(r.ci_hi, 4), gss_mean=round(r.A_mean, 4),
                   paired_vs=None, paired_delta=None, paired_wins=None, sota=float(r.bar), sota_source="pomazi_2026",
                   status="ok", blocked_reason="", note=note)
        json.dump(out, open(ROOT / f"experiments/results/{slug}.json", "w"), indent=2)
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
