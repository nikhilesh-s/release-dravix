#!/usr/bin/env python3
"""Pomazi-reconstruction round 1b (config: experiments/configs/pomazi_round1b.json). Pre-declared; no tuning.
Changes vs round 1: tolerance formulation groups, hemp excluded, PRIMARY = leave-one-system-out."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import eval_protocol as ep
import train_pomazi_round1 as r1  # reuse FEATS, COMP, make, hp, TARGETS (candidates unchanged)

CFG = json.load(open(ROOT / "experiments/configs/pomazi_round1b.json"))
BARS = json.load(open(ROOT / "experiments/configs/pomazi_round1.json"))["targets"]
FEATS, COMP, TARGETS, make, hp = r1.FEATS, r1.COMP, r1.TARGETS, r1.make, r1.hp
CODE = "data/v2/train_pomazi_round1b.py"
CIDS = ["rf", "gbr_theirs", "hgb", "ridge"]


def tol_groups(C, tol=0.1):
    C = C + 0.0  # removes -0.0
    adj = (np.abs(C[:, None, :] - C[None, :, :]).max(axis=2) < tol)
    return connected_components(csr_matrix(adj), directed=False)[1]


def build(tcol):
    d = pd.read_csv(ROOT / "data/v2/pomazi_recon_inputs.csv")
    d = d[(d.has_inputs == True) & d[tcol].notna() & d.dup_of.isna() & (d.fibre_type != "hemp")].reset_index(drop=True)
    grp = tol_groups(d[COMP].values.astype(float))
    system = d.system.fillna("other").values
    return d[FEATS].values.astype(float), d[tcol].values.astype(float), grp, system, d.heat_flux_kWm2.values


def run(cid, tcol, X, y, grp, system, hf):
    m = make(cid, tcol)
    ra = []
    for s in range(50):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s)
        ra.append(r2_score(y[te], clone(m).fit(X[tr], y[tr]).predict(X[te])))
    ra = np.array(ra)
    # PRIMARY: leave-one-system-out
    oc = np.empty(len(y))
    for sy in np.unique(system):
        te = system == sy
        oc[te] = clone(m).fit(X[~te], y[~te]).predict(X[te])
    plo, phi = ep.bootstrap_ci(y, oc, grp)
    hfr = {int(h): r2_score(y[hf == h], oc[hf == h]) for h in (25, 50) if (hf == h).sum() > 1}
    # SECONDARY: repeated GroupKFold 5x10 on tolerance groups
    rkf, oof = ep.repeated_group_kfold(m, X, y, grp)
    slo, shi = ep.bootstrap_ci(y, oof[0], grp)
    qlo, qhi = ep.bootstrap_ci(y, oof.mean(axis=0), grp)
    return dict(target=tcol, cid=cid, P_r2=r2_score(y, oc), P_lo=plo, P_hi=phi, hf25=hfr.get(25, np.nan), hf50=hfr.get(50, np.nan),
                S_mean=rkf.mean(), S_std=rkf.std(), S0_lo=slo, S0_hi=shi, Sp_lo=qlo, Sp_hi=qhi,
                A_mean=ra.mean(), A_p025=np.percentile(ra, 2.5), A_p975=np.percentile(ra, 97.5),
                bar=BARS[tcol], n_rows=len(y), n_groups=len(np.unique(grp)), n_systems=len(np.unique(system)))


def main():
    t0 = time.time()
    rows = []
    for tcol in TARGETS:
        X, y, grp, system, hf = build(tcol)
        print(f"{tcol}: {len(y)} rows / {len(np.unique(grp))} tolerance groups (hemp excluded); systems {dict(zip(*np.unique(system, return_counts=True)))}", flush=True)
        for cid in CIDS:
            r = run(cid, tcol, X, y, grp, system, hf)
            rows.append(r)
            print(f"  {tcol} {cid}: PRIMARY LOSO {r['P_r2']:.3f} [{r['P_lo']:.3f},{r['P_hi']:.3f}] hf25 {r['hf25']:.3f} hf50 {r['hf50']:.3f} | "
                  f"SEC {r['S_mean']:.3f}±{r['S_std']:.3f} rep0 [{r['S0_lo']:.3f},{r['S0_hi']:.3f}] pooled [{r['Sp_lo']:.3f},{r['Sp_hi']:.3f}] | "
                  f"A {r['A_mean']:.3f} [{r['A_p025']:.3f},{r['A_p975']:.3f}] ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "experiments/runs/pomazi_round1b_all.csv", index=False)
    sel = df.loc[df.groupby("target").P_r2.idxmax()]
    print("\nSELECTED (highest PRIMARY leave-one-system-out R2 per target)")
    print(f"{'target':12s} {'cand':11s} {'rows/grp':>9s} {'PRIMARY R2 [CI]':24s} {'SEC mean±std':13s} {'rep0 CI':16s} {'pooled CI':16s} {'A mean [p2.5,p97.5]':22s} {'bar':>5s}")
    for _, r in sel.iterrows():
        print(f"{r.target:12s} {r.cid:11s} {int(r.n_rows):4d}/{int(r.n_groups):<4d} {r.P_r2:6.3f} [{r.P_lo:6.3f},{r.P_hi:6.3f}]  "
              f"{r.S_mean:6.3f}±{r.S_std:5.3f}  [{r.S0_lo:6.3f},{r.S0_hi:6.3f}] [{r.Sp_lo:6.3f},{r.Sp_hi:6.3f}] "
              f"{r.A_mean:6.3f} [{r.A_p025:6.3f},{r.A_p975:6.3f}] {r.bar:5.2f}")
    print("\nTRIGGER CHECK (selected candidate per target; rule: primary ci_lo > bar, secondary pooled ci_lo > bar, A mean > bar)")
    trig = []
    for _, r in sel.iterrows():
        a, b, c = r.P_lo > r.bar, r.Sp_lo > r.bar, r.A_mean > r.bar
        print(f"{r.target} {r.cid}: primary ci_lo {r.P_lo:.3f} > {r.bar}: {a}; secondary pooled ci_lo {r.Sp_lo:.3f} > {r.bar}: {b}; A mean {r.A_mean:.3f} > {r.bar}: {c}")
        if a: trig.append(f"TRIGGER: {r.target} leave_one_system_out primary ci_lo {r.P_lo:.3f} vs {r.bar}")
        if b: trig.append(f"TRIGGER: {r.target} gkf_tolerance5x10 pooled ci_lo {r.Sp_lo:.3f} vs {r.bar}")
        if c: trig.append(f"TRIGGER: {r.target} protocol_A mean {r.A_mean:.3f} vs {r.bar}")
    print("\n" + ("\n".join(trig) if trig else "No exit triggers fired for selected candidates."))
    print("\nOther candidates (not selected) that would satisfy a trigger condition (informational):")
    for _, r in df.iterrows():
        if r.name in sel.index: continue
        fl = [n for n, v in (("primary_ci_lo", r.P_lo > r.bar), ("secondary_pooled_ci_lo", r.Sp_lo > r.bar), ("A_mean", r.A_mean > r.bar)) if v]
        if fl: print(f"  {r.target} {r.cid}: {', '.join(fl)}")
    for _, r in sel.iterrows():
        tcol = r.target
        label = TARGETS[tcol][1]
        slug = f"pomazi_r1b_{tcol}_{r.cid}"
        note = (f"Recon, hemp excl, tolerance groups. Primary leave-one-system-out R2; secondary GKF5x10 pooled CI "
                f"[{r.Sp_lo:.3f},{r.Sp_hi:.3f}]; A (50x80/20) {r.A_mean:.3f}. "
                f"{'gbr_theirs=sklearn stand-in for XGBoost.' if r.cid == 'gbr_theirs' else ''}")[:300]
        out = dict(slug=slug, target=label, round_config="experiments/configs/pomazi_round1b.json",
                   dataset="data/v2/pomazi_recon_inputs.csv; has_inputs & target notna & dup_of empty & fibre_type != hemp",
                   features=FEATS, model=dict(rf="RandomForestRegressor", gbr_theirs="GradientBoostingRegressor",
                                              hgb="HistGradientBoostingRegressor", ridge="Ridge")[r.cid],
                   hyperparameters=hp(r.cid, tcol), code=CODE, n_rows=int(r.n_rows), n_groups=int(r.n_systems),
                   protocol="leave_one_system_out", rkf_mean=round(r.P_r2, 4), rkf_std=0.0,
                   ci_lo=round(r.P_lo, 4), ci_hi=round(r.P_hi, 4), gss_mean=round(r.A_mean, 4),
                   paired_vs=None, paired_delta=None, paired_wins=None, sota=float(r.bar), sota_source="pomazi_2026",
                   status="ok", blocked_reason="", note=note)
        json.dump(out, open(ROOT / f"experiments/results/{slug}.json", "w"), indent=2)
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
