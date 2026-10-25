#!/usr/bin/env python3
"""BME char-residue attempts C1-C3 + TTI attempt T1 on the Pomazi reconstruction
(config: experiments/configs/bme_char_round1.json, committed before scoring)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import eval_protocol as ep
import train_pomazi_round1 as r1
import train_pomazi_round1b as r1b

CFG = json.load(open(ROOT / "experiments/configs/bme_char_round1.json"))
# ponytail: loading assumed wt% of composite; no inorganic FR is present in the 60 residue rows, so only the fibre term is live
YIELD = {"ATH": 0.654, "MDH": 0.69, "silica": 1.0, "SiO2": 1.0, "clay": 1.0, "MMT": 1.0, "glass": 1.0, "ZnO": 1.0, "TiO2": 1.0}
PFR = ("APP", "RDP", "TEDAP")
TOLDY = ["polym11020303", "ppch.19468", "polym8090322", "ma10050467"]
EXCL = "T58|ERYSIS|GE-60|sorbitol polyglycidyl|pentaerythritol polyglycidyl|glycerol polyglycidyl|GFTE|DETDA"
PFR_RE = "phosph|APP|DOPO|RDP|TEDAP|phosphate|phosphonate"


def frame(tcol):
    d = pd.read_csv(ROOT / "data/v2/pomazi_recon_inputs.csv")
    d = d[(d.has_inputs == True) & d[tcol].notna() & d.dup_of.isna() & (d.fibre_type != "hemp")].reset_index(drop=True)
    d["system"] = d.system.fillna("other")
    return d


def static_c1(d):
    f = pd.DataFrame(index=d.index)
    for p in PFR:
        f[f"P_{p}"] = sum(np.where(d[n].fillna("").str.startswith(p) & d[u].fillna("").str.contains("wt% P"), d[l].fillna(0), 0)
                          for n, l, u in (("fr1_name", "fr1_loading", "fr1_unit"), ("fr2_name", "fr2_loading", "fr2_unit")))
    f["is_P_promoter"] = (f[[f"P_{p}" for p in PFR]].sum(axis=1) > 0).astype(float)
    inorg = d.fibre_wtpct.fillna(0) * 1.0
    for n, l in (("fr1_name", "fr1_loading"), ("fr2_name", "fr2_loading")):
        for k, yv in YIELD.items():
            inorg = inorg + np.where(d[n].fillna("").str.contains(k), d[l].fillna(0) * yv, 0)
    f["inorganic_floor_pct"] = inorg
    return f


def neat_char(d, y, tr):
    """Per row: mean residue of training matrix rows of the same formulation key, excluding the row itself."""
    key_full = (d.system + "|" + d.fr1_name.fillna("-") + "|" + d.fr2_name.fillna("-") + "|" + d.P_pct.round(2).astype(str)).values
    is_mat = (d.fibre_wtpct.fillna(0) == 0).values
    no_fr = (d.fr1_name.isna() & d.fr2_name.isna()).values
    sysv = d.system.values
    trm = np.zeros(len(d), bool); trm[tr] = True
    out = np.empty(len(d))
    for i in range(len(d)):
        pool = trm & is_mat; pool[i] = False
        for m in (pool & (key_full == key_full[i]), pool & (sysv == sysv[i]) & no_fr, pool):
            if m.any():
                out[i] = y[m].mean(); break
        else:
            out[i] = np.nan
    return out


def lit_prior_model():
    L = pd.read_csv(ROOT / "data/v2/cone_lit_benchmark_v1.csv", low_memory=False)
    L = L[L.char_residue_pct.between(0, 100)].copy()
    comp = L.components.fillna("") + " " + L.fr_summary.fillna("")
    bad = set(L.doi[comp.str.contains(EXCL, case=False)].dropna()) | {x for x in L.doi.dropna().unique() if any(t in x for t in TOLDY)}
    L = L[~L.doi.isin(bad)]
    X = np.c_[L.heat_flux_kWm2, L.fr_total_wtpct, L.filler_total_wtpct, (L.matrix_canonical == "EP"),
              (L.material_form == "fibre_composite"), comp[L.index].str.contains(PFR_RE, case=False)].astype(float)
    m = clone(r1.make("hgb", "residue_pct")).fit(X, L.char_residue_pct.values)
    return m, dict(n_rows=len(L), n_dois=int(L.doi.nunique()), excluded_dois=sorted(bad))


def lit_feat(d, m):
    X = np.c_[d.heat_flux_kWm2, d.fr1_ratio_pct.fillna(0) + d.fr2_ratio_pct.fillna(0), d.fibre_wtpct.fillna(0),
              np.ones(len(d)), d.fibre_wtpct.fillna(0) > 0, d.P_pct.fillna(0) > 0].astype(float)
    return m.predict(X)


def make_attempt(aid, d, X, y, lit_m=None):
    """Return fp(tr, te) -> predictions on te."""
    if aid == "C1_mass_balance":
        S = static_c1(d).values
        fib = d.fibre_wtpct.fillna(0).values / 100
        def fp(tr, te):
            nc = neat_char(d, y, tr)
            Z = np.c_[X, S, nc, S[:, -1] + (1 - fib) * nc]
            return clone(r1.make("hgb", "residue_pct")).fit(Z[tr], y[tr]).predict(Z[te])
    elif aid == "C2_logit":
        def fp(tr, te):
            p = np.clip(y[tr] / 100, 0.005, 0.995)
            z = clone(r1.make("hgb", "residue_pct")).fit(X[tr], np.log(p / (1 - p))).predict(X[te])
            return 100 / (1 + np.exp(-z))
    elif aid == "C3_lit_prior":
        Z = np.c_[X, lit_feat(d, lit_m)]
        def fp(tr, te):
            return clone(r1.make("hgb", "residue_pct")).fit(Z[tr], y[tr]).predict(Z[te])
    elif aid == "T1_ignition_physics":
        hf = d.heat_flux_kWm2.values
        def fp(tr, te):
            z = clone(r1.make("rf", "tti_s")).fit(X[tr], np.sqrt(y[tr]) * hf[tr]).predict(X[te])
            return (z / hf[te]) ** 2
    elif aid in ("baseline_hgb", "baseline_rf"):
        cid = aid.split("_")[1]
        def fp(tr, te):
            return clone(r1.make(cid, "x")).fit(X[tr], y[tr]).predict(X[te])
    return fp


def score(fp, y, grp, system):
    allr = np.arange(len(y))
    pf = fp(allr, allr)
    flo, fhi = ep.bootstrap_ci(y, pf, grp)
    ra = np.array([r2_score(y[te], fp(tr, te)) for tr, te in
                   (train_test_split(allr, test_size=0.2, random_state=s) for s in range(50))])
    oc = np.empty(len(y))
    for sy in np.unique(system):
        te = np.flatnonzero(system == sy); tr = np.flatnonzero(system != sy)
        oc[te] = fp(tr, te)
    return pf, (flo, fhi), ra, r2_score(y, oc)


def main():
    lit_m, lit_info = lit_prior_model()
    print("lit prior:", {k: v for k, v in lit_info.items() if k != "excluded_dois"}, "excluded", lit_info["excluded_dois"], flush=True)
    plan = {"residue_pct": ("baseline_hgb", ["C1_mass_balance", "C2_logit", "C3_lit_prior"]),
            "tti_s": ("baseline_rf", ["T1_ignition_physics"])}
    for tcol, (base, aids) in ((k, v) for k, v in plan.items() if len(sys.argv) < 2 or k in sys.argv[1:]):
        d = frame(tcol)
        X, y, grp, system, hf = r1b.build(tcol)
        assert np.allclose(y, d[tcol].astype(float).values)
        fb, hb = CFG["bars_full_fit"][tcol], CFG["bars_held_out"][tcol]
        _, _, ra0, loso0 = score(make_attempt(base, d, X, y), y, grp, system)
        print(f"{tcol} {base}: A {ra0.mean():.3f} LOSO {loso0:.3f}", flush=True)
        for aid in aids:
            pf, (flo, fhi), ra, loso = score(make_attempt(aid, d, X, y, lit_m), y, grp, system)
            boot = np.random.default_rng(0).choice(ra, (2000, len(ra))).mean(axis=1)
            alo, ahi = np.percentile(boot, [2.5, 97.5])
            wins = int((ra > ra0).sum())
            res = dict(slug=f"bme_char_round1_{aid}", target=r1.TARGETS[tcol][1], attempt=aid, desc=CFG["attempts"][aid],
                       round_config="experiments/configs/bme_char_round1.json", code="data/v2/train_bme_char_round1.py",
                       n_rows=len(y), n_groups=len(np.unique(grp)), baseline=base,
                       full_fit=dict(protocol="training fit (same protocol as BME: full-dataset fit)", r2=round(r2_score(y, pf), 4),
                                     ci_lo=round(flo, 4), ci_hi=round(fhi, 4), mae=round(mean_absolute_error(y, pf), 3), bme=fb,
                                     higher=r2_score(y, pf) > fb, clear=flo > fb),
                       protocol_A=dict(protocol="random 80/20 x 50, seeds 0-49", mean=round(ra.mean(), 4), mean_ci=[round(alo, 4), round(ahi, 4)],
                                       split_p025=round(np.percentile(ra, 2.5), 4), split_p975=round(np.percentile(ra, 97.5), 4),
                                       baseline_mean=round(ra0.mean(), 4), paired_wins_vs_baseline=wins, bme=hb,
                                       higher=ra.mean() > hb, clear=bool(ra.mean() > hb and alo > hb and wins >= 40)),
                       loso_r2=round(loso, 4), baseline_loso_r2=round(loso0, 4),
                       lit_prior=lit_info if aid == "C3_lit_prior" else None,
                       comparable="pending-human", note="Reconstruction, not BME's population (Arnav 2026-10-02).")
            json.dump(res, open(ROOT / f"experiments/results/bme_char_round1_{aid}.json", "w"), indent=2, default=bool)
            print(f"  {aid:20s} FULL {r2_score(y, pf):.3f} [{flo:.3f},{fhi:.3f}] vs {fb} | A {ra.mean():.3f} CI[{alo:.3f},{ahi:.3f}] "
                  f"split[{np.percentile(ra, 2.5):.3f},{np.percentile(ra, 97.5):.3f}] wins {wins}/50 vs {hb} clear={res['protocol_A']['clear']} | LOSO {loso:.3f}", flush=True)


if __name__ == "__main__":
    main()
