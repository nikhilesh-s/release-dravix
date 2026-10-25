#!/usr/bin/env python3
"""BME full-dataset fit on the Pomazi reconstruction (config: experiments/configs/bme_fullfit_round1.json).
Fit on all rows, score the same rows (BME Table 8 protocol). Plus protocol-A recheck with a CI of the mean."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import eval_protocol as ep
import train_pomazi_round1 as r1
import train_pomazi_round1b as r1b

CFG = json.load(open(ROOT / "experiments/configs/bme_fullfit_round1.json"))
SEL = {"phrr_kWm2": "rf", "tti_s": "rf", "thr_MJm2": "gbr_theirs", "residue_pct": "hgb"}


def protocol_a(m, X, y):
    ra = np.array([r2_score(y[te], clone(m).fit(X[tr], y[tr]).predict(X[te]))
                   for tr, te in (train_test_split(np.arange(len(y)), test_size=0.2, random_state=s) for s in range(50))])
    boot = np.random.default_rng(0).choice(ra, (2000, len(ra))).mean(axis=1)
    return ra, np.percentile(boot, [2.5, 97.5])


def main():
    out = {}
    for tcol, cid in SEL.items():
        X, y, grp, system, hf = r1b.build(tcol)
        fits = {}
        for c in r1b.CIDS:
            p = clone(r1.make(c, tcol)).fit(X, y).predict(X)
            lo, hi = ep.bootstrap_ci(y, p, grp)
            fits[c] = dict(r2=round(r2_score(y, p), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), mae=round(mean_absolute_error(y, p), 3))
        ra, (alo, ahi) = protocol_a(r1.make(cid, tcol), X, y)
        fb, hb = CFG["bars_full_fit"][tcol], CFG["bars_held_out"][tcol]
        f = fits[cid]
        res = dict(slug=f"bme_fullfit_{tcol}", target=r1.TARGETS[tcol][1], round_config="experiments/configs/bme_fullfit_round1.json",
                   code="data/v2/train_bme_fullfit_round1.py", dataset=CFG["dataset"], n_rows=len(y), n_groups=len(np.unique(grp)),
                   model=cid, hyperparameters=r1.hp(cid, tcol),
                   full_fit=dict(protocol="training fit (same protocol as BME: full-dataset fit)", **f, bme=fb,
                                 higher=f["r2"] > fb, clear=f["ci_lo"] > fb),
                   full_fit_other_candidates={c: v for c, v in fits.items() if c != cid},
                   protocol_A=dict(protocol="random 80/20 x 50, seeds 0-49", mean=round(ra.mean(), 4), mean_ci=[round(alo, 4), round(ahi, 4)],
                                   split_p025=round(np.percentile(ra, 2.5), 4), split_p975=round(np.percentile(ra, 97.5), 4),
                                   share_splits_above_bme=round((ra > hb).mean(), 2), bme=hb,
                                   higher=ra.mean() > hb, clear=np.percentile(ra, 2.5) > hb),
                   comparable="pending-human", note="Reconstruction, not BME's population (Arnav 2026-10-02).")
        out[tcol] = res
        json.dump(res, open(ROOT / f"experiments/results/bme_fullfit_{tcol}.json", "w"), indent=2, default=bool)
        print(f"{tcol:12s} {cid:10s} n={len(y)} FULL {f['r2']:.3f} [{f['ci_lo']:.3f},{f['ci_hi']:.3f}] MAE {f['mae']:.2f} vs {fb} | "
              f"A {ra.mean():.3f} CI[{alo:.3f},{ahi:.3f}] split[{np.percentile(ra,2.5):.3f},{np.percentile(ra,97.5):.3f}] "
              f">bar {(ra > hb).mean():.2f} vs {hb}", flush=True)
        print("   others:", {c: v["r2"] for c, v in fits.items()}, flush=True)


if __name__ == "__main__":
    main()
