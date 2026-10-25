"""conformal_round1: validate paper-grouped conformal intervals for the engine and write api/uncertainty.joblib.

Follows experiments/configs/conformal_round1.json exactly (commit it before running).

    OMP_NUM_THREADS=4 python3 api/build_uncertainty.py | tee experiments/runs/conformal_round1.log
    python3 api/build_uncertainty.py --reuse-oof   # re-run the scoring/artifact step from the saved OOF arrays

Writes experiments/runs/conformal_round1.json (report) and api/uncertainty.joblib (serving artifact).
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine/score"))
from api import model, uncertainty as UQ  # noqa: E402
from cfrps import ENGINE_OUTPUTS, WEIGHTS, cfrps  # noqa: E402

sys.path.insert(0, str(ROOT / "data/v2"))
import conformal as C  # noqa: E402

E, T = model.E, model.E.T
SHA = "0b2af8e1817bdda1a9b1980e481b44d75ac62763f032696090aacd5545e05f99"
# same rows and inputs, evidence text reduced to locators (release/build_public_repo.sh, OA_EVIDENCE=locator)
PUBLIC_SHA = "ab4b5f5ba5f6880ef0459083e5e5b3eb7c40543ec9342c87ee2fd656db8f4b0e"
REPEATS, K = 5, 5
LEVELS = {"80": 0.20, "90": 0.10}
METHODS = ("A_abs", "B_spread", "C_knn")
OUT_JSON = ROOT / "experiments/runs/conformal_round1.json"
OOF_CACHE = Path(tempfile.gettempdir()) / "dravix_conformal_round1_oof.joblib"


def say(*a):
    print(*a, flush=True)


def paper_folds(groups, seed):
    """fold id per row; GroupKFold over all snapshot papers so every target shares the same paper split."""
    f = np.empty(len(groups), int)
    for k, (_, te) in enumerate(GroupKFold(K, shuffle=True, random_state=seed).split(groups, groups=groups)):
        f[te] = k
    return f


def oof(df, X, Z, B):
    groups = df["group"].to_numpy()
    mats = df["matrix_canonical"].to_numpy()
    forms = df["material_form"].to_numpy()
    res = {}
    for t, col in T.TARGETS.items():
        m = ((df[col] > 0) & df["heat_flux_kWm2"].notna()).to_numpy()
        res[t] = dict(rows=np.flatnonzero(m), mu=np.zeros((REPEATS, m.sum())), sd=np.zeros((REPEATS, m.sum())),
                      d=np.zeros((REPEATS, m.sum())), ood_cat=np.zeros((REPEATS, m.sum()), bool))
    for r in range(REPEATS):
        f = paper_folds(groups, r)
        for k in range(K):
            fit = f != k
            # unseen matrix in validation = fewer than 3 papers with it among the fitting papers (engine rule)
            ppm = pd.Series(groups[fit]).groupby(mats[fit]).nunique()
            seen_m = set(ppm[ppm >= 3].index) - {"other"}
            seen_f = set(forms[fit])
            for t, col in T.TARGETS.items():
                R = res[t]
                rows = R["rows"]
                tr, te = rows[fit[rows]], rows[~fit[rows]]
                pos = np.flatnonzero(~fit[rows])
                mdl = T.primary().fit(X[tr], np.log1p(df[col].to_numpy(float)[tr]))
                mem = UQ.members(mdl, X[te])
                R["mu"][r, pos], R["sd"][r, pos] = mem.mean(0), mem.std(0)
                R["d"][r, pos] = UQ.nn_dist(Z[te], Z[tr])
                R["ood_cat"][r, pos] = ~np.isin(mats[te], list(seen_m)) | ~np.isin(forms[te], list(seen_f))
            say(f"  repeat {r} fold {k} done ({time.time() - T0:.0f} s)")
    return res


def conformal_step(y, mu, norm, w, groups, method, seed):
    """Paper-grouped split of the OOF rows: q from 4/5 of papers, applied to the other 1/5.
    Returns per-row half-widths {level: h} on the log scale and per-row normalizer median used."""
    h = {L: np.zeros(len(y)) for L in LEVELS}
    f = paper_folds(groups, seed)
    s_all = np.abs(np.log1p(y) - mu)
    for k in range(K):
        cal, te = f != k, f == k
        med = float(np.median(norm[cal])) if method != "A_abs" else 0.0
        sig = UQ.sigma(method, norm, norm, med)
        for L, a in LEVELS.items():
            q = C.heldout_halfwidth((s_all / sig)[cal], w[cal], a)
            h[L][te] = q * sig[te]
    return h


def metrics(y, mu, h, groups, ymin, ymax):
    cov = (np.abs(np.log1p(y) - mu) <= h).astype(float)
    lo, hi = np.clip(np.expm1(mu - h), ymin, ymax), np.clip(np.expm1(mu + h), ymin, ymax)
    return dict(row=cov.mean(), paper=pd.Series(cov).groupby(groups).mean().mean(),
                width_mean=float((hi - lo).mean()), width_median=float(np.median(hi - lo)),
                logwidth_mean=float((2 * h).mean())), cov, hi - lo


def summarize(per_repeat):
    out = {}
    for key in per_repeat[0]:
        v = np.array([p[key] for p in per_repeat])
        out[key] = round(float(v.mean()), 4)
        if key in ("row", "paper"):
            out[key + "_min"], out[key + "_max"] = round(float(v.min()), 4), round(float(v.max()), 4)
    return out


def nearest_corr(c):
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, 1e-6)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / d[:, None] / d[None, :]


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-oof", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    snap = Path(T.SNAP).read_bytes()
    sha = hashlib.sha256(snap).hexdigest()
    assert sha in (SHA, PUBLIC_SHA), "snapshot changed: not the engine training set"
    B = joblib.load(model.ARTIFACT)
    assert B["meta"]["data_sha256"] == sha, "compact artifact was not trained on this snapshot"
    E._B = B
    df = pd.read_csv(T.SNAP, low_memory=False)
    for c in T.NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    groups = df["group"].to_numpy()

    # engine training X (exactly engine/dravix_engine.py train); the query builder must agree with it
    T.TOPN = 100
    X = T.build_X(df, B["cmap"], B["mat_cats"], B["form_cats"])
    X = X.reindex(columns=B["columns"]).to_numpy(float)
    Xq = UQ.features(E._frame(df), B)
    ok = np.isclose(X, Xq, equal_nan=True) | (np.isnan(X) & (np.array(B["columns"]) == "is_control")[None, :])
    assert ok.all(), f"feature builder disagrees with engine training on {(~ok).sum()} cells"
    say(f"snapshot ok: {len(df)} rows / {len(set(groups))} papers; X {X.shape}; query builder == training builder")

    med = np.nanmedian(X, 0)
    med = np.where(np.isnan(med), 0.0, med)
    Xf = np.where(np.isnan(X), med, X)
    std = Xf.std(0)
    keep = std > 0
    space = dict(med=med, std=np.where(keep, std, 1.0), keep=keep)
    Z = UQ.scale(X, space)
    space["ref"] = Z

    if a.reuse_oof:
        res = joblib.load(OOF_CACHE)
        say(f"reused OOF from {OOF_CACHE}")
    else:
        res = oof(df, X, Z, B)
        joblib.dump(res, OOF_CACHE)

    nadd = np.array([UQ.n_additives(c) for c in df["components"]])
    max_add = int(np.percentile(nadd, 99))
    flux_bad = ~df["heat_flux_kWm2"].between(*UQ.FLUX_RANGE).to_numpy()

    report = dict(round="conformal_round1", config="experiments/configs/conformal_round1.json", snapshot_sha256=SHA,
                  repeats=REPEATS, folds=K, targets={}, max_additives=max_add)
    art = dict(version="conformal_round1", targets=list(ENGINE_OUTPUTS), method={}, q80={}, q90={}, med={},
               ood_dist={}, mask={}, max_additives=max_add, coverage={})
    h90_row = {}  # per target, repeat 0: per-row 90% log half-width from the conformal step (for the CFRS check)
    for t, col in T.TARGETS.items():
        R = res[t]
        rows = R["rows"]
        y, g = df[col].to_numpy(float)[rows], groups[rows]
        w = C.group_weights(g)
        bm = B["models"][t]
        thr = float(np.percentile(R["d"][0], 95))
        cand = {}
        for meth in METHODS:
            per = {L: [] for L in LEVELS}
            split = []
            for r in range(REPEATS):
                norm = {"A_abs": R["sd"][r], "B_spread": R["sd"][r], "C_knn": R["d"][r]}[meth]
                h = conformal_step(y, R["mu"][r], norm, w, g, meth, 1000 + r)
                for L in LEVELS:
                    mt, cov, wid = metrics(y, R["mu"][r], h[L], g, bm["ymin"], bm["ymax"])
                    per[L].append(mt)
                    if L == "90":
                        flag = R["ood_cat"][r] | (nadd[rows] > max_add) | flux_bad[rows] | (R["d"][r] > thr)
                        split.append(dict(ood_rate=flag.mean(), cov_flagged=cov[flag].mean() if flag.any() else np.nan,
                                          cov_unflagged=cov[~flag].mean(), width_median_flagged=float(np.median(wid[flag])) if flag.any() else np.nan,
                                          width_median_unflagged=float(np.median(wid[~flag]))))
                if r == 0:
                    h90_row.setdefault(t, {})[meth] = h["90"]
            cand[meth] = {L: summarize(per[L]) for L in LEVELS}
            cand[meth]["ood_split_90"] = {k: round(float(np.nanmean([s[k] for s in split])), 4) for k in split[0]}
        elig = [m for m in METHODS if all(cand[m][L]["row"] >= 1 - LEVELS[L] - 0.02 and cand[m][L]["paper"] >= 1 - LEVELS[L] - 0.05
                                          for L in LEVELS)]
        pick = min(elig, key=lambda m: cand[m]["90"]["logwidth_mean"]) if elig else "A_abs"
        say(f"\n{t}: {len(rows)} rows / {len(set(g))} papers; eligible {elig}; selected {pick}")
        for m in METHODS:
            c8, c9 = cand[m]["80"], cand[m]["90"]
            say(f"  {m:9s} 80%: cov {c8['row']:.3f} (paper {c8['paper']:.3f}) width mean {c8['width_mean']:.1f} median {c8['width_median']:.1f}"
                f" | 90%: cov {c9['row']:.3f} [{c9['row_min']:.3f}, {c9['row_max']:.3f}] (paper {c9['paper']:.3f})"
                f" width mean {c9['width_mean']:.1f} median {c9['width_median']:.1f} log {c9['logwidth_mean']:.3f}"
                f" | OOD {cand[m]['ood_split_90']}")

        # deployed quantiles: pooled OOF scores of all repeats, each paper weight 1 (weights / REPEATS)
        normp = {"A_abs": R["sd"], "B_spread": R["sd"], "C_knn": R["d"]}[pick].ravel()
        medp = float(np.median(normp)) if pick != "A_abs" else 0.0
        sp = np.abs(np.log1p(np.tile(y, REPEATS)) - R["mu"].ravel()) / UQ.sigma(pick, normp, normp, medp)
        wp = np.tile(w, REPEATS) / REPEATS
        art["method"][t], art["med"][t], art["ood_dist"][t] = pick, medp, thr
        art["q80"][t], art["q90"][t] = C.heldout_halfwidth(sp, wp, 0.20), C.heldout_halfwidth(sp, wp, 0.10)
        mask = np.zeros(len(df), bool)
        mask[rows] = True
        art["mask"][t] = mask
        art["coverage"][t] = {L: cand[pick][L]["row"] for L in LEVELS}
        report["targets"][t] = dict(rows=int(len(rows)), papers=int(len(set(g))), candidates=cand, eligible=elig,
                                    selected=pick, deployed=dict(q80=art["q80"][t], q90=art["q90"][t], med=medp,
                                                                 ood_dist_p95=thr))

    # correlation of signed normalized OOF residuals between targets (repeat 0, pairwise-complete)
    n = len(df)
    Rn = np.full((n, 4), np.nan)
    for j, t in enumerate(ENGINE_OUTPUTS):
        R, col = res[t], T.TARGETS[t]
        rows = R["rows"]
        norm = {"A_abs": R["sd"][0], "B_spread": R["sd"][0], "C_knn": R["d"][0]}[art["method"][t]]
        Rn[rows, j] = (np.log1p(df[col].to_numpy(float)[rows]) - R["mu"][0]) / UQ.sigma(art["method"][t], norm, norm, art["med"][t])
    corr = nearest_corr(pd.DataFrame(Rn).corr().to_numpy())
    art["corr"] = corr
    report["residual_corr"] = {"targets": list(ENGINE_OUTPUTS), "matrix": np.round(corr, 3).tolist()}
    say(f"\nresidual correlation {list(ENGINE_OUTPUTS)}:\n{np.round(corr, 3)}")

    # CFRS check (report-only): rows with all four targets measured, repeat 0 OOF
    pos = {t: dict(zip(res[t]["rows"], range(len(res[t]["rows"])))) for t in ENGINE_OUTPUTS}
    both = sorted(set.intersection(*(set(res[t]["rows"]) for t in ENGINE_OUTPUTS)))
    L = np.linalg.cholesky(corr)
    Zd = np.random.default_rng(0).standard_normal((UQ.N_DRAWS, 4)) @ L.T
    ymin = np.array([B["models"][t]["ymin"] for t in ENGINE_OUTPUTS])
    ymax = np.array([B["models"][t]["ymax"] for t in ENGINE_OUTPUTS])
    wts = {k: WEIGHTS[k] for k in ENGINE_OUTPUTS}
    wts = {k: v / sum(wts.values()) for k, v in wts.items()}
    cov80, cov90, wid90 = [], [], []
    for i in both:
        meas = cfrps({t: float(df.at[i, T.TARGETS[t]]) for t in ENGINE_OUTPUTS})["score"]
        mu = np.array([res[t]["mu"][0, pos[t][i]] for t in ENGINE_OUTPUTS])
        h9 = np.array([h90_row[t][art["method"][t]][pos[t][i]] for t in ENGINE_OUTPUTS])
        point = cfrps(dict(zip(ENGINE_OUTPUTS, np.clip(np.expm1(mu), ymin, ymax))))["score"]
        iv = UQ.cfrs_interval(point, UQ.cfrs_draws(mu, h9, ymin, ymax, Zd, wts))
        cov80.append(iv["interval_80"][0] <= meas <= iv["interval_80"][1])
        cov90.append(iv["interval_90"][0] <= meas <= iv["interval_90"][1])
        wid90.append(iv["interval_90"][1] - iv["interval_90"][0])
    report["cfrs_check"] = dict(rows=len(both), papers=int(len(set(groups[both]))), coverage_80=round(float(np.mean(cov80)), 4),
                                coverage_90=round(float(np.mean(cov90)), 4), width90_mean=round(float(np.mean(wid90)), 2),
                                width90_median=round(float(np.median(wid90)), 2))
    say(f"\nCFRS check ({len(both)} rows with all 4 targets): {report['cfrs_check']}")

    art["space"] = dict(med=space["med"], std=space["std"], keep=keep, ref=space["ref"])
    joblib.dump(art, UQ.ARTIFACT, compress=("xz", 6))
    report["artifact"] = dict(path="api/uncertainty.joblib", mb=round(UQ.ARTIFACT.stat().st_size / 1e6, 2))
    report["seconds"] = round(time.time() - T0, 1)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float) + "\n")
    say(f"\nwrote {UQ.ARTIFACT} ({report['artifact']['mb']} MB) and {OUT_JSON} in {report['seconds']} s")


if __name__ == "__main__":
    main()
