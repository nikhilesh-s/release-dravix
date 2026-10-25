#!/usr/bin/env python3
"""Same-data round samedata_huang_dopo_ep_ml_cone (DOPO-EP-ML pHRR + THR), followed literally
(experiments/configs/samedata_huang_dopo_ep_ml_cone.json + samedata_common.json). Dravix env (.venv312) only.
Step-0 matrices and their re-runs come from data/v2/export_dopo_cone.py (authors' env).

  splits          write experiments/runs/samedata_huang_dopo_ep_ml_cone_splits.json (before any fit)
  fit TASK CAND   CAND in c1 (also writes c4, b0) | c2
  report          metrics, results JSON, summary
Built with PriorLabs-TabPFN.
"""
import hashlib, json, os, sys, time
from pathlib import Path

os.environ.update(TABPFN_MODEL_CACHE_DIR=str(Path.home() / "Library/Caches/tabpfn"), TABPFN_EXCLUDE_DEVICES="cuda,mps",
                  HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "5")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

WT = Path("~/dravix-dopo-cone")
sys.path.insert(0, str(WT / "data/v2"))
import eval_protocol as ep  # noqa: E402

RUNS = WT / "experiments/runs"
FOLDS = RUNS / "samedata_dopo_cone_folds"
PREDS = RUNS / "samedata_dopo_cone_preds"
SPLITS = RUNS / "samedata_huang_dopo_ep_ml_cone_splits.json"
RESULT = WT / "experiments/results/samedata/samedata_huang_dopo_ep_ml_cone.json"
CLONE = WT / "data/v2/raw/dopo_ep_ml/code"
PUB = CLONE / "results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5/{t}/molecule/formulation/without_BDE/{t}_outer_predictions.csv"
REG_CKPT = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / "tabpfn-v2-regressor.ckpt"
REG_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
TASKS = ("PHRR", "THR")
BAR = {"PHRR": dict(value=0.604, sd=0.042, ref=0.576, ref_sd=0.052, mae=161.57, unit="kW/m2"),
       "THR": dict(value=0.483, sd=0.133, ref=0.362, ref_sd=0.248, mae=12.43, unit="MJ/m2")}
CANDS = ["c1", "c2", "c3", "c4"]
P1 = [f"P1_mol_r{r}" for r in range(10)]
P2 = [f"P2_r{r}" for r in range(10)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meta(task):
    return np.load(FOLDS / f"{task}_meta.npz")


# ---------------------------------------------------------------- splits
def splits():
    assert not SPLITS.exists(), "splits already written"
    out = {}
    for task in TASKS:
        m = meta(task)
        rid, y, ref, mol = m["record_id"], m["y"], m["ref"], m["mol"]
        Z = np.zeros((len(y), 1))
        gkf = lambda g, r: [te.tolist() for _, te in GroupKFold(5, shuffle=True, random_state=r).split(Z, y, g)]
        pub = pd.read_csv(str(PUB).format(t=task), encoding="utf-8-sig")
        fold = dict(zip(pub.Record_ID.astype(str), pub.outer_fold.astype(int)))
        f = np.array([fold[r] for r in rid])
        S = {"P1_pub_mol": [np.flatnonzero(f == k).tolist() for k in sorted(set(f))]}
        S.update({n: gkf(mol, r) for r, n in enumerate(P1)})
        S.update({n: gkf(ref, r) for r, n in enumerate(P2)})
        unsh = [te.tolist() for _, te in GroupKFold(5).split(Z, y, mol)]
        print(task, "P1_pub_mol == GroupKFold(5) unshuffled by molecule:",
              sorted(map(sorted, unsh)) == sorted(map(sorted, S["P1_pub_mol"])))
        for name, sets in S.items():  # every row tested once; no group straddles train/test
            allte = np.concatenate([np.asarray(t) for t in sets])
            assert len(allte) == len(y) and len(set(allte)) == len(y), name
            g = ref if name.startswith("P2") else mol
            for t in sets:
                assert not set(g[t]) & set(np.delete(g, t)), name
        out[task] = {"record_id": rid.tolist(), "n_rows": int(len(y)), "splits": S,
                     "note": "TEST row indices per fold (record_id order); P1_*: their molecule groups; P2_*: Reference"}
    out["_meta"] = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "sklearn": __import__("sklearn").__version__,
                    "P1_pub_mol": "published per-row outer_fold", "P1_mol_rX / P2_rX": "GroupKFold(5, shuffle=True, random_state=X)"}
    SPLITS.write_text(json.dumps(out))
    print("wrote", SPLITS)


# ---------------------------------------------------------------- candidates
def c1_model():
    """samedata_common.json c1 (= data/v2/train_samedata_dopo.py c1_model, regression branch)."""
    from sklearn.base import BaseEstimator, clone
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class Avg(BaseEstimator):
        def __init__(self, ests):
            self.ests = ests

        def fit(self, X, y):
            self.f_ = [clone(e).fit(X, y) for e in self.ests]
            return self

        def predict(self, X):
            return np.mean([e.predict(X) for e in self.f_], axis=0)

    return Avg([RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=10.0))])


def keys(task):
    return sorted(p.stem for p in (FOLDS / task).glob("*__f*.npz"))


def fit_c1(task):
    m = meta(task)
    ok = (m["loading"] > 0) & (m["ctrl"] > 0)  # rows with the same paper's neat-EP value
    out = {c: {} for c in ("c1", "c4", "b0")}
    t = time.time()
    for i, k in enumerate(keys(task)):
        d = np.load(FOLDS / task / f"{k}.npz")
        tr, te, ytr = d["tr"], d["te"], d["ytr"]
        p1 = c1_model().fit(d["Xtr"], ytr).predict(d["Xte"])
        mtr, mte = ok[tr], ok[te]
        ctr, cte = m["ctrl"][tr][mtr], m["ctrl"][te][mte]
        pr = c1_model().fit(d["Xtr"][mtr], np.log(ytr[mtr] / ctr)).predict(d["Xte"][mte])
        p4, b0 = p1.copy(), p1.copy()
        p4[mte] = np.clip(cte * np.exp(pr), ytr.min(), ytr.max())
        b0[mte] = cte * np.median(ytr[mtr] / ctr)
        out["c1"][k], out["c4"][k], out["b0"][k] = p1, p4, b0
        if i % 15 == 0:
            print(f"  {task} {i + 1} {k} Xtr {d['Xtr'].shape} ratio rows {mtr.sum()} {time.time() - t:.0f}s", flush=True)
    for c, v in out.items():
        np.savez_compressed(PREDS / f"{task}_{c}.npz", **v)
    print(f"{task} c1/c4/b0 done in {(time.time() - t) / 60:.1f} min")


def _c2_fold(args):
    task, k = args
    import torch
    torch.set_num_threads(1)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    d = np.load(FOLDS / task / f"{k}.npz")
    try:
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
        return k, m.fit(d["Xtr"], d["ytr"]).predict(d["Xte"]), None
    except Exception as e:  # noqa: BLE001  -- config: any TabPFN exception blocks c2/c3, never retried
        return k, None, f"{type(e).__name__}: {e}"


def fit_c2(task):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    assert sha256(REG_CKPT) == REG_SHA
    cache = PREDS / f"{task}_c2"
    cache.mkdir(exist_ok=True)
    ks, t = keys(task), time.time()
    todo = [k for k in ks if not (cache / f"{k}.npy").exists()]
    errs = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_c2_fold, (task, k)) for k in todo]
        for i, fu in enumerate(as_completed(futs)):
            k, p, e = fu.result()
            if e:
                errs[k] = e
                for x in futs:
                    x.cancel()
                break
            np.save(cache / f"{k}.npy", p)
            if i % 10 == 0:
                print(f"  {task} c2 {i + 1}/{len(todo)} {time.time() - t:.0f}s", flush=True)
    if errs:
        (PREDS / f"{task}_c2_BLOCKED.json").write_text(json.dumps(errs, indent=1))
        print("BLOCKED", errs)
        return
    np.savez_compressed(PREDS / f"{task}_c2.npz", **{k: np.load(cache / f"{k}.npy") for k in ks})
    print(f"{task} c2 done in {(time.time() - t) / 60:.1f} min")


# ---------------------------------------------------------------- report
def report():
    S = json.loads(SPLITS.read_text())
    res, lines = {}, []
    for task in TASKS:
        m = meta(task)
        y, ref, mod = m["y"], m["ref"], m["loading"] > 0
        Sp = S[task]["splits"]
        P = {c: dict(np.load(PREDS / f"{task}_{c}.npz")) for c in ("c1", "c4", "b0")}
        blocked = (PREDS / f"{task}_c2_BLOCKED.json").exists()
        if not blocked:
            P["c2"] = dict(np.load(PREDS / f"{task}_c2.npz"))
            P["c3"] = {k: (P["c1"][k] + P["c2"][k]) / 2 for k in P["c1"]}
        P["r0"] = {k: np.load(FOLDS / task / f"{k}.npz")["r0"] for k in keys(task)}
        gate = json.loads((FOLDS / f"{task}_gate.json").read_text())
        full = {n: json.loads((FOLDS / task / "r0_full" / f"{n}.json").read_text()) for n in P1 + P2
                if (FOLDS / task / "r0_full" / f"{n}.json").exists()}

        def oof(c, name):
            if c == "r0_full":
                return np.asarray(full[name]["oof"])
            o = np.empty(len(y))
            for f, te in enumerate(Sp[name]):
                o[np.asarray(te)] = P[c][f"{name}__f{f}"]
            return o

        def perfold(o, name):
            r = [dict(R2=r2_score(y[te], o[te]), MAE=mean_absolute_error(y[te], o[te]),
                      RMSE=mean_squared_error(y[te], o[te]) ** 0.5, mod_R2=r2_score(y[te][mod[te]], o[te][mod[te]]))
                 for te in map(np.asarray, Sp[name])]
            return {k: [x[k] for x in r] for k in r[0]}

        R = {}
        pubd = FOLDS / task / "r0_full" / "P1_pub_mol.json"  # diagnostic: their routine, outer folds = published ones
        if pubd.exists():
            pf = perfold(np.asarray(json.loads(pubd.read_text())["oof"]), "P1_pub_mol")
            R["r0_full_pubfolds", "P1_pub_mol"] = dict(mean=float(np.mean(pf["R2"])), sd=float(np.std(pf["R2"], ddof=1)),
                                                       per_fold=pf["R2"], MAE=float(np.mean(pf["MAE"])),
                                                       mod_R2=float(np.mean(pf["mod_R2"])),
                                                       selected_models=json.loads(pubd.read_text())["selected_models"])
        for c in ["r0_full", "r0", "c1", "c2", "c3", "c4", "b0"]:
            if c not in P and c != "r0_full":
                continue
            # P1_pub_mol: their exact folds, their aggregation
            if c == "r0_full":
                pf = dict(R2=gate["per_fold"], MAE=gate["MAE_per_fold"], mod_R2=gate["modified_R2_per_fold"])
            else:
                pf = perfold(oof(c, "P1_pub_mol"), "P1_pub_mol")
            R[c, "P1_pub_mol"] = dict(mean=float(np.mean(pf["R2"])), sd=float(np.std(pf["R2"], ddof=1)),
                                      per_fold=pf["R2"], MAE=float(np.mean(pf["MAE"])), mod_R2=float(np.mean(pf["mod_R2"])))
            if c == "r0_full" and len(full) < 20:
                continue
            fm = [perfold(oof(c, n), n) for n in P1]
            v = np.array([np.mean(x["R2"]) for x in fm])
            R[c, "P1_mol"] = dict(per_repeat=v.tolist(), mean=v.mean(), sd=v.std(), repeat0=v[0],
                                  MAE=float(np.mean([np.mean(x["MAE"]) for x in fm])),
                                  mod_R2=float(np.mean([np.mean(x["mod_R2"]) for x in fm])),
                                  per_fold_sd_repeat0=float(np.std(fm[0]["R2"], ddof=1)))
            oo = [oof(c, n) for n in P2]
            v = np.array([r2_score(y, o) for o in oo])
            lo, hi = ep.bootstrap_ci(y, oo[0], ref)
            R[c, "P2"] = dict(per_repeat=v.tolist(), mean=v.mean(), sd=v.std(), repeat0=v[0], ci_lo=float(lo),
                              ci_hi=float(hi), MAE=float(np.mean([mean_absolute_error(y, o) for o in oo])),
                              mod_R2=float(np.mean([r2_score(y[mod], o[mod]) for o in oo])))
        sel = {}
        for proto in ("P1_mol", "P2"):
            avail = [c for c in CANDS if (c, proto) in R]
            sel[proto] = max(avail, key=lambda c: (R[c, proto]["mean"], -CANDS.index(c)))
            for c in avail + ["r0", "b0"]:
                for ctl in ("r0_full", "r0"):
                    if (ctl, proto) in R and c != ctl:
                        a, b = np.array(R[c, proto]["per_repeat"]), np.array(R[ctl, proto]["per_repeat"])
                        R[c, proto][f"paired_wins_vs_{ctl}"] = f"{int((a > b).sum())}/10"
        res[task] = dict(n_rows=int(len(y)), n_modified=int(mod.sum()), n_refs=int(len(set(ref))),
                         n_mol_groups=int(len(set(m["mol"]))), bar=BAR[task], gate=gate, c2_blocked=blocked,
                         ctrl_features_kept=json.loads((FOLDS / f"{task}_ctrl_features_kept.json").read_text())["P1_pub_mol__f0"],
                         selected=sel, results={f"{c}|{p}": v for (c, p), v in R.items()},
                         r0_full_selected_models={n: full[n]["selected_models"] for n in full})
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(dict(
        slug="samedata_huang_dopo_ep_ml_cone", round_config="experiments/configs/samedata_huang_dopo_ep_ml_cone.json",
        dataset="Huang/Zheng/Shi DOPO-EP-ML v1.0.2 (data/v2/raw/dopo_ep_ml/code/data/DOPO_EP_new.csv), unpublished manuscript",
        code="data/v2/export_dopo_cone.py (authors' env) + data/v2/train_samedata_dopo_cone.py",
        comparable_note="comparability is a human decision; never set here", attribution="Built with PriorLabs-TabPFN",
        metric_notes="P1_pub_mol / P1_mol: mean of 5 per-fold R2 (their aggregation); P1_pub_mol sd over folds (ddof=1), "
                     "P1_mol sd over 10 repeats; P2 pooled OOF R2, CI = repeat-0 paper bootstrap (2000, seed 0); MAE in target units",
        tasks=res), indent=1, default=float))
    print("wrote", RESULT)
    for task, r in res.items():
        print(f"\n== {task} gate {r['gate']['mean']:.4f} passed {r['gate']['passed']} folds_equal {r['gate']['folds_equal_published']}"
              f" sel {r['selected']} ctrl kept {r['ctrl_features_kept']}")
        for k, v in r["results"].items():
            print(f"{k:22s} mean {v['mean']:.4f} sd {v['sd']:.4f} MAE {v['MAE']:.2f} modR2 {v['mod_R2']:.3f} "
                  + " ".join(f"{a}={b}" for a, b in v.items() if a.startswith(("paired", "ci_"))))


if __name__ == "__main__":
    PREDS.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode == "splits":
        splits()
    elif mode == "fit":
        {"c1": fit_c1, "c2": fit_c2}[sys.argv[3]](sys.argv[2])
    elif mode == "report":
        report()
