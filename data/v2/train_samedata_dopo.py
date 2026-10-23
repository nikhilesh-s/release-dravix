#!/usr/bin/env python3
"""Same-data rounds samedata_huang_dopo_ep_ml (LOI) and _ul94 (UL-94 V-0), followed literally
(experiments/configs/samedata_huang_dopo_ep_ml*.json + samedata_common.json). Dravix env (.venv312) only.

  splits          write experiments/runs/samedata_huang_dopo_ep_ml_splits.json (before any fit)
  env             append pip freezes + TabPFN checkpoints (V2 classifier fetched once) to experiments/runs/samedata_env.txt
  fit TASK CAND   CAND in c1 c2 (LOI) / c1u c2u (UL94_V0), on the step-0 fold matrices (data/v2/export_dopo_folds.py)
  report          metrics, result JSONs, summary
Built with PriorLabs-TabPFN.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

MODE = sys.argv[1]
CACHE = Path.home() / "Library" / "Caches" / "tabpfn"
REG_CKPT = CACHE / "tabpfn-v2-regressor.ckpt"
CLF_CKPT = CACHE / "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"  # ModelSource.get_classifier_v2().default_filename
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(CACHE)
os.environ["TABPFN_EXCLUDE_DEVICES"] = "cuda,mps"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" if (MODE == "env" and not CLF_CKPT.exists()) else "1"  # 0 only for the one fetch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "5")  # n_jobs=-1 resolves to 5 cores

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

ROOT = Path("~/dravix")
sys.path.insert(0, str(ROOT / "data/v2"))
import eval_protocol as ep  # noqa: E402

RUNS, RES = ROOT / "experiments/runs", ROOT / "experiments/results"
FOLDS = RUNS / "samedata_dopo_folds"
PREDS = RUNS / "samedata_dopo_preds"
SPLITS = RUNS / "samedata_huang_dopo_ep_ml_splits.json"
INC = ROOT / "data/sources/incoming/huang_dopo_ep_ml"
REG_SHA, REG_SIZE = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736", 44390977
ATTR = "Built with PriorLabs-TabPFN"
TASKS = {"LOI": "samedata_huang_dopo_ep_ml", "UL94_V0": "samedata_huang_dopo_ep_ml_ul94"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- splits
def pub_folds(csv, rid):
    import pandas as pd
    p = pd.read_csv(csv, encoding="utf-8-sig")
    fold = dict(zip(p.Record_ID.astype(str), p.outer_fold.astype(int)))
    f = np.array([fold[r] for r in rid])
    return [np.flatnonzero(f == k).tolist() for k in sorted(set(f))]


def splits():
    assert not SPLITS.exists(), "splits already written"
    out = {}
    for task in TASKS:
        m = np.load(FOLDS / f"{task}_meta.npz")
        rid, y, ref, mol = m["record_id"], m["y"], m["ref"], m["mol"]
        Z = np.zeros((len(y), 1))
        gkf = lambda g, r: [te.tolist() for _, te in GroupKFold(5, shuffle=True, random_state=r).split(Z, y, g)]
        S = {"P1_pub": pub_folds(INC / ("LOI_outer_predictions_reference.csv" if task == "LOI"
                                        else "UL94_V0_outer_predictions_reference.csv"), rid)}
        if task == "LOI":
            S.update({f"P1_r{r}": gkf(ref, r) for r in range(10)})  # shared by P1 and P2
            S["P1_pub_mol"] = pub_folds(INC / "LOI_outer_predictions_molecule.csv", rid)
            S.update({f"P1_mol_r{r}": gkf(mol, r) for r in range(10)})
            # info: config says P1_pub == GroupKFold(5) unshuffled by Reference
            unsh = [te.tolist() for _, te in GroupKFold(5).split(Z, y, ref)]
            print("LOI P1_pub == GroupKFold(5) unshuffled by Reference:",
                  sorted(map(sorted, unsh)) == sorted(map(sorted, S["P1_pub"])))
        else:
            S.update({f"P1_r{r}": [te.tolist() for _, te in StratifiedGroupKFold(5, shuffle=True, random_state=r)
                                   .split(Z, y.astype(int), ref)] for r in range(10)})
            S.update({f"P2_r{r}": gkf(ref, r) for r in range(10)})
        for name, sets in S.items():  # every row tested exactly once; groups never straddle
            allte = np.concatenate([np.asarray(t) for t in sets])
            assert len(allte) == len(y) and len(set(allte)) == len(y), name
            g = mol if "mol" in name else ref
            for t in sets:
                assert not set(g[t]) & set(np.delete(g, t)), name
        out[task] = {"record_id": rid.tolist(), "n_rows": int(len(y)),
                     "groups": {"P1_pub/P1/P2": "Reference (DOI URL), 130 papers",
                                "P1_pub_mol/P1_mol": "their get_groups() molecule labels (== Canonical_main_co_pair partition)"},
                     "note": "lists are TEST row indices per fold (rows in record_id order); train = all other rows",
                     "splits": S}
        print(task, {k: [len(t) for t in v] for k, v in list(S.items())[:3]})
    out["_meta"] = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code": "data/v2/train_samedata_dopo.py splits",
                    "sklearn": __import__("sklearn").__version__,
                    "P1_pub": "published per-row outer_fold (LOI/UL94 *_outer_predictions_reference.csv)",
                    "LOI P1_rX": "GroupKFold(5, shuffle=True, random_state=X) by Reference = eval_protocol P2 splits",
                    "UL94 P1_rX": "StratifiedGroupKFold(5, shuffle=True, random_state=X) by Reference",
                    "UL94 P2_rX": "GroupKFold(5, shuffle=True, random_state=X) by Reference"}
    SPLITS.write_text(json.dumps(out))
    print("wrote", SPLITS)


# ---------------------------------------------------------------- env
def freeze(py):
    return subprocess.run([str(Path.home() / ".local/bin/uv"), "pip", "freeze", "--python", py],
                          capture_output=True, text=True).stdout


def env():
    import logging, platform
    import torch
    torch.set_num_threads(1)
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    fetch = os.environ["HF_HUB_OFFLINE"] == "0"
    logs = []

    class H(logging.Handler):
        def emit(self, r):
            logs.append(f"{r.levelname}: {r.getMessage()}")
    logging.getLogger("tabpfn").addHandler(H())
    logging.getLogger("tabpfn").setLevel(logging.INFO)
    rng = np.random.default_rng(0)  # synthetic data only (no dataset labels)
    X = rng.normal(size=(60, 4))
    err = None
    try:
        TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0).fit(X, (X[:, 0] > 0).astype(int))
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    hf_failed = any("HuggingFace download failed" in m for m in logs)
    L = [f"# samedata_huang_dopo_ep_ml(+_ul94) -- data/v2/train_samedata_dopo.py env, {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
         f"python {platform.python_version()} ({sys.executable}); torch.get_num_threads()={torch.get_num_threads()}",
         *[f"{k}={os.environ.get(k)}" for k in ("TABPFN_MODEL_CACHE_DIR", "TABPFN_EXCLUDE_DEVICES", "HF_HUB_DISABLE_TELEMETRY",
                                                "HF_HUB_OFFLINE", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT")],
         f"tabpfn.__version__={tabpfn.__version__}",
         f"V2 regressor {REG_CKPT.name}: size={REG_CKPT.stat().st_size} sha256={sha256(REG_CKPT)} "
         f"(binding {REG_SHA}, {REG_SIZE}) match={sha256(REG_CKPT) == REG_SHA and REG_CKPT.stat().st_size == REG_SIZE}",
         f"V2 classifier fetch this invocation: {fetch}; tabpfn log: {logs}; HF download failed: {hf_failed}; error: {err}"]
    if CLF_CKPT.exists() and not hf_failed and err is None:
        L.append(f"V2 classifier {CLF_CKPT.name} (HF Prior-Labs/TabPFN-v2-clf, tabpfn 9.1.0 downloader): "
                 f"size={CLF_CKPT.stat().st_size} sha256={sha256(CLF_CKPT)}  <- binds every later c2u/c3u run")
    else:
        L.append("V2 classifier NOT available from HF -> c2u/c3u not run, family {c1u}")
    L.append(f"cache ckpt files: {sorted(p.name for p in CACHE.rglob('*.ckpt'))}")
    L.append("# Dravix interpreter freeze (~/.local/bin/uv pip freeze --python .venv312/bin/python):")
    L.append(freeze(str(ROOT / ".venv312/bin/python")).strip())
    L.append("# step-0 authors' env freeze (.venv_dopo, uv venv --python 3.12.7 + their requirements.txt @ e550e4a):")
    L.append(freeze(str(ROOT / ".venv_dopo/bin/python")).strip())
    L.append("# step-0 runtime note: xgboost 3.2.0 / lightgbm 4.6.0 macOS wheels need libomp.dylib; not installed system-wide, "
             "so the step-0 runs set DYLD_FALLBACK_LIBRARY_PATH=.venv_dopo/lib/python3.12/site-packages/sklearn/.dylibs "
             "(the libomp shipped inside the pinned scikit-learn 1.6.1 wheel). Nothing else installed.")
    with open(RUNS / "samedata_env.txt", "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


# ---------------------------------------------------------------- candidates (samedata_common.json, fixed a priori)
def c1_model(cls):
    from sklearn.base import BaseEstimator, clone
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class Avg(BaseEstimator):  # data/v2/train_loi_rescore_r8.py class Avg (np.mean of member outputs)
        def __init__(self, ests):
            self.ests = ests

        def fit(self, X, y):
            self.f_ = [clone(e).fit(X, y) for e in self.ests]
            return self

        def predict(self, X):
            return np.mean([e.predict(X) for e in self.f_], axis=0)

        def predict_proba(self, X):
            return np.mean([e.predict_proba(X) for e in self.f_], axis=0)

    imp = lambda: SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if not cls:
        return Avg([RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                    ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                    make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0))])
    return Avg([RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))])


def fold_keys(task):
    return sorted(p.stem for p in (FOLDS / task).glob("*.npz"))


def _c2_fold(args):
    task, key = args
    import torch
    torch.set_num_threads(1)
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    d = np.load(FOLDS / task / f"{key}.npz")
    try:
        if task == "LOI":
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
            return key, m.fit(d["Xtr"], d["ytr"]).predict(d["Xte"]), None
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
        m.fit(d["Xtr"], d["ytr"].astype(int))
        assert list(m.classes_) == [0, 1]
        return key, m.predict_proba(d["Xte"]), None
    except Exception as e:  # noqa: BLE001  -- config: any TabPFN exception -> c2/c3 BLOCKED, never retried
        return key, None, f"{type(e).__name__}: {e}"


def fit(task, cand):
    PREDS.mkdir(parents=True, exist_ok=True)
    keys, out, t = fold_keys(task), {}, time.time()
    print(f"{task} {cand}: {len(keys)} fold fits; LOKY_MAX_CPU_COUNT={os.environ['LOKY_MAX_CPU_COUNT']} "
          f"HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}", flush=True)
    if cand in ("c1", "c1u"):
        for i, k in enumerate(keys):
            d = np.load(FOLDS / task / f"{k}.npz")
            m = c1_model(cand == "c1u").fit(d["Xtr"], d["ytr"] if cand == "c1" else d["ytr"].astype(int))
            out[k] = m.predict(d["Xte"]) if cand == "c1" else m.predict_proba(d["Xte"])
            if i % 10 == 0:
                print(f"  {i + 1}/{len(keys)} {k} Xtr {d['Xtr'].shape} {time.time() - t:.0f}s", flush=True)
    else:
        assert sha256(REG_CKPT if task == "LOI" else CLF_CKPT) == (REG_SHA if task == "LOI" else
                                                                  "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        cache = PREDS / f"{task}_{cand}"  # one .npy per fold, so an interrupted run resumes without refitting
        cache.mkdir(exist_ok=True)
        prio = lambda k: (0 if k.startswith(("P1_r", "P2_r")) else 1 if k.startswith(("P1_pub__", "P3")) else 2, k)
        todo = [k for k in sorted(keys, key=prio) if not (cache / f"{k}.npy").exists()]
        print(f"  cached {len(keys) - len(todo)}, to fit {len(todo)}", flush=True)
        errs = {}
        with ProcessPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(_c2_fold, (task, k)) for k in todo]
            for i, fu in enumerate(as_completed(futs)):
                k, p, e = fu.result()
                if e:
                    errs[k] = e
                else:
                    np.save(cache / f"{k}.npy", p)
                print(f"  {i + 1}/{len(todo)} {k} {e or ''} {time.time() - t:.0f}s", flush=True)
                if e:  # config: any TabPFN exception blocks c2/c3; stop instead of burning the remaining folds
                    for x in futs:
                        x.cancel()
                    break
        out = {k: np.load(cache / f"{k}.npy") for k in keys if (cache / f"{k}.npy").exists()}
        if errs:
            (PREDS / f"{task}_{cand}_BLOCKED.json").write_text(json.dumps(errs, indent=1))
            print(f"BLOCKED: {len(errs)} TabPFN errors, first: {next(iter(errs.values()))}")
            return
    assert len(out) == len(keys), f"{len(out)}/{len(keys)} folds"
    np.savez_compressed(PREDS / f"{task}_{cand}.npz", **out)
    print(f"{task} {cand} done in {(time.time() - t) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- report
BAR = {"LOI": dict(metric="R2", value=0.709, sd=0.084, mol=0.716, p3=(-0.21, 2.47)),
       "UL94_V0": dict(metric="macro-F1", value=0.817, sd=0.051)}
CANDS = {"LOI": ["c1", "c2", "c3"], "UL94_V0": ["c1u", "c2u", "c3u"]}
DESC = {"r0": {"LOI": "their front end + GradientBoostingRegressor(n_estimators=220, learning_rate=0.03, max_depth=2, "
                      "subsample=0.85, random_state=42), no inner selection (authors' env)",
               "UL94_V0": "their front end + their ExtraTreesClassifier (_classification_model('ExtraTrees', 42)) + their "
                          "threshold rule (inner grouped OOF -> Platt -> optimise_threshold) on the training part (authors' env)"},
        "r0_full": "their evaluate_task_nested(split_strategy='reference', fixed, baseline_inclusive, no BDE, 5x5, "
                   "random_state=42) in the authors' pinned env",
        "c1": "equal average of RF(500, leaf 2, rs 0), ET(500, leaf 2, rs 0), median-impute+indicator/StandardScaler/Ridge(10)",
        "c2": "TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0)",
        "c3": "(c1 + c2) / 2",
        "c1u": "mean predict_proba of RFC(500, leaf 2, rs 0), ETC(500, leaf 2, rs 0), impute/scale/LogisticRegression(C=0.1); argmax",
        "c2u": "TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0); argmax",
        "c3u": "mean of c1u and c2u probabilities; argmax"}


def boot_ci(y, pred, groups, stat, n_boot=2000, seed=0):
    """ep.bootstrap_ci resampling (papers with replacement, 2000 draws, seed 0) with a pluggable statistic."""
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    s = [stat(y[r], pred[r]) for r in (np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))]) for _ in range(n_boot))]
    return np.percentile(s, [2.5, 97.5])


def report():
    S = json.loads(SPLITS.read_text())
    summary = {}
    for task, rnd in TASKS.items():
        reg = task == "LOI"
        m = np.load(FOLDS / f"{task}_meta.npz")
        y, ref, load = m["y"], m["ref"], m["loading"]
        yi = y.astype(int)
        gate = json.loads((FOLDS / f"{task}_gate.json").read_text())
        fold = {k: np.load(FOLDS / task / f"{k}.npz") for k in fold_keys(task)}
        cands = CANDS[task]
        P, blocked = {}, {}
        for c in cands[:2]:
            f = PREDS / f"{task}_{c}.npz"
            if f.exists():
                P[c] = dict(np.load(f))
            else:
                blocked[c] = (PREDS / f"{task}_{c}_BLOCKED.json").read_text() if (PREDS / f"{task}_{c}_BLOCKED.json").exists() \
                    else "not run"
        if cands[0] in P and cands[1] in P:
            P[cands[2]] = {k: (P[cands[0]][k] + P[cands[1]][k]) / 2 for k in P[cands[0]]}
        else:
            blocked[cands[2]] = "needs both members"
        if reg:
            P["r0"] = {k: d["r0"] for k, d in fold.items()}
        else:
            P["r0"] = {k: (d["r0"] >= float(d["r0_thr"])).astype(int) for k, d in fold.items()}
        label = lambda c, p: p if (reg or c == "r0") else np.argmax(p, axis=1)  # argmax of averaged probabilities

        def fold_metrics(c, name):
            rows = []
            for f in range(5):
                k, te = f"{name}__f{f}", np.asarray(S[task]["splits"][name][f])
                p = label(c, P[c][k])
                if reg:
                    mod = load[te] > 0
                    rows.append(dict(R2=r2_score(y[te], p), RMSE=mean_squared_error(y[te], p) ** 0.5,
                                     MAE=mean_absolute_error(y[te], p), mod_R2=r2_score(y[te][mod], p[mod])))
                else:
                    mod = load[te] > 0
                    rows.append(dict(macro_F1=f1_score(yi[te], p, average="macro", zero_division=0),
                                     accuracy=accuracy_score(yi[te], p),
                                     weighted_F1=f1_score(yi[te], p, average="weighted", zero_division=0),
                                     mod_macro_F1=f1_score(yi[te][mod], p[mod], average="macro", zero_division=0)))
            return {key: [r[key] for r in rows] for key in rows[0]}

        def oof(c, name):
            o = np.empty(len(y)) if reg else np.empty(len(y), int)
            for f in range(5):
                o[np.asarray(S[task]["splits"][name][f])] = label(c, P[c][f"{name}__f{f}"])
            return o

        prim = "R2" if reg else "macro_F1"
        protos = {"P1_pub": ["P1_pub"], "P1": [f"P1_r{r}" for r in range(10)]}
        if reg:
            protos.update({"P1_pub_mol": ["P1_pub_mol"], "P1_mol": [f"P1_mol_r{r}" for r in range(10)],
                           "P2": [f"P1_r{r}" for r in range(10)], "P3": ["P3"]})
        else:
            protos["P2"] = [f"P2_r{r}" for r in range(10)]
        R = {}
        for c in ["r0"] + cands:
            if c not in P:
                continue
            for proto, names in protos.items():
                if proto == "P3":
                    d = fold["P3__f0"]
                    e = d["elig"].astype(bool)
                    p = P[c]["P3__f0"][e]
                    R[c, proto] = dict(n=int(e.sum()), R2=r2_score(d["yte"][e], p), MAE=mean_absolute_error(d["yte"][e], p))
                elif proto == "P2":
                    oo = [oof(c, n) for n in names]
                    st = (lambda a, b: r2_score(a, b)) if reg else (lambda a, b: f1_score(a.astype(int), b, average="macro", zero_division=0))
                    v = np.array([st(y, o) for o in oo])
                    lo, hi = ep.bootstrap_ci(y, oo[0], ref) if reg else boot_ci(yi, oo[0], ref, st)
                    sec = {}
                    if reg:
                        sec = dict(RMSE=[mean_squared_error(y, o) ** 0.5 for o in oo], MAE=[mean_absolute_error(y, o) for o in oo])
                    else:
                        sec = dict(accuracy=[accuracy_score(yi, o) for o in oo],
                                   weighted_F1=[f1_score(yi, o, average="weighted", zero_division=0) for o in oo])
                    R[c, proto] = dict(per_repeat=v.tolist(), mean=v.mean(), sd=v.std(), repeat0=v[0], ci_lo=lo, ci_hi=hi,
                                       secondary_per_repeat=sec)
                else:
                    fm = [fold_metrics(c, n) for n in names]
                    agg = {key: [float(np.mean(x[key])) for x in fm] for key in fm[0]}
                    v = np.array(agg[prim])
                    r = dict(per_repeat=v.tolist(), mean=v.mean(), repeat0=v[0], secondary_per_repeat=agg)
                    if len(names) == 1:  # published folds: their aggregation, SD over folds ddof=1
                        r.update(per_fold=fm[0], sd=float(np.std(fm[0][prim], ddof=1)))
                    else:
                        r["sd"] = v.std()
                    R[c, proto] = r
        # selection (within protocol, mean over 10 repeats, ties -> listed first) + paired wins vs r0
        sel = {}
        for proto in [p for p in ("P1", "P1_mol", "P2") if p in protos]:
            avail = [c for c in cands if (c, proto) in R]
            if avail:
                sel[proto] = max(avail, key=lambda c: (R[c, proto]["mean"], -cands.index(c)))
            for c in avail:
                a, b = np.array(R[c, proto]["per_repeat"]), np.array(R["r0", proto]["per_repeat"])
                R[c, proto]["paired_wins_vs_r0"] = f"{int((a > b).sum())}/10"
        summary[task] = dict(R=R, sel=sel, gate=gate, blocked=blocked, protos=list(protos))
        write_results(task, rnd, R, sel, gate, blocked, len(y), len(np.unique(ref)))
    write_summary(summary)


def write_results(task, rnd, R, sel, gate, blocked, n, ng):
    reg = task == "LOI"
    bar = BAR[task]
    common = dict(target="LOI (%)" if reg else "UL-94 V-0 (binary)", round_config=f"experiments/configs/{rnd}.json",
                  dataset="Huang/Zheng/Shi DOPO-EP-ML (data/sources/incoming/huang_dopo_ep_ml/DOPO_EP_new.csv, 599 rows, "
                          "130 Reference groups), unpublished manuscript",
                  features=("authors' LOI input: view compact, K=260, basic descriptors, V7 on, no BDE, neat-EP masking; "
                            "their in-fold front end (nonempty/median impute/dedup/variance 1e-8/SelectKBest f_regression k=260)"
                            if reg else "authors' UL94_V0 input: view morgan_r3, K=all, advanced descriptors, V7 on, no BDE, "
                            "neat-EP masking; their in-fold front end with f_classif"),
                  code="data/v2/train_samedata_dopo.py + data/v2/export_dopo_folds.py (step 0, authors' env)",
                  n_rows=n, n_groups=ng, sota=bar["value"], sota_source="Huang et al. DOPO-EP-ML Table 2 (Reference)",
                  comparable_note="comparability is a human decision; never set here")
    rows = [("r0_full", "P1_pub")] + list(R)
    for c in sorted({c for c, _ in rows}):
        for proto in sorted({p for cc, p in rows if cc == c}):
            slug = f"{rnd}_{c}_{proto}"
            if c == "r0_full":
                r = dict(per_fold=gate["per_fold"], mean=gate["mean"], sd=gate["sd"], gate=gate["gate"],
                         gate_passed=gate["passed"], folds_equal_published=gate["folds_equal_published"],
                         max_abs_pred_diff_vs_published=gate["max_abs_pred_diff_vs_published"],
                         selected_models=gate["selected_models"])
                pf = FOLDS / f"{task}_gate_pubfolds.json"
                if pf.exists():  # FLAG 1 diagnostic: same routine, outer folds forced to the published ones
                    r["diagnostic_published_outer_folds"] = json.loads(pf.read_text())
            else:
                r = {k: v for k, v in R[c, proto].items()}
            d = dict(slug=slug, **common, candidate=c, protocol=proto, model=DESC[c] if c in DESC and isinstance(DESC[c], str)
                     else DESC["r0"][task], hyperparameters="fixed a priori (samedata_common.json); no tuning",
                     metric=("R2" if reg else "macro-F1") if proto != "P3" else "R2 and MAE on the n eligible external rows",
                     status="ok", blocked_reason="", selected=bool(sel.get(proto) == c), **_jsonable(r))
            if c in ("c2", "c3", "c2u", "c3u"):
                d["attribution"] = ATTR
            d["note"] = _note(task, c, proto, sel)
            (RES / f"{slug}.json").write_text(json.dumps(d, indent=1))
    for c, why in blocked.items():
        for proto in ("P1_pub", "P1", "P2"):
            slug = f"{rnd}_{c}_{proto}"
            (RES / f"{slug}.json").write_text(json.dumps(dict(slug=slug, **common, candidate=c, protocol=proto,
                                                              status="blocked", blocked_reason=why[:500]), indent=1))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _note(task, c, proto, sel):
    s = f"{task} {proto} {c}"
    if proto == "P1_pub" or proto == "P1_pub_mol":
        s += "; published folds, information only (never selected on)"
    if sel.get(proto) == c:
        s += "; c* of this protocol (mean over 10 repeats)"
    if c in ("c2", "c3", "c2u", "c3u"):
        s += "; " + ATTR
    return s


def pubfold_note(task):
    f = FOLDS / f"{task}_gate_pubfolds.json"
    if not f.exists():
        return ""
    g = json.loads(f.read_text())
    return (f" Diagnostic (same routine, outer folds forced to the published per-row folds): {g['mean']:.4f} ± {g['sd']:.4f} "
            f"-> {'PASS' if g['passed'] else 'FAIL'}, models {g['selected_models']} (see samedata_flags_dopo.md FLAG 1).")


def f3(x):
    return f"{x:.3f}"


def write_summary(summary):
    L = ["# samedata_huang_dopo_ep_ml (LOI) + _ul94 (UL-94 V-0): results",
         "", f"Written {time.strftime('%Y-%m-%d %H:%M %Z')} by data/v2/train_samedata_dopo.py report. Configs followed literally; "
         "no tuning. Source manuscript unpublished, not peer-reviewed. " + ATTR + ".", ""]
    for task, rnd in TASKS.items():
        T = summary[task]
        R, sel, gate, bar = T["R"], T["sel"], T["gate"], BAR[task]
        reg = task == "LOI"
        L += [f"## {rnd} ({'LOI R2' if reg else 'UL-94 V-0 macro-F1'})", "",
              f"Reproduction gate (r0_full on P1_pub, authors' env): per-fold mean {gate['mean']:.4f} ± {gate['sd']:.4f} "
              f"(gate [{gate['gate'][0]}, {gate['gate'][1]}]) -> {'PASS' if gate['passed'] else 'FAIL (BLOCKED)'}; "
              f"folds equal the published per-row folds: {gate['folds_equal_published']}; selected models {gate['selected_models']}; "
              f"published {bar['value']} ± {bar['sd']}." + (pubfold_note(task)), ""]
        if T["blocked"]:
            why = lambda w: next(iter(json.loads(w).values())) if str(w).startswith("{") else w
            L += ["Blocked: " + "; ".join(f"{c}: {str(why(w))[:200]}" for c, w in T["blocked"].items()), ""]
        L += ["| candidate | " + " | ".join(p for p in T["protos"]) + " |", "|---|" + "---|" * len(T["protos"])]
        for c in ["r0"] + CANDS[task]:
            cells = []
            for p in T["protos"]:
                r = R.get((c, p))
                if r is None:
                    cells.append("—")
                elif p == "P3":
                    cells.append(f"R2 {f3(r['R2'])}, MAE {f3(r['MAE'])} (n={r['n']})")
                elif p in ("P1_pub", "P1_pub_mol"):
                    cells.append(f"{f3(r['mean'])} ± {f3(r['sd'])}")
                else:
                    x = f"{f3(r['mean'])} ± {f3(r['sd'])} (rep0: {f3(r["repeat0"])})"
                    if p == "P2":
                        x += f" CI [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
                    if c != "r0":
                        x += f"; {r['paired_wins_vs_r0']} > r0"
                    if sel.get(p) == c:
                        x = "**" + x + "**"
                    cells.append(x)
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L += ["", "P1_pub / P1_pub_mol: mean ± SD (ddof=1) over the 5 published folds, information only. P1 / P1_mol: mean of "
              "per-fold metric per repeat, mean ± SD (ddof=0) over 10 repeats (rep0 = repeat-0 value). P2: pooled OOF per repeat; "
              "CI = repeat-0 paper bootstrap (2000, seed 0). 'k/10 > r0' = paired repeats in which the candidate's value "
              "is higher than r0's on the same splits. Bold = c* (highest 10-repeat mean, within protocol).", ""]
        mod = "mod_R2" if reg else "mod_macro_F1"
        L.append("Modified-only rows (Loading > 0), same folds, mean of per-fold values (reported, not selected on): " +
                 "; ".join(f"{c} P1_pub {f3(np.mean(R[c, 'P1_pub']['per_fold'][mod]))}, P1 "
                           f"{f3(np.mean(R[c, 'P1']['secondary_per_repeat'][mod]))}"
                           for c in ["r0"] + CANDS[task] if (c, "P1") in R) +
                 (". Their modified-formulation subset: 0.419 ± 0.210." if reg else ". Their modified-only subset: 0.755 ± 0.067; "
                  "molecule-grouped headline 0.779 ± 0.031."))
        if reg:
            L.append("Context (always printed): the overall 0.709 is lifted by the 131 neat-epoxy rows; Dravix's RPUF LOI "
                     "benchmark has no such rows. P1_mol bar (molecule-grouped headline) 0.716 ± 0.036; P3 their -0.21 / 2.47.")
        else:
            L.append("Their tuned threshold (r0) is an advantage on F1 that c1u-c3u (argmax) do not get; the gap is printed, not adjusted.")
        L.append("")
        if gate["passed"]:
            L.append("Allowed sentences (templates from samedata_common.json, filled verbatim):")
            met = "R2" if reg else "macro-F1"
            if "P1" in sel:
                c = sel["P1"]
                L.append(f"- P1: \"On Huang, Zheng and Shi's data and their protocol (10 random splits instead of their one), "
                         f"Dravix's {c} reaches mean {met} {f3(R[c, 'P1']['mean'])} (repeat 0: {f3(R[c, 'P1']['repeat0'])}) vs "
                         f"their published {bar['value']} (their single split); their own model re-run on the same 10 splits "
                         f"gives {f3(R['r0', 'P1']['mean'])}.\" Paired: {c} > r0 in {R[c, 'P1']['paired_wins_vs_r0']} repeats. "
                         f"All three: " + ", ".join(f"{x} {f3(R[x, 'P1']['mean'])}" if (x, 'P1') in R else f"{x} BLOCKED"
                                                    for x in CANDS[task]) + ".")
            if "P2" in sel:
                c = sel["P2"]
                L.append(f"- P2: \"Under Dravix's paper-grouped 5x10 protocol on the same rows: {c} {f3(R[c, 'P2']['mean'])} "
                         f"[{f3(R[c, 'P2']['ci_lo'])}, {f3(R[c, 'P2']['ci_hi'])}], their model re-run {f3(R['r0', 'P2']['mean'])}.\" "
                         f"Paired: {c} > r0 in {R[c, 'P2']['paired_wins_vs_r0']} repeats. All three: " +
                         ", ".join(f"{x} {f3(R[x, 'P2']['mean'])}" if (x, 'P2') in R else f"{x} BLOCKED"
                                   for x in CANDS[task]) + ".")
            trig = [p for p in ("P1",) if p in sel and R[sel[p], p]["mean"] > bar["value"]] + \
                   [p for p in ("P2",) if p in sel and R[sel[p], p]["ci_lo"] > bar["value"]]
            L.append(f"- Exit trigger (c* P1 mean > bar, or P2 ci_lo > bar): {'YES -> FLAG to owner via orchestrator: ' + ', '.join(trig) if trig else 'no'}.")
        else:
            L.append("Reproduction gate failed: no same-protocol sentence (config rule).")
        L.append("")
    L.append("Multiplicity: c* is the best of 3 by mean (winner's curse, order of the between-candidate spread); all three "
             "and r0 are printed. No p-values, no significance language.")
    (RUNS / "samedata_huang_dopo_ep_ml_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if MODE == "fit":
        fit(sys.argv[2], sys.argv[3])
    else:
        {"splits": splits, "env": env, "report": report}[MODE]()
