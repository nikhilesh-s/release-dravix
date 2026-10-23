#!/usr/bin/env python3
"""Same-data comparisons on published open LOI datasets, followed literally from
experiments/configs/samedata_common.json + experiments/configs/samedata_<dataset>.json.

  ~/dravix/.venv312/bin/python data/v2/train_samedata.py --dataset chen_cej2023|ma_jmi2025

Order: data sha256 + label-free asserts -> experiments/runs/samedata_<ds>_splits.json (before any fit) ->
r0 reproduction gate -> P1 (r0, c1, c2, c3; 10 seeds) -> P2 (Chen only, eval_protocol paper-grouped 5x10).
Writes experiments/results/samedata_<ds>_<cand>_<protocol>.json, experiments/runs/samedata_<ds>_preds.npz and
experiments/runs/samedata_<ds>_summary.md; FLAGs are appended to experiments/runs/samedata_flags.md.
Built with PriorLabs-TabPFN.
"""
import os
from pathlib import Path

CACHE = Path.home() / "Library" / "Caches" / "tabpfn"
# common.c2.weights_env_licence (set before tabpfn/torch import); BLAS/loky limits are compute-only (5-core budget).
os.environ.update(TABPFN_MODEL_CACHE_DIR=str(CACHE), TABPFN_EXCLUDE_DEVICES="cuda,mps", HF_HUB_DISABLE_TELEMETRY="1",
                  HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  LOKY_MAX_CPU_COUNT="5")

import argparse, datetime as dt, hashlib, json, sys, time, warnings  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest,  # noqa: E402
                              RandomForestRegressor)
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict, train_test_split  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
import eval_protocol as ep  # noqa: E402

warnings.filterwarnings("ignore")  # numpy-Accelerate matmul RuntimeWarnings flood the log; TabPFN warnings are recorded separately
T0 = time.time()
N_JOBS = 5
CKPT = CACHE / "tabpfn-v2-regressor.ckpt"
CKPT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
ATTR = "Built with PriorLabs-TabPFN"
INC = ROOT / "data/sources/incoming"
CANDS = ("r0", "c1", "c2", "c3")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---- candidates (common.candidates_regression; c1 members copied from data/v2/train_loi_rescore_r8.py) ----------------
class Avg(BaseEstimator, RegressorMixin):
    def __init__(self, ests):
        self.ests = ests

    def fit(self, X, y):
        self.f_ = [clone(e).fit(X, y) for e in self.ests]
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.f_], axis=0)


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def et():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10.0))


def c1():
    return Avg([rf(), et(), ridge()])


def c2():
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


def r0(ds):
    if ds == "chen_cej2023":  # samedata_chen_cej2023.json "r0 (reproduction control)"
        return make_pipeline(StandardScaler(), RFE(Ridge(alpha=1, random_state=1, max_iter=100000), n_features_to_select=100),
                             PolynomialFeatures(degree=2), GradientBoostingRegressor(random_state=1))
    return make_pipeline(StandardScaler(), Ridge(alpha=0.1))  # samedata_ma_jmi2025.json


def job(cand, ds, X, y, tr, te):
    """One fit on X[tr] -> prediction on X[te]. Runs in a loky worker for r0/c2 (1 thread each)."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    if cand != "c2":
        return dict(pred=ep._fit_predict(r0(ds) if cand == "r0" else c1(), X[tr], y[tr], X[te], False))
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = c2().fit(X[tr], y[tr])
            pred = m.predict(X[te])
        msgs = sorted({str(x.message).splitlines()[0] for x in w if issubclass(x.category, UserWarning)})
        attrs = dict(n_estimators_=int(m.n_estimators_), devices_=[str(d) for d in m.devices_],
                     use_autocast_=bool(m.use_autocast_),
                     forced_inference_dtype_=None if m.forced_inference_dtype_ is None else str(m.forced_inference_dtype_))
        return dict(pred=pred, warnings=msgs, attrs=attrs)
    except Exception as e:  # common.c2.on_error: c2 and c3 BLOCKED, never retried
        return dict(pred=None, error=f"{type(e).__name__}: {e}")


def metrics(yt, yp):
    return dict(r2=float(r2_score(yt, yp)), rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                mae=float(mean_absolute_error(yt, yp)))


# ---- data (config 'rows' / 'features' / 'groups_for_P2', label-free asserts before any fit) -------------------------
def load(ds, cfg):
    if ds == "chen_cej2023":
        f5, f4 = ROOT / cfg["rows"]["file"], ROOT / cfg["groups_for_P2"]["file"]
        assert sha256(f5) == cfg["rows"]["sha256"] and sha256(f4) == cfg["groups_for_P2"]["sha256"], "sha256 mismatch"
        d = pd.read_csv(f5)
        assert d.shape == (1053, 307) and "LOI" in d.columns
        feats = [c for c in d.columns if c != "LOI"]
        assert len(feats) == 306 and feats[-1] == "addition amount (wt%)"
        X, y = d[feats].to_numpy(np.float64), d["LOI"].to_numpy(np.float64)
        m = pd.read_excel(f4)
        assert len(m) == 1053
        ok = (np.array_equal(m["LOI"].to_numpy(np.float64), y)
              and np.array_equal(m["Addition amount (wt%)"].to_numpy(np.float64), d["addition amount (wt%)"].to_numpy(np.float64)))
        doi = m["DOI"].ffill()
        g = doi.astype(str).to_numpy()
        n_g = int(doi.nunique())
        if not (ok and doi.notna().all() and n_g == 254):
            raise SystemExit(f"BLOCKED: mmc4/mmc5 alignment check failed (equal={ok}, distinct DOIs={n_g})")
        log(f"chen: mmc4 LOI + loading equal mmc5 on all 1053 rows; {n_g} distinct forward-filled DOIs; "
            f"{int(d.duplicated().sum())} exact duplicate rows kept")
        return dict(X=X, y=y, g=g, feats=feats, row_id=np.arange(1053), n_raw=1053)
    f = ROOT / cfg["rows"]["file"]
    assert sha256(f) == cfg["rows"]["sha256"], "sha256 mismatch"
    x = pd.read_excel(f)
    assert x.shape == (227, 32)
    feats = [c for c in x.columns if c not in ("No", "LOI")]
    assert len(feats) == 30
    Xall = x[feats].to_numpy(np.float64)
    # their 2-of-3 outlier vote on standardised features; LOI is not read here
    Z = StandardScaler().fit_transform(Xall)
    votes = ((IsolationForest(contamination=0.05, random_state=8585).fit_predict(Z) == -1).astype(int)
             + (OneClassSVM(nu=0.05, kernel="rbf", gamma="scale").fit_predict(Z) == -1)
             + (LocalOutlierFactor(n_neighbors=2, contamination=0.05).fit_predict(Z) == -1))
    out = x["No"].to_numpy()[votes >= 2].tolist()
    keep = votes < 2
    if out != [3, 18, 27, 32, 33, 34, 40, 42, 46, 54] or keep.sum() != 217:
        raise SystemExit(f"BLOCKED: outlier rule gives {out} ({keep.sum()} rows)")
    log(f"ma: outlier rule reproduces No={out} -> {int(keep.sum())} rows")
    yall = x["LOI"].to_numpy(np.float64)
    return dict(X=Xall[keep], y=yall[keep], g=None, feats=feats, row_id=x["No"].to_numpy()[keep], n_raw=227,
                X227=Xall, y227=yall, row_id227=x["No"].to_numpy())


# ---- splits (written before any fit) ---------------------------------------------------------------------------------
def p1_splits(X, y, test_size, seeds):
    out = []
    for s in seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=s)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=s)
        assert np.array_equal(X[tr], Xtr) and np.array_equal(y[te], yte)
        out.append(dict(seed=s, train=tr.tolist(), test=te.tolist()))
    return out


def make_splits(ds, D):
    S = {}
    if ds == "chen_cej2023":
        tr, te = train_test_split(np.arange(1053), random_state=1)  # their main.py call, default test_size
        Xtr, _, _, yte = train_test_split(D["X"], D["y"], random_state=1)
        assert np.array_equal(D["X"][tr], Xtr) and np.array_equal(D["y"][te], yte)
        S["P1_pub"] = [dict(seed=1, test_size="default (0.25)", train=tr.tolist(), test=te.tolist())]
        S["P1"] = p1_splits(D["X"], D["y"], 0.25, range(10))
        assert S["P1"][1]["train"] == S["P1_pub"][0]["train"] and S["P1"][1]["test"] == S["P1_pub"][0]["test"]
        S["P2"] = []
        for r in range(10):
            folds = GroupKFold(n_splits=5, shuffle=True, random_state=r).split(D["X"], D["y"], D["g"])
            S["P2"].append(dict(repeat=r, folds=[dict(train=a.tolist(), test=b.tolist()) for a, b in folds]))
    else:
        S["P1"] = p1_splits(D["X"], D["y"], 0.2, range(10))
        S["P1_all227"] = p1_splits(D["X227"], D["y227"], 0.2, range(10))
    return S


# ---- runners ------------------------------------------------------------------------------------------------------------
def run_jobs(tasks, ds, X, y):
    """tasks: list of (cand, key, tr, te). r0/c2 in 5 loky workers; c1 in-process (its forests use n_jobs=-1 -> 5 threads)."""
    par = [t for t in tasks if t[0] in ("r0", "c2")]
    par.sort(key=lambda t: t[0] != "c2")  # slowest first
    res = dict(zip([(c, k) for c, k, _, _ in par],
                   Parallel(n_jobs=N_JOBS, backend="loky")(delayed(job)(c, ds, X, y, np.array(tr), np.array(te))
                                                           for c, k, tr, te in par)))
    for c, k, tr, te in tasks:
        if c == "c1":
            res[(c, k)] = job(c, ds, X, y, np.array(tr), np.array(te))
    return res


def run_p1(ds, X, y, splits, cands):
    t = time.time()
    res = run_jobs([(c, sp["seed"], sp["train"], sp["test"]) for c in cands for sp in splits], ds, X, y)
    out = {}
    for c in cands:
        rows = [res[(c, sp["seed"])] for sp in splits]
        bad = [sp["seed"] for sp, r in zip(splits, rows) if r.get("error")]
        err = f"{len(y)}-row set, seeds {bad}: {rows[[sp['seed'] for sp in splits].index(bad[0])]['error']}" if bad else None
        out[c] = dict(error=err, preds=None if err else [r["pred"] for r in rows],
                      warnings=sorted({m for r in rows for m in r.get("warnings", [])}),
                      attrs=rows[0].get("attrs"))
    add_c3(out)
    for c, o in out.items():
        if o["error"] is None:
            o["per"] = [dict(seed=sp["seed"], n_train=len(sp["train"]), n_test=len(sp["test"]),
                             **metrics(y[np.array(sp["test"])], p)) for sp, p in zip(splits, o["preds"])]
    log(f"P1 block ({', '.join(cands)}) {time.time() - t:.0f}s")
    return out


def add_c3(out):
    if "c1" in out and "c2" in out:
        if out["c2"]["error"] or out["c1"]["error"]:
            out["c3"] = dict(error="BLOCKED with c2: " + str(out["c2"]["error"] or out["c1"]["error"]), preds=None)
        else:
            out["c3"] = dict(error=None, preds=[(a + b) / 2 for a, b in zip(out["c1"]["preds"], out["c2"]["preds"])],
                             warnings=out["c2"]["warnings"], attrs=out["c2"]["attrs"])


def run_p2(ds, D, splits):
    X, y, g = D["X"], D["y"], D["g"]
    t = time.time()
    # fold check: our stored folds == eval_protocol.repeated_group_kfold folds (DummyRegressor OOF must be bitwise equal)
    _, oof_ep = ep.repeated_group_kfold(DummyRegressor(), X, y, g)
    oof_ours = np.empty_like(oof_ep)
    for rp in splits:
        for f in rp["folds"]:
            oof_ours[rp["repeat"], f["test"]] = ep._fit_predict(DummyRegressor(), X[f["train"]], y[f["train"]], X[f["test"]], False)
    assert np.array_equal(oof_ep, oof_ours), "P2 folds differ from eval_protocol.repeated_group_kfold"
    log("P2 folds == eval_protocol.repeated_group_kfold folds (DummyRegressor OOF bitwise equal)")
    tasks = [(c, (rp["repeat"], i), f["train"], f["test"]) for c in ("r0", "c2") for rp in splits for i, f in enumerate(rp["folds"])]
    res = run_jobs(tasks, ds, X, y)
    out = {}
    for c in ("r0", "c2"):
        bad = [k for (cc, k), r in res.items() if cc == c and r.get("error")]
        err = f"P2 (repeat, fold) {bad}: {res[(c, bad[0])]['error']}" if bad else None
        oof = np.empty((10, len(y)))
        if err is None:
            for rp in splits:
                for i, f in enumerate(rp["folds"]):
                    oof[rp["repeat"], f["test"]] = res[(c, (rp["repeat"], i))]["pred"]
        rows = [v for (cc, _), v in res.items() if cc == c]
        out[c] = dict(error=err, oof=None if err else oof,
                      warnings=sorted({m for r in rows for m in r.get("warnings", [])}),
                      attrs=next((r.get("attrs") for r in rows if r.get("attrs")), None))
    log(f"P2 r0 + c2 {time.time() - t:.0f}s; c1 via eval_protocol.repeated_group_kfold (literal)")
    rk, oof1 = ep.repeated_group_kfold(c1(), X, y, g)
    out["c1"] = dict(error=None, oof=oof1)
    if out["c2"]["error"]:
        out["c3"] = dict(error="BLOCKED with c2: " + out["c2"]["error"], oof=None)
    else:
        out["c3"] = dict(error=None, oof=(oof1 + out["c2"]["oof"]) / 2, warnings=out["c2"]["warnings"], attrs=out["c2"]["attrs"])
    for c, o in out.items():
        if o["error"] is None:
            o["per"] = [dict(repeat=r, **metrics(y, o["oof"][r])) for r in range(10)]
            o["ci"] = [float(v) for v in ep.bootstrap_ci(y, o["oof"][0], g)]
    log(f"P2 total {time.time() - t:.0f}s")
    return out


# ---- reporting --------------------------------------------------------------------------------------------------------
DESC = {"r0": "their published model (reproduction control); see round_config 'r0 (reproduction control)'",
        "c1": "Dravix LOI recipe: equal average of RF(500, msl=2), ET(500, msl=2), median-impute+indicator/StandardScaler/Ridge(10)",
        "c2": "TabPFN v2 regressor, package defaults (tabpfn 9.1.0)",
        "c3": "(c1 + c2) / 2, both fit on the same training rows"}
HP = {"c1": "RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1); ExtraTreesRegressor(same); "
            "make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))",
      "c2": "TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0); nothing else passed",
      "c3": "mean of c1 and c2 predictions"}


def summarise(per):
    a = {k: np.array([p[k] for p in per]) for k in ("r2", "rmse", "mae")}
    return dict(r2_mean=float(a["r2"].mean()), r2_sd=float(a["r2"].std()), r2_repeat0=float(a["r2"][0]),
                rmse_mean=float(a["rmse"].mean()), rmse_sd=float(a["rmse"].std()), mae_mean=float(a["mae"].mean()),
                mae_sd=float(a["mae"].std()))


def write_result(ds, cfg, D, cand, proto, o, r0per, extra, wall):
    slug = f"samedata_{ds}_{cand}_{proto}"
    base = dict(slug=slug, target=cfg["target"], round_config=f"experiments/configs/samedata_{ds}.json",
                shared_config="experiments/configs/samedata_common.json",
                dataset=f"{cfg['source']['paper']}; {cfg['rows']['file']} ({len(D['y'])} rows)",
                features=D["feats"], model=DESC[cand], hyperparameters=HP.get(cand, cfg["r0 (reproduction control)"] if isinstance(
                    cfg["r0 (reproduction control)"], str) else cfg["r0 (reproduction control)"]["definition"]),
                code="data/v2/train_samedata.py", n_rows=int(len(D["y"])),
                n_groups=int(len(np.unique(D["g"]))) if D["g"] is not None else None,
                protocol=f"{proto}: {cfg['protocols'][proto] if proto in cfg['protocols'] else extra.get('protocol_text', '')}",
                sota=cfg["bar"]["value"], sota_source=cfg["source"]["paper"], status="ok", blocked_reason="",
                external_check=True, not_a_dravix_target="same-data comparison; never enters leaderboard.csv / numbers_of_record.json")
    if o["error"]:
        base.update(status="blocked", blocked_reason=o["error"])
    else:
        s = summarise(o["per"])
        base.update(s)
        if proto == "P2":
            base.update(rkf_mean=s["r2_mean"], rkf_std=s["r2_sd"], ci_lo=o["ci"][0], ci_hi=o["ci"][1],
                        ci_method="eval_protocol.bootstrap_ci(y, oof[0], groups=DOI), 2000 draws, seed 0 (repeat 0)")
        base["sd_ddof"] = 0
        base["per_repeat"] = o["per"]
        if cand != "r0" and r0per is not None:
            d = np.array([p["r2"] for p in o["per"]]) - np.array([p["r2"] for p in r0per])
            base.update(paired_vs="r0 (same splits)", paired_delta=float(d.mean()),
                        paired_wins=f"{int((d > 0).sum())}/{len(d)}")
    if o.get("block"):
        base.update(status="blocked", blocked_reason=o["block"])
    if cand in ("c2", "c3"):
        base.update(attribution=ATTR, tabpfn_version=__import__("tabpfn").__version__, checkpoint=CKPT.name,
                    checkpoint_sha256=CKPT_SHA, tabpfn_fitted_attrs=o.get("attrs"), tabpfn_warnings=o.get("warnings"))
    base.update(sklearn_version=sklearn.__version__, numpy_version=np.__version__, torch_version=torch.__version__,
                wall_time_s=round(wall, 1), **extra)
    note = f"{cand} {proto}: " + ("BLOCKED (common.c2.on_error); " if base["status"] == "blocked" else "") + (
        o["error"][:150] if o["error"] else f"R2 mean {base['r2_mean']:.4f} sd {base['r2_sd']:.4f} rep0 {base['r2_repeat0']:.4f}")
    note += f"; external same-data check vs {cfg['bar']['value']} ({cfg['round']})"
    if cand in ("c2", "c3"):
        note += f"; {ATTR}"
    base["note"] = note[:300]
    (ROOT / f"experiments/results/{slug}.json").write_text(json.dumps(base, indent=1))
    return base


def flag(ds, text):
    p = ROOT / "experiments/runs/samedata_flags.md"
    new = not p.exists()
    with open(p, "a") as fh:
        if new:
            fh.write("# samedata FLAGs (worker notes for the orchestrator)\n\n")
        fh.write(f"- FLAG [{ds}] {dt.datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()}: {text}\n")
    log("FLAG:", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["chen_cej2023", "ma_jmi2025"])
    ds = ap.parse_args().dataset
    cfg = json.loads((ROOT / f"experiments/configs/samedata_{ds}.json").read_text())
    assert cfg["status"].startswith("declared")
    assert sha256(CKPT) == CKPT_SHA and CKPT.stat().st_size == 44390977, "TabPFN checkpoint mismatch"
    log(f"{ds}: python {sys.version.split()[0]} sklearn {sklearn.__version__} numpy {np.__version__} torch {torch.__version__} "
        f"tabpfn {__import__('tabpfn').__version__}; checkpoint sha256 ok; HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}")
    D = load(ds, cfg)
    S = make_splits(ds, D)
    sp_path = ROOT / f"experiments/runs/samedata_{ds}_splits.json"
    sp_path.write_text(json.dumps(dict(dataset=ds, written_before_any_fit=dt.datetime.now().isoformat(timespec="seconds"),
                                       row_index="0-based position in the analysed matrix (file order after the config's exclusions)",
                                       row_id=D["row_id"].tolist(), splits=S)))
    log(f"wrote {sp_path.relative_to(ROOT)} ({', '.join(f'{k}: {len(v)}' for k, v in S.items())})")
    X, y, bar = D["X"], D["y"], cfg["bar"]["value"]
    R, npz, gate_ok = {}, {}, None
    authors = {"chen_cej2023": "Chen and co-authors", "ma_jmi2025": "Ma and co-authors"}[ds]

    if ds == "chen_cej2023":
        # ---- reproduction gate: r0 on P1_pub ----
        t = time.time()
        sp = S["P1_pub"][0]
        if (len(sp["train"]), len(sp["test"])) != (790, 263):
            flag(ds, f"P1_pub split size: their call train_test_split(X, y, random_state=1) gives {len(sp['train'])}/{len(sp['test'])} "
                     "rows in sklearn 1.7.1 (test = ceil(0.25*1053)), not the paper's stated 790/263. Run literally as the call; "
                     "config 'exact_split_source' says 263, which the call cannot give.")
        g0 = job("r0", ds, X, y, np.array(sp["train"]), np.array(sp["test"]))
        m0 = metrics(y[np.array(sp["test"])], g0["pred"])
        gate_ok = 0.637 <= m0["r2"] <= 0.647
        log(f"GATE r0 P1_pub (train {len(sp['train'])} / test {len(sp['test'])}): R2 {m0['r2']:.4f} RMSE {m0['rmse']:.4f} "
            f"MAE {m0['mae']:.4f} vs published 0.642 -> {'PASS' if gate_ok else 'FAIL'} [0.637, 0.647] ({time.time() - t:.0f}s)")
        if not gate_ok:
            flag(ds, f"Chen protocol not reproduced: r0 P1_pub test R2 {m0['r2']:.4f} outside [0.637, 0.647]; no P1 sentence")
        # ---- P1 ----
        P1 = run_p1(ds, X, y, S["P1"], CANDS[:3])
        assert np.array_equal(P1["r0"]["preds"][1], g0["pred"]), "r0 not deterministic (P1 seed 1 != P1_pub)"
        log("r0 P1 seed 1 predictions == P1_pub gate predictions (bitwise)")
        pub = {"r0": dict(error=None, per=[dict(seed=1, n_train=len(sp["train"]), n_test=len(sp["test"]), **m0)])}
        for c in ("c1", "c2", "c3"):  # information only: P1_pub split == P1 seed-1 split (asserted in make_splits)
            pub[c] = dict(P1[c], per=None if P1[c]["error"] else [P1[c]["per"][1]])
        # ---- P2 ----
        P2 = run_p2(ds, D, S["P2"])
        outs = {"P1_pub": pub, "P1": P1, "P2": P2}
    else:
        P1 = run_p1(ds, X, y, S["P1"], ("r0",))  # r0 first: its P1 mean is the gate (no exact split exists)
        r0m = float(np.mean([p["r2"] for p in P1["r0"]["per"]]))
        gate_ok = 0.889 <= r0m <= 0.949
        log(f"GATE r0 P1 mean R2 {r0m:.4f} vs [0.889, 0.949] -> {'PASS' if gate_ok else 'FAIL'}")
        if not gate_ok:
            flag(ds, f"Ma protocol not reproduced: r0 P1 mean R2 {r0m:.4f} outside [0.889, 0.949]; no P1 sentence")
        loo = [float(r2_score(y[sp["train"]], cross_val_predict(r0(ds), X[sp["train"]], y[sp["train"]], cv=LeaveOneOut())))
               for sp in S["P1"]]
        log(f"r0 LOOCV R2 on each training part: mean {np.mean(loo):.4f} (published 0.945 on theirs)")
        P1["r0"]["loocv_train_r2"] = loo
        P1.update(run_p1(ds, X, y, S["P1"], ("c1", "c2")))  # adds c3
        P227 = run_p1(ds, D["X227"], D["y227"], S["P1_all227"], CANDS[:3])
        outs = {"P1": P1, "P1_all227": P227}

    # ---- common.c2.on_error: any TabPFN exception on this dataset -> c2 and c3 BLOCKED for the dataset ----
    c2_errs = [f"{k}: {o['c2']['error']}" for k, o in outs.items() if o["c2"]["error"]]
    if c2_errs:
        why = ("common.c2.on_error: TabPFN raised on this dataset (" + " | ".join(c2_errs)
               + ") -> c2 and c3 BLOCKED for the dataset, not retried. Metrics already computed elsewhere are kept for "
                 "transparency and are not selectable.")
        for o in outs.values():
            for c in ("c2", "c3"):
                o[c]["block"] = why
        flag(ds, why)

    # ---- results ----
    if ds == "chen_cej2023":
        for c in CANDS:
            R[(c, "P1_pub")] = write_result(ds, cfg, D, c, "P1_pub", pub[c], None, dict(
                role="reproduction gate" if c == "r0" else "information only (never selected on, never the claim)",
                gate_interval=[0.637, 0.647] if c == "r0" else None, gate_pass=gate_ok if c == "r0" else None,
                same_split_as="P1 seed 1"), 0.0)
    r0per = P1["r0"]["per"]
    for c in CANDS:
        extra = {}
        if ds == "ma_jmi2025":
            o227 = P227[c]
            extra["sensitivity_all227"] = (dict(status="blocked", reason=o227["error"]) if o227["error"] else dict(
                note="P1 on all 227 rows (no exclusion), same seeds; reported only, never selected on, never the claim",
                **summarise(o227["per"]), per_repeat=o227["per"]))
            if c == "r0":
                extra.update(loocv_train_r2=P1["r0"]["loocv_train_r2"], loocv_train_r2_mean=float(np.mean(P1["r0"]["loocv_train_r2"])),
                             gate_interval=[0.889, 0.949], gate_pass=gate_ok)
        R[(c, "P1")] = write_result(ds, cfg, D, c, "P1", P1[c], None if c == "r0" else r0per, extra, 0.0)
        if P1[c]["preds"] is not None:
            npz[f"P1_{c}"] = np.array([np.asarray(p) for p in P1[c]["preds"]])
    if ds == "chen_cej2023":
        for c in CANDS:
            R[(c, "P2")] = write_result(ds, cfg, D, c, "P2", P2[c], None if c == "r0" else P2["r0"]["per"], {}, 0.0)
            if P2[c]["oof"] is not None:
                npz[f"P2_{c}_oof"] = P2[c]["oof"]
    np.savez(ROOT / f"experiments/runs/samedata_{ds}_preds.npz", y=y, **npz)

    # ---- selection, triggers, summary ----
    protos = ["P1_pub", "P1", "P2"] if ds == "chen_cej2023" else ["P1"]
    md = [f"# samedata {ds}: summary\n", f"Config: experiments/configs/samedata_{ds}.json (+ samedata_common.json). "
          f"Code: data/v2/train_samedata.py. Bar: R2 {bar} ({cfg['bar']['set']}). {ATTR} (c2, c3).\n"]
    if ds == "chen_cej2023":
        r = R[("r0", "P1_pub")]
        md.append(f"Reproduction gate: r0 on P1_pub (train {r['per_repeat'][0]['n_train']} / test {r['per_repeat'][0]['n_test']}) "
                  f"R2 {r['r2_mean']:.4f} vs published 0.642, interval [0.637, 0.647]: **{'PASS' if gate_ok else 'FAIL'}**.\n")
    else:
        md.append(f"Reproduction gate: r0 P1 mean R2 {R[('r0', 'P1')]['r2_mean']:.4f}, interval [0.889, 0.949]: "
                  f"**{'PASS' if gate_ok else 'FAIL'}** (no exact split exists). r0 LOOCV R2 on the training parts: mean "
                  f"{R[('r0', 'P1')]['loocv_train_r2_mean']:.4f} (their 0.945, context only).\n")
    cstar = {}
    for proto in protos:
        md.append(f"\n## {proto}\n\n| cand | R2 mean ± SD (ddof 0) | R2 repeat 0 | RMSE mean | MAE mean | wins vs r0 |"
                  + (" CI rep-0 |" if proto == "P2" else "") + ("\n|---|---|---|---|---|---|" + ("---|" if proto == "P2" else "")))
        for c in CANDS:
            r = R[(c, proto)]
            if "r2_mean" not in r:
                md.append(f"| {c} | BLOCKED: {r['blocked_reason'][:80]} | | | | |")
                continue
            tag = " (BLOCKED, not selectable)" if r["status"] == "blocked" else ""
            md.append(f"| {c}{tag} | {r['r2_mean']:.4f} ± {r['r2_sd']:.4f} | {r['r2_repeat0']:.4f} | {r['rmse_mean']:.3f} | "
                      f"{r['mae_mean']:.3f} | {r.get('paired_wins', '-')} |" + (f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |" if proto == "P2" else ""))
        if proto == "P1_pub":
            md.append("\nP1_pub is one split (= P1 seed 1); c1-c3 here are information only.")
            continue
        fam = [c for c in ("c1", "c2", "c3") if R[(c, proto)]["status"] == "ok"]
        best = max(fam, key=lambda c: R[(c, proto)]["r2_mean"])  # max keeps the first listed on exact ties
        cstar[proto] = best
        rb, r0r = R[(best, proto)], R[("r0", proto)]
        md.append(f"\nSelected c* ({proto}, highest mean R2 among {', '.join(fam)}): **{best}**. "
                  f"c* R2 > r0 R2 in {rb['paired_wins']} paired repeats (same splits); mean paired difference {rb['paired_delta']:+.4f}.")
        if ds == "ma_jmi2025":
            md.append("\nSensitivity, all 227 rows (no exclusion; never selected on, never the claim): " + "; ".join(
                f"{c} {R[(c, proto)]['sensitivity_all227']['r2_mean']:.4f} ± {R[(c, proto)]['sensitivity_all227']['r2_sd']:.4f}"
                if R[(c, proto)]["sensitivity_all227"].get("r2_mean") is not None else f"{c} blocked"
                for c in CANDS))
        if proto == "P1":
            if rb["r2_mean"] > bar:
                flag(ds, f"exit trigger: c* {best} P1 mean R2 {rb['r2_mean']:.4f} > published bar {bar}; orchestrator writes the blocker for wording")
            if gate_ok:
                s = (f"On {authors}' data and their protocol (10 random splits instead of their one), Dravix's {best} reaches mean R2 "
                     f"{rb['r2_mean']:.3f} (repeat 0: {rb['r2_repeat0']:.3f}) vs their published {bar} (their single split); "
                     f"their own model re-run on the same 10 splits gives {r0r['r2_mean']:.3f}.")
            else:
                s = "No same-protocol sentence: the reproduction gate failed (config rule)."
        else:
            for c in fam:
                if R[(c, proto)]["ci_lo"] > bar:
                    flag(ds, f"exit trigger: {c} P2 ci_lo {R[(c, proto)]['ci_lo']:.4f} > bar {bar}; orchestrator writes the blocker")
            s = (f"Under Dravix's paper-grouped 5x10 protocol on the same rows: {best} {rb['r2_mean']:.3f} "
                 f"[{rb['ci_lo']:.3f}, {rb['ci_hi']:.3f}], their model re-run {r0r['r2_mean']:.3f}.")
        md.append(f"\nAllowed sentence ({proto}, template verbatim):\n\n> {s}")
    md.append("\nAll three candidates and r0 are printed (multiplicity: c* is the best of 3 by mean; winner's curse applies). "
              "No p-values. Wording beyond the template is the owner's decision.")
    md.append(f"\nRuntime: {time.time() - T0:.0f}s wall.")
    (ROOT / f"experiments/runs/samedata_{ds}_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    log(f"done; c* = {cstar}")


if __name__ == "__main__":
    main()
