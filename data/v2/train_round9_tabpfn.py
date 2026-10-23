#!/usr/bin/env python3
"""Round 9 (B2 TabPFN V2 + B3 in-fold Duan smearing) -- experiments/configs/round9_tabpfn_smearing.json, followed literally.

  python data/v2/train_round9_tabpfn.py --target SETUP      # step 0, synthetic data only; run twice (2nd = fresh process)
  python data/v2/train_round9_tabpfn.py --target pHRR|CR|FIGRA

Per target: step 1 reproduces the incumbent (exit 2 if outside 0.002), then the candidates in config order. pHRR and FIGRA
then run adoption gate v2 over their family (round9_<t>_gate_v2.json). CR's single gate run is over the joint family with
cr_round2 (o1, o2, n1, p1, p2) and is assembled by train_cr_round2.py --assemble-gate. LOI: train_round9_tabpfn_loi.py.
TTI is never run here. Built with PriorLabs-TabPFN.
"""
import os
import sys
from pathlib import Path

TGT = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else ""
CACHE = Path.home() / "Library" / "Caches" / "tabpfn"  # the darwin default of tabpfn get_cache_dir(), made explicit
CKPT = CACHE / "tabpfn-v2-regressor.ckpt"
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(CACHE)
os.environ["TABPFN_EXCLUDE_DEVICES"] = "cuda,mps"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" if (TGT == "SETUP" and not CKPT.exists()) else "1"  # 0 only for the one-time fetch
os.environ["OMP_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import sklearn  # noqa: E402
import tabpfn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS, RESULTS = ROOT / "experiments" / "runs", ROOT / "experiments" / "results"
sys.path.insert(0, str(HERE))
import adoption_gate  # noqa: E402
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402
from train_blend_round1 import Avg, data_cr, data_figra, data_phrr, et, ridge  # noqa: E402

CONFIG = "experiments/configs/round9_tabpfn_smearing.json"
SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
SIZE = 44390977
ATTR = "Built with PriorLabs-TabPFN"
PINS = {"torch": "2.8.0", "scikit-learn": "1.7.1", "numpy": "2.2.6", "scipy": "1.16.1", "pandas": "2.3.3",
        "huggingface-hub": "1.18.0", "safetensors": "0.7.0", "pydantic": "2.13.4", "pydantic-settings": "2.14.1",
        "matplotlib": "3.10.6", "joblib": "1.5.2", "tqdm": "4.67.1", "filelock": "3.18.0", "typing_extensions": "4.15.0",
        "pyparsing": "3.2.4"}
NEW = {"tabpfn": "9.1.0", "einops": "0.8.2", "lightgbm": "4.7.0", "narwhals": "2.26.0", "skrub": "0.10.1", "pydot": "4.0.1",
       "cloudpickle": "3.1.2", "mlx": "0.32.3", "mlx-metal": "0.32.3"}
UV = Path.home() / ".local" / "bin" / "uv"


def tabpfn_model():
    # the only TabPFN configuration in this round; never TabPFNRegressor() bare (that resolves to the 3.5 checkpoint)
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fitted_attrs(m):
    return dict(n_estimators_=int(m.n_estimators_), devices_=[str(d) for d in m.devices_], use_autocast_=bool(m.use_autocast_),
                forced_inference_dtype_=None if m.forced_inference_dtype_ is None else str(m.forced_inference_dtype_))


# ------------------------------------------------------------------ step 0 -------------------------------------------
def setup():
    norm = lambda s: s.lower().replace("_", "-")  # noqa: E731
    bad = [f"{k} {version(k)} != {v}" for k, v in {**PINS, **NEW}.items() if version(k) != v]
    if tabpfn.__version__ != "9.1.0":
        bad.append(f"tabpfn.__version__ {tabpfn.__version__}")
    chk = subprocess.run([str(UV), "pip", "check", "--python", sys.executable], capture_output=True, text=True)
    chk_txt = chk.stdout + chk.stderr
    named = [l for l in chk_txt.splitlines() if any(norm(p) in norm(l) for p in NEW)]
    frz = subprocess.run([str(UV), "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    frz_lines = [l for l in frz.splitlines() if norm(l.split("==")[0]) in {norm(p) for p in {**PINS, **NEW}}]
    if bad or named:
        print(f"BLOCKED: post-install checks failed: {bad} {named}")
        sys.exit(2)
    print("post-install checks OK (versions, uv pip check, tabpfn.__version__ 9.1.0)")

    fetch = os.environ["HF_HUB_OFFLINE"] == "0"
    log_lines = []
    if fetch:
        class H(logging.Handler):
            def emit(self, rec):
                log_lines.append(self.format(rec))
        h = H()
        h.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        lg = logging.getLogger("tabpfn")
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([1, -2, 0.5, 0, 3]) + 0.1 * rng.normal(size=200)
    Xte = rng.normal(size=(50, 5))  # the "50 synthetic rows": next draws of the same generator (see round9_flags.md)
    preds, m = [], None
    for _ in range(2):
        m = tabpfn_model().fit(X, y)
        preds.append(m.predict(Xte))
    for l in log_lines:
        print("  download log:", l)
    if fetch and not any("HuggingFace" in l or "Successfully downloaded" in l for l in log_lines):
        print("WARNING: no HuggingFace download line captured")
    if any("Direct URL" in l or "Attempting download from" in l for l in log_lines):
        print("BLOCKED: weights came from the direct/S3 fallback, not the HF downloader")
        sys.exit(2)

    ck_sha, ck_size = sha256(CKPT), CKPT.stat().st_size
    ckpts = sorted(p.name for p in CACHE.rglob("*.ckpt"))
    forbidden = [p for p in ckpts if p != CKPT.name]
    print(f"checkpoint {CKPT} size {ck_size} sha256 {ck_sha}; ckpt files in cache: {ckpts}")
    if ck_sha != SHA256 or ck_size != SIZE or forbidden:
        print("BLOCKED: weight check failed (sha256 / size / non-V2 checkpoint in cache)")
        sys.exit(2)
    print("S0.3 weight check OK")

    saved = RUNS / "round9_setup_preds.npz"
    if not saved.exists():
        ok = np.array_equal(preds[0], preds[1])
        stage = "in-process (fit+predict twice)"
        if ok:
            np.savez(saved, pred=preds[0], X=X, y=y, Xte=Xte)
    else:
        q = np.load(saved)["pred"]
        ok = np.array_equal(preds[0], preds[1]) and np.array_equal(preds[0], q)
        stage = "fresh process vs saved predictions (and twice in-process)"
    print(f"S0.4 determinism, {stage}: bitwise equal = {ok}")
    if not ok:
        print("BLOCKED: TabPFN predictions not bitwise reproducible")
        sys.exit(2)
    at = fitted_attrs(m)
    print(f"S0.5 fitted attributes: {at} (expected 8, cpu, False, None)")

    env = RUNS / "round9_env.txt"
    prev = env.read_text() if env.exists() else ""
    with open(env, "w") as f:
        f.write(f"# round9_tabpfn_smearing step 0 -- written by data/v2/train_round9_tabpfn.py --target SETUP, {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"python {sys.version.split()[0]} ({sys.executable}); config pin is 3.12.4 on Arnav's machine, this venv is the stand-in\n")
        f.write(f"invocation: HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']} ({'weight fetch' if fetch else 'offline; scoring runs use 1'})\n")
        for k in ("TABPFN_MODEL_CACHE_DIR", "TABPFN_EXCLUDE_DEVICES", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE", "OMP_NUM_THREADS"):
            f.write(f"{k}={os.environ[k]}\n")
        f.write(f"torch.get_num_threads()={torch.get_num_threads()}\n")
        f.write("# freeze (~/.local/bin/uv pip freeze --python .venv312/bin/python), packages named in software_pins:\n")
        f.write("\n".join(frz_lines) + "\n")
        f.write(f"# uv pip check: {' | '.join(chk_txt.strip().splitlines())}\n")
        f.write(f"tabpfn.__version__={tabpfn.__version__}\n")
        f.write(f"checkpoint={CKPT} size={ck_size} sha256={ck_sha} (expected {SHA256}, {SIZE}); cache ckpt files={ckpts}\n")
        f.write(f"S0.4 determinism: {stage}: bitwise equal {ok}\n")
        f.write(f"S0.5 fitted attributes: {json.dumps(at)}\n")
        for l in log_lines:
            f.write(f"download log: {l}\n")
        if prev:
            f.write("# ---- previous invocation ----\n" + prev)
    print(f"wrote {env.relative_to(ROOT)}")


# ------------------------------------------------------------------ helpers ------------------------------------------
def smear_oof(M, X, y, g, inc_oof):
    """B3 algorithm verbatim; also the per-fold consistency check against the incumbent's OOF (relative 1e-9)."""
    oof, facs = np.empty((10, len(y))), []
    for r in range(10):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, g):
            assert set(g[tr]).isdisjoint(g[te])
            z = np.log1p(y[tr])
            m = clone(M).fit(X[tr], z)  # identical to the incumbent's fold model
            inner = np.empty(len(tr))
            for a, b in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X[tr], z, g[tr]):
                inner[b] = clone(M).fit(X[tr][a], z[a]).predict(X[tr][b])
            s = np.mean(np.exp(z - inner))  # unweighted, not clipped
            oof[r, te] = np.exp(m.predict(X[te])) * s - 1
            if not np.allclose((oof[r, te] + 1) / s, inc_oof[r, te] + 1, rtol=1e-9, atol=0):
                print(f"BLOCKED: smearing consistency check failed (repeat {r})")
                sys.exit(2)
            facs.append(float(s))
    f = np.array(facs)
    return oof, dict(n=len(f), mean=float(f.mean()), min=float(f.min()), max=float(f.max()), share_below_1=float((f < 1).mean()),
                     factors=[float(v) for v in f])


def topk(y, g, oof_inc, oof_c):
    """reported_only.top_k_removal (same construction as cr_round2.json). Diagnostic only."""
    row = ((y - oof_inc) ** 2 - (y - oof_c) ** 2).mean(axis=0)
    mats = np.unique(g)
    con = {str(m): float(row[g == m].sum()) for m in mats}
    ranked = sorted(con, key=lambda m: (-con[m], m))
    tot = sum(con.values())
    out = dict(top5=[[m, con[m]] for m in ranked[:5]], top5_share_of_net=sum(con[m] for m in ranked[:5]) / tot if tot else None)
    for k in (1, 3, 5):
        keep = ~np.isin(g.astype(str), ranked[:k])
        d = np.array([r2_score(y[keep], oof_c[r, keep]) - r2_score(y[keep], oof_inc[r, keep]) for r in range(10)])
        out[f"drop_top{k}"] = dict(mean_gain=float(d.mean()), wins=int((d > 0).sum()))
    out["not_established"] = bool(out["drop_top5"]["mean_gain"] <= 0 or out["drop_top5"]["wins"] < 8)
    return out


def score(model, X, y, g, logt):
    t = time.time()
    rk, oof = repeated_group_kfold(model, X, y, g, log_target=logt)
    lo, hi = bootstrap_ci(y, oof[0], g)
    plo, phi = bootstrap_ci(y, oof.mean(axis=0), g)  # pooled CI as train_phrr_promote_c2.pooled_ci
    return dict(rk=rk, oof=oof, lo=float(lo), hi=float(hi), plo=float(plo), phi=float(phi), wall=time.time() - t)


def finish(res, oof, y, g, t0):
    rk = np.array([r2_score(y, p) for p in oof])
    lo, hi = bootstrap_ci(y, oof[0], g)
    plo, phi = bootstrap_ci(y, oof.mean(axis=0), g)
    return dict(res, rk=rk, oof=oof, lo=float(lo), hi=float(hi), plo=float(plo), phi=float(phi), wall=time.time() - t0)


def line(cid, r, inc=None):
    s = f"{cid:4s} GKF {r['rk'].mean():.4f} +- {r['rk'].std():.4f} CI [{r['lo']:.4f}, {r['hi']:.4f}] pooled [{r['plo']:.4f}, {r['phi']:.4f}]"
    if inc is not None:
        d = r["rk"] - inc["rk"]
        s += f"  d {d.mean():+.4f} wins {(d > 0).sum()}/10"
    if "cat" in r:
        s += f"  category-GKF {r['cat']:.4f}"
    return s + f"  ({r['wall']:.0f}s)"


SPEC = {
    "pHRR": dict(data=data_phrr, exp="exp074", nor="pHRR", rec=(0.6113, 0.5155, 0.8376), slug="phrr",
                 hgb=dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)),
    "CR": dict(data=data_cr, exp="exp117", nor="Char Residue", rec=(0.6504, 0.4389, 0.7893), slug="cr",
               hgb=dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)),
    "FIGRA": dict(data=data_figra, exp="exp119", nor="FIGRA", rec=(0.4975, 0.3464, 0.6475), slug="figra",
                  hgb=dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=5, random_state=0)),
}
DESC = {
    "pHRR": {"p1": "TabPFN V2 alone on the 10 exp074 features, log1p scale",
             "p2": "Avg([exp074 HGB, TabPFN V2]) on the log1p scale (0.5 HGB + 0.5 TabPFN), then expm1",
             "s1": "exp074 HGB unchanged with in-fold Duan smearing"},
    "CR": {"p1": "TabPFN V2 alone on the 11 exp117 features, raw %, no clipping",
           "p2": "Avg([exp117 model, TabPFN V2]) raw scale, nested (HGB/ET/Ridge 1/6 each, TabPFN 1/2), no clipping"},
    "FIGRA": {"p1": "TabPFN V2 alone on the 9 exp119 features, log1p scale",
              "p2": "Avg([exp119 model, TabPFN V2]) in the log1p wrapper, nested (HGB/ET/Ridge 1/6 each, TabPFN 1/2), then expm1",
              "s1": "exp119 model unchanged with in-fold Duan smearing"},
}


def run_target(t):
    t0 = time.time()
    S = SPEC[t]
    nor = json.load(open(ROOT / "experiments" / "numbers_of_record.json"))["targets"][S["nor"]]["exp_id"]
    if nor != S["exp"]:
        print(f"STOP: numbers_of_record names {nor} for {t}, config expects {S['exp']}; amend the config first")
        sys.exit(2)
    D = S["data"]()
    X, y, g, logt = D["X"], D["y"], D["g"], D["log"]
    hp = {k: v for k, v in D["inc"].get_params().items() if k in S["hgb"]}
    assert hp == S["hgb"], hp
    inc_model = D["inc"] if t == "pHRR" else Avg([D["inc"], et(), ridge()])
    print(f"{t}: {len(y)} rows / {len(np.unique(g))} groups / {X.shape[1]} features; log1p: {logt}; incumbent {S['exp']}")
    print(f"torch threads {torch.get_num_threads()}, OMP {os.environ['OMP_NUM_THREADS']}, "
          f"LOKY_MAX_CPU_COUNT {os.environ.get('LOKY_MAX_CPU_COUNT', 'unset')}, HF_HUB_OFFLINE {os.environ['HF_HUB_OFFLINE']}")
    cat = D["extra"][1] if t == "pHRR" else None

    # ---- step 1 ----
    inc = score(inc_model, X, y, g, logt)
    if cat is not None:
        inc["cat_rk"], inc["cat_oof"] = repeated_group_kfold(inc_model, X, y, cat, log_target=logt)
        inc["cat"] = float(inc["cat_rk"].mean())
    print(line("inc", inc), flush=True)
    m0, l0, h0 = S["rec"]
    if not (abs(inc["rk"].mean() - m0) <= 0.002 and abs(inc["lo"] - l0) <= 0.002 and abs(inc["hi"] - h0) <= 0.002):
        print(f"BLOCKED: {S['exp']} not reproduced within 0.002 (mean, ci_lo, ci_hi = {inc['rk'].mean():.4f}, "
              f"{inc['lo']:.4f}, {inc['hi']:.4f}); stopping.")
        sys.exit(2)
    print(f"REPRO {S['exp']} ok (record {S['rec']})", flush=True)

    # ---- candidates, config order ----
    cands = {"p1": tabpfn_model(), "p2": Avg([clone(inc_model), tabpfn_model()])}
    order = ["p1", "p2"] + (["s1"] if t in ("pHRR", "FIGRA") else [])
    res = {}
    for cid in order:
        if cid == "s1":
            t1 = time.time()
            oof, fac = smear_oof(inc_model, X, y, g, inc["oof"])
            r = finish(dict(fac=fac), oof, y, g, t1)
            if cat is not None:
                coof, cfac = smear_oof(inc_model, X, y, cat, inc["cat_oof"])
                r["cat"] = float(np.mean([r2_score(y, p) for p in coof]))
                r["cat_fac"] = {k: v for k, v in cfac.items() if k != "factors"}
        else:
            r = score(cands[cid], X, y, g, logt)
            if cat is not None:
                r["cat"] = float(repeated_group_kfold(cands[cid], X, y, cat, log_target=logt)[0].mean())
        res[cid] = r
        print(line(cid, r, inc), flush=True)

    # TabPFN fitted attributes (one extra fit on repeat-0 fold-0 training rows, metadata only)
    tr0 = next(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g))[0]
    attrs = fitted_attrs(tabpfn_model().fit(X[tr0], np.log1p(y[tr0]) if logt else y[tr0]))
    print("TabPFN fitted attributes:", attrs)

    # ---- outputs ----
    trig = [c for c in order if t == "CR" and res[c]["lo"] > 0.82]
    print("Triggers:", trig or "none")
    paths = {}
    for cid in order:
        r = res[cid]
        d = r["rk"] - inc["rk"]
        tk = topk(y, g, inc["oof"], r["oof"])
        slug = f"r9_{'smear' if cid == 's1' else 'tabpfn'}_{S['slug']}_{cid}"
        note = f"round 9 {t} {cid} (provisional id {slug}); report vs {S['exp']} on identical folds"
        if "cat" in r:
            note += f"; category-GKF {r['cat']:.3f} (incumbent {inc['cat']:.3f})"
        if tk["not_established"]:
            note += "; gain concentrated in <= 5 materials, not established"
        if cid != "s1":
            note += f"; {ATTR}"
        out = dict(slug=slug, target=D["target"], round_config=CONFIG,
                   dataset=f"{D['dataset'].rsplit(', as ', 1)[0]}; incumbent {S['exp']} rows, order and groups",
                   features=D["feats"], model=DESC[t][cid],
                   hyperparameters=("TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0); "
                                    "package defaults, tabpfn 9.1.0" if cid != "s1" else
                                    "Duan smearing factor from inner GroupKFold(5, shuffle, random_state=0) OOF log residuals; no free parameter"),
                   code="data/v2/train_round9_tabpfn.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))), protocol="gkf_group5x10",
                   rkf_mean=round(float(r["rk"].mean()), 4), rkf_std=round(float(r["rk"].std()), 4),
                   ci_lo=round(r["lo"], 4), ci_hi=round(r["hi"], 4), pooled_ci_lo=round(r["plo"], 4), pooled_ci_hi=round(r["phi"], 4),
                   paired_vs=S["exp"], paired_delta=round(float(d.mean()), 4), paired_wins=f"{int((d > 0).sum())}/10",
                   sota=D["sota"], sota_source=D["sota_src"], status="ok", blocked_reason="", note=note,
                   rkf_mean_full=float(r["rk"].mean()), ci_lo_full=r["lo"], ci_hi_full=r["hi"],
                   per_repeat_r2=[float(v) for v in r["rk"]], top_k_removal=tk, wall_time_s=round(r["wall"], 1))
        if "cat" in r:
            out["category_grouped_mean"] = round(r["cat"], 4)
        if cid == "s1":
            out["smearing_factors"] = r["fac"]
            if "cat_fac" in r:
                out["smearing_factors_category_gkf"] = r["cat_fac"]
        else:
            out.update(tabpfn_version=tabpfn.__version__, torch_version=torch.__version__, sklearn_version=sklearn.__version__,
                       checkpoint=CKPT.name, checkpoint_sha256=sha256(CKPT), attribution=ATTR, **attrs)
        p = RESULTS / f"{slug}.json"
        p.write_text(json.dumps(out, indent=2))
        paths[cid] = p
        print(f"wrote {p.relative_to(ROOT)}  top-k: {json.dumps({k: tk[k] for k in ('drop_top1', 'drop_top3', 'drop_top5')})}")

    sel = order  # every family member is selectable in round 9 (pHRR, FIGRA, CR p1/p2)
    arrays = dict(y=y, groups=np.asarray(g).astype(str), row_id=np.arange(len(y)), oof_incumbent=inc["oof"],
                  inc_per_repeat_r2=inc["rk"], inc_ci=np.array([inc["lo"], inc["hi"]]),
                  order=np.array(order), selectable=np.array(sel), **{f"oof_{c}": res[c]["oof"] for c in order})
    if t == "CR":  # joint CR family with cr_round2: these are B2's CR arrays, assembled by train_cr_round2.py --assemble-gate
        npz = RUNS / "round9_cr_b2_oof.npz"
        np.savez(npz, **arrays)
        print(f"wrote {npz.relative_to(ROOT)} (B2 CR members; the CR gate runs once over [o1, o2, n1, p1, p2])")
    else:
        npz = RUNS / f"round9_{S['slug']}_oof.npz"
        np.savez(npz, **arrays)
        print(f"wrote {npz.relative_to(ROOT)}")
        gate = adoption_gate.run_npz(str(npz), {c: str(p) for c, p in paths.items()}, str(RUNS / f"round9_{S['slug']}_gate_v2.json"))
        annotate(paths, gate)
    print(f"runtime {time.time() - t0:.0f}s")


def annotate(paths, gate):
    """After the gate: add the gate_v2 per-candidate report to each family member's result JSON (reported only)."""
    for c, p in paths.items():
        o = json.loads(Path(p).read_text())
        o["gate_v2"] = dict(gate["per_candidate"][c], K=gate["K"], q=gate["q"], selected=(c == gate["best"]))
        if c == gate["best"]:
            o["note"] += (f"; gate v2 selected (K={gate['K']}, q={gate['q']:.3f}, LB={gate['per_candidate'][c]['lb_maxT']:+.4f}): "
                          f"G1 {gate['G1']}, G2 {gate['G2']}, G3 {gate['G3']}")
        Path(p).write_text(json.dumps(o, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["SETUP", "pHRR", "CR", "FIGRA"])
    a = ap.parse_args()
    if a.target == "SETUP":
        setup()
    else:
        if not CKPT.exists() or sha256(CKPT) != SHA256:
            print("BLOCKED: step 0 has not passed (checkpoint missing or hash mismatch)")
            sys.exit(2)
        run_target(a.target)
