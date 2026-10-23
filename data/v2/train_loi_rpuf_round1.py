#!/usr/bin/env python3
"""LOI (RPUF literature) round 1 -- follows experiments/configs/loi_round1.json exactly.

Run: python3 data/v2/train_loi_rpuf_round1.py | tee experiments/runs/train_loi_rpuf_round1.log
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
import eval_protocol as ep  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SOTA = 0.84
DATA = ROOT / "data/v2/loi_rpuf_literature.csv"
CLS_PREFIX = "cls_"
EL_COLS = ["el_P", "el_N", "el_Si", "el_B", "el_halogen", "el_metal"]
BOOLS = ["has_EG", "has_APP", "reactive"]

df = pd.read_csv(DATA)
cls_cols = [c for c in df.columns if c.startswith(CLS_PREFIX)]
FEATURES = ["total_fr_wtpct"] + cls_cols + EL_COLS + ["density_kgm3", "n_frs"] + BOOLS
for b in BOOLS:
    df[b] = df[b].astype(str).str.lower().isin(["true", "1"]).astype(float)


def prep(d):
    d = d.reset_index(drop=True)
    X = d[FEATURES].astype(float).to_numpy()
    y = d["loi_pct"].to_numpy(float)
    g = d["ref_id"].to_numpy()
    cl = d[cls_cols].fillna(0.0)
    dom = np.where(cl.max(axis=1) > 0, cl.idxmax(axis=1).str[len(CLS_PREFIX):], "none")
    return X, y, g, dom


bench = df[df.loi_pct.notna() & (df.extraction != "figure")]
assert len(bench) == 333 and bench.ref_id.nunique() == 56, (len(bench), bench.ref_id.nunique())
X, y, g, dom = prep(bench)
print(f"benchmark: {len(y)} rows / {len(np.unique(g))} papers; {len(FEATURES)} features; "
      f"dominant-class groups: {dict(pd.Series(dom).value_counts())}")

# ---- candidates -------------------------------------------------------------
fi = {c: i for i, c in enumerate(FEATURES)}
el_idx = [fi[c] for c in EL_COLS]
oth_idx = [i for i in range(len(FEATURES)) if i not in el_idx]


def ridge():
    # elements: NaN->0 + missing indicators (config F3); everything else: median in-fold + indicators (F4 etc.)
    ct = ColumnTransformer([
        ("el", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True), el_idx),
        ("oth", SimpleImputer(strategy="median", add_indicator=True), oth_idx),
    ])
    return Pipeline([("imp", ct), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)  # NaN-native (sklearn>=1.4)


def hgb():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         min_samples_leaf=5, random_state=0)


CANDS = {"r1": ridge, "rf": rf, "hgb": hgb}
try:
    import xgboost  # noqa: F401
    from xgboost import XGBRegressor
    CANDS["xgb"] = lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, random_state=0)
    xgb_status = "ran"
except ImportError:
    xgb_status = "skipped: not installed"
print("xgb:", xgb_status)

runs = []  # all runs for CSV


def proto_A(mk, X, y, tag):
    sc = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        m = mk().fit(Xtr, ytr)
        sc.append(r2_score(yte, m.predict(Xte)))
        runs.append(dict(set=tag, protocol="A", cand=tag_c[0], rep=s, r2=sc[-1]))
    return np.array(sc)


def proto_B(mk, X, y, g, tag):
    rkf, oof = ep.repeated_group_kfold(mk(), X, y, g)
    for i, v in enumerate(rkf):
        runs.append(dict(set=tag, protocol="B", cand=tag_c[0], rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    return rkf, lo, hi


def proto_C(mk, X, y, dom, tag):
    rc, _ = ep.repeated_group_kfold(mk(), X, y, dom)
    for i, v in enumerate(rc):
        runs.append(dict(set=tag, protocol="C", cand=tag_c[0], rep=i, r2=v))
    return rc


res = {}
tag_c = [None]
for cid, mk in CANDS.items():
    tag_c[0] = cid
    t = time.time()
    A = proto_A(mk, X, y, "bench")
    Bk, lo, hi = proto_B(mk, X, y, g, "bench")
    C = proto_C(mk, X, y, dom, "bench")
    res[cid] = dict(A=A, A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5),
                    B_mean=Bk.mean(), B_std=Bk.std(), ci_lo=lo, ci_hi=hi, C_mean=C.mean())
    print(f"  {cid} done in {time.time()-t:.0f}s")

print("\n=== Summary (333 rows / 56 papers) ===")
print(f"{'cand':5} {'A mean [p2.5, p97.5]':28} {'B mean ± std':16} {'B CI (paper bootstrap)':24} {'C mean':8}")
for cid, r in res.items():
    print(f"{cid:5} {r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(34)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(17)
          + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(25) + f"{r['C_mean']:.3f}")
if xgb_status != "ran":
    print("xgb   skipped: not installed")

best = max(res, key=lambda c: res[c]["B_mean"])
r = res[best]
print(f"\nSelected (highest protocol-B mean): {best}")
print(f"Protocol B ci_lo > {SOTA}: {r['ci_lo'] > SOTA}  (ci_lo={r['ci_lo']:.4f})")
print(f"Protocol A mean > {SOTA} (any candidate): { {c: bool(v['A_mean'] > SOTA) for c, v in res.items()} }; selected: {r['A_mean'] > SOTA}")
trig = []
if r["ci_lo"] > SOTA:
    trig.append(f"TRIGGER: B ci_lo {r['ci_lo']:.3f} vs {SOTA}")
for c, v in res.items():
    if v["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({c}) {v['A_mean']:.3f} vs {SOTA}")
print("Triggers:", trig if trig else "none")

# ---- sensitivities (best candidate by B; reported only) ----------------------
print(f"\n=== Sensitivities ({best}; report only) ===")
mk = CANDS[best]
tag_c[0] = best


def sens(name, d):
    Xs, ys, gs, ds = prep(d)
    A = proto_A(mk, Xs, ys, name)
    Bk, lo, hi = proto_B(mk, Xs, ys, gs, name)
    print(f"  {name}: {len(ys)} rows / {len(np.unique(gs))} papers | A {A.mean():.3f} [{np.percentile(A,2.5):.3f}, "
          f"{np.percentile(A,97.5):.3f}] | B {Bk.mean():.3f} ± {Bk.std():.3f} CI [{lo:.3f}, {hi:.3f}]")


sens("incl_figure_rows", df[df.loi_pct.notna()])
sens("excl_neat_controls", bench[bench.n_frs > 0])
sun = bench[bench.in_sunfu_refs == "y"]
if len(sun) >= 30 and sun.ref_id.nunique() >= 5:
    sens("sunfu_refs_only", sun)
else:
    print(f"  sunfu_refs_only: too few ({len(sun)} benchmark rows / {sun.ref_id.nunique()} papers < 30 rows); not run")

# ---- permutation importance (in-sample diagnostic) ---------------------------
m = mk().fit(X, y)
pi = permutation_importance(m, X, y, n_repeats=10, random_state=0, scoring="r2")
order = np.argsort(-pi.importances_mean)[:8]
print(f"\n=== Permutation importance, {best} fit on all 333 rows (IN-SAMPLE diagnostic, n_repeats=10) ===")
for i in order:
    print(f"  {FEATURES[i]:32} {pi.importances_mean[i]:.4f} ± {pi.importances_std[i]:.4f}")

# ---- outputs -----------------------------------------------------------------
pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round1_all.csv", index=False)
hp = {"r1": "Pipeline(ColumnTransformer[el: constant 0 + missing indicators; other: median + missing indicators], StandardScaler, Ridge alpha=1.0)",
      "rf": "n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native",
      "hgb": "learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=5, random_state=0",
      "xgb": "n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0"}
names = {"r1": "Ridge", "rf": "RandomForestRegressor", "hgb": "HistGradientBoostingRegressor", "xgb": "XGBRegressor"}
note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}] (random 80/20 x50); C(dominant-class GKF) {r['C_mean']:.3f}; "
        f"B primary. independent reconstruction, 333 rows / 56 papers. xgb {xgb_status}.")
assert len(note) <= 300, len(note)
out = dict(slug=f"loi_rpuf_r1_{best}", target="LOI (%) RPUF literature",
           round_config="experiments/configs/loi_round1.json",
           dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (333 rows / 56 papers)",
           features=FEATURES, model=names[best], hyperparameters=hp[best],
           code="data/v2/train_loi_rpuf_round1.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
           protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
           ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
           paired_vs="", paired_delta="", paired_wins="", sota=SOTA, sota_source="sun_fu_2026",
           status="ok", blocked_reason="", note=note)
(ROOT / f"experiments/results/loi_rpuf_r1_{best}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote experiments/results/loi_rpuf_r1_{best}.json; runtime {time.time()-T0:.0f}s")
