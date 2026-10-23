#!/usr/bin/env python3
"""LOI (RPUF literature) round 2 -- follows experiments/configs/loi_round2.json exactly.

Data, features, protocols A/B/C, seeds and CI method are copied from train_loi_rpuf_round1.py
(that script runs on import, so it is not imported).

Run: python3 data/v2/train_loi_rpuf_round2.py | tee experiments/runs/train_loi_rpuf_round2.log
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
n_le = int((y <= 15.5).sum())
print(f"rows with LOI <= 15.5: {n_le} (min LOI {y.min():.1f})")
assert n_le == 0

fi = {c: i for i, c in enumerate(FEATURES)}
MONO_COLS = ["el_P", "el_N", "total_fr_wtpct", "cls_expandable_graphite", "cls_intumescent(APP-based)", "cls_P-N"]
MONO = [1 if c in MONO_COLS else 0 for c in FEATURES]
assert sum(MONO) == 6


# ---- models ---------------------------------------------------------------
def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def hgb_mono():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         min_samples_leaf=5, random_state=0, monotonic_cst=MONO)


class LogShiftRF(BaseEstimator, RegressorMixin):
    """round-1 rf on y' = log(LOI - 15); predictions back-transformed exp(.)+15."""
    def fit(self, X, y):
        self.m_ = rf().fit(X, np.log(np.asarray(y) - 15.0))
        return self

    def predict(self, X):
        return np.exp(self.m_.predict(X)) + 15.0


class AvgRFHGB(BaseEstimator, RegressorMixin):
    """equal-weight average of round-1 rf and c2 (hgb_mono) predictions, per fold."""
    def fit(self, X, y):
        self.a_ = rf().fit(X, y)
        self.b_ = hgb_mono().fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.a_.predict(X) + 0.5 * self.b_.predict(X)


BASE = ("rf", rf)
NEW = {"c1": LogShiftRF, "c2": hgb_mono, "c3": AvgRFHGB}
SHORT = {"c1": "rf_logshift", "c2": "hgb_mono", "c3": "rf_hgbmono_avg"}
MODEL_NAME = {"c1": "RandomForestRegressor on log(LOI-15)", "c2": "HistGradientBoostingRegressor (monotonic_cst)",
              "c3": "0.5*RandomForestRegressor + 0.5*HistGradientBoostingRegressor (monotonic_cst)"}
HP = {"c1": "rf n_estimators=500, min_samples_leaf=2, random_state=0 on y'=log(LOI-15), pred=exp(.)+15",
      "c2": "learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=5, random_state=0, "
            "monotonic_cst +1 on " + ", ".join(MONO_COLS),
      "c3": "equal-weight mean of round-1 rf (n_estimators=500, min_samples_leaf=2, random_state=0) and c2 per fold"}

runs = []


def run_all(cid, mk):
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, mk().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(set="bench", protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    Bk, oof = ep.repeated_group_kfold(mk(), X, y, g)
    for i, v in enumerate(Bk):
        runs.append(dict(set="bench", protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    Ck, _ = ep.repeated_group_kfold(mk(), X, y, dom)
    for i, v in enumerate(Ck):
        runs.append(dict(set="bench", protocol="C", cand=cid, rep=i, r2=v))
    print(f"  {cid} done in {time.time()-t:.0f}s")
    return dict(A=A, A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5),
                Bk=Bk, B_mean=Bk.mean(), B_std=Bk.std(), ci_lo=lo, ci_hi=hi, C_mean=Ck.mean())


# ---- reproduce round-1 rf -------------------------------------------------
res = {"rf": run_all(*BASE)}
r0 = res["rf"]
print(f"\nReproduction of round-1 rf: B {r0['B_mean']:.4f} CI [{r0['ci_lo']:.4f}, {r0['ci_hi']:.4f}] A {r0['A_mean']:.4f} "
      f"(round 1: B 0.3512, CI [0.1296, 0.5196], A 0.6296)")
ok = (abs(r0["B_mean"] - 0.351) <= 0.002 and abs(r0["ci_lo"] - 0.130) <= 0.002 and abs(r0["ci_hi"] - 0.520) <= 0.002)
print("Reproduction within 0.002:", ok)
if not ok:
    print("BLOCKED: round-1 rf not reproduced; stopping.")
    pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round2_all.csv", index=False)
    sys.exit(2)

for cid, mk in NEW.items():
    res[cid] = run_all(cid, mk)

# ---- summary --------------------------------------------------------------
for cid in NEW:
    d = res[cid]["Bk"] - r0["Bk"]
    res[cid]["wins"] = int((d > 0).sum())
    res[cid]["delta"] = float(d.mean())

print("\n=== Summary (333 rows / 56 papers) ===")
print(f"{'cand':5} {'A mean [p2.5, p97.5]':24} {'B mean ± std':16} {'B CI (paper bootstrap)':22} {'C mean':8} {'wins/10':8} {'dB'}")
for cid, r in res.items():
    pw = f"{r['wins']}/10" if cid in NEW else "-"
    dl = f"{r['delta']:+.4f}" if cid in NEW else "-"
    print(f"{cid:5} {r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(30)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(17)
          + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(23) + f"{r['C_mean']:.3f}".ljust(9) + pw.ljust(8) + dl)

best = max(NEW, key=lambda c: res[c]["B_mean"])
b = res[best]
adopt_wins = b["wins"] >= 8
adopt_ci = b["ci_lo"] >= 0.130
adopt_ci_vs_repro = b["ci_lo"] >= r0["ci_lo"]
adopt = adopt_wins and adopt_ci
print(f"\nSelected (highest protocol-B mean among c1-c3): {best} (B {b['B_mean']:.4f})")
print(f"Rule: wins>=8/10: {adopt_wins} ({b['wins']}/10); B ci_lo>=0.130: {adopt_ci} ({b['ci_lo']:.4f}; "
      f"vs reproduced rf ci_lo {r0['ci_lo']:.4f}: {adopt_ci_vs_repro})")
print(f"Verdict: {'ADOPT ' + best + ' over round-1 rf' if adopt else 'NOT ADOPTED; round-1 rf remains incumbent'}")
trig = []
for c, v in res.items():
    if c in NEW and v["ci_lo"] > SOTA:
        trig.append(f"TRIGGER: B ci_lo ({c}) {v['ci_lo']:.3f} vs {SOTA}")
    if c in NEW and v["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({c}) {v['A_mean']:.3f} vs {SOTA}")
print("Triggers:", trig if trig else "none")

# ---- outputs --------------------------------------------------------------
pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round2_all.csv", index=False)
for cid in NEW:
    r = res[cid]
    note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}] (random 80/20 x50); C(dominant-class GKF) {r['C_mean']:.3f}; "
            f"B primary. paired vs round-1 rf: {r['wins']}/10 repeats, mean dB {r['delta']:+.4f}. "
            f"{'selected' if cid == best else 'not selected'}; {'adopted' if (cid == best and adopt) else 'not adopted'}.")
    assert len(note) <= 300, len(note)
    out = dict(slug=f"loi_rpuf_r2_{cid}_{SHORT[cid]}", target="LOI (%) RPUF literature",
               round_config="experiments/configs/loi_round2.json",
               dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (333 rows / 56 papers)",
               features=FEATURES, model=MODEL_NAME[cid], hyperparameters=HP[cid],
               code="data/v2/train_loi_rpuf_round2.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
               protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
               ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
               paired_vs="exp064", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10",
               sota=SOTA, sota_source="sun_fu_2026", status="ok", blocked_reason="", note=note)
    (ROOT / f"experiments/results/loi_rpuf_r2_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 3 result JSONs; runtime {time.time()-T0:.0f}s")
