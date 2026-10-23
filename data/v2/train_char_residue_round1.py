#!/usr/bin/env python3
"""Dravix V2 -- Char Residue round 1 (config: experiments/configs/cr_round1.json, pre-declared).

Data loading is a verbatim replica of train_char_residue_v2.py Part A (exp033).
Step 1: reproduce exp033 (rkf 0.538, CI [0.112, 0.713], 497 rows / 55 materials) or write blocked JSONs and stop.
Step 2: c1 (incumbent features + inorganic-fraction proxy), c2 (HGB), c3 (logit target on best of c1/c2).
All on identical repeated GroupKFold 5x10 splits (groups = material). Nothing tuned.

Inorganic proxy: char_yield_sta.csv has only atmosphere N2 (no air runs), so the config's fallback applies:
binary flags for categories whose products contain mineral/inorganic content (gypsum, fibre-cement, mineral
wool / glass fibre, shingles, structural). Fixed a priori (below), not chosen from results.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from eval_protocol import bootstrap_ci, repeated_group_kfold, grouped_shuffle_r2

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
RES_DIR = f"{ROOT}/experiments/results"
SOTA, SOTA_SRC = 0.82, "pomazi_2026"
EXP033 = dict(rkf_mean=0.538, ci_lo=0.112, ci_hi=0.713, rows=497, materials=55)
RF = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)
HGB = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)
MINERAL_CATS = ["interior finish", "exterior finish", "insulation", "roofing", "structural"]


def write_json(cid, short, **kw):
    d = dict(slug=f"cr_r1_{cid}_{short}", target="Char Residue (cone residue %, FSRI)",
             round_config="experiments/configs/cr_round1.json", code="data/v2/train_char_residue_round1.py",
             protocol="rgkf5x10", paired_vs="exp033", sota=SOTA, sota_source=SOTA_SRC)
    d.update(kw)
    with open(f"{RES_DIR}/{d['slug']}.json", "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


# ---------------- exp033 data loading (verbatim from train_char_residue_v2.py Part A) ----------------
cone = pd.read_csv(f"{DATA_DIR}/char_residue_cone_fsri.csv")
sta = pd.read_csv(f"{DATA_DIR}/char_yield_sta.csv")
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")
sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct", aggfunc="mean")
            .rename(columns=lambda r: f"sta_res_{int(r)}K").reset_index())
mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                      mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean")).reset_index()
feat = (sta_wide.merge(mcc_agg, on="material", how="outer")
        .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left"))
df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()].merge(feat, on="material", how="left")
STA_COLS = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
df = df[df[STA_COLS].notna().any(axis=1) & df.mcc_char_yield_pct.notna()].reset_index(drop=True)
y, groups = df.cone_residue_pct.values, df.material.values
BASE = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct"] + STA_COLS  # exp033 A3 full
X0 = df[BASE].values
print(f"rows {len(df)} / materials {df.material.nunique()}")
print(f"STA atmospheres present: {sorted(sta.atmosphere.unique())} -> no air runs; using category flags")

# ---------------- own OOF loop (same splits as eval_protocol.repeated_group_kfold) ----------------
def fwd(v, tf):
    return logit(np.clip(v / 100.0, 0.005, 0.995)) if tf else v


def inv(p, tf):
    return expit(p) * 100.0 if tf else p


def rgkf(model, X, tf=False, repeats=10):
    oof = np.empty((repeats, len(y)))
    for r in range(repeats):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, groups):
            assert set(groups[tr]).isdisjoint(groups[te])
            m = clone(model).fit(X[tr], fwd(y[tr], tf))
            oof[r, te] = inv(m.predict(X[te]), tf)
    return np.array([r2_score(y, p) for p in oof]), oof


def gss(model, X, tf=False):
    s = []
    for seed in range(10):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
        m = clone(model).fit(X[tr], fwd(y[tr], tf))
        s.append(r2_score(y[te], inv(m.predict(X[te]), tf)))
    return np.mean(s)


def evaluate(model, X, tf=False):
    r, oof = rgkf(model, X, tf)
    lo, hi = bootstrap_ci(y, oof[0], groups)
    return dict(r=r, oof=oof, rkf_mean=r.mean(), rkf_std=r.std(), ci_lo=lo, ci_hi=hi, gss_mean=gss(model, X, tf))


# ---------------- Step 1: reproduce exp033 ----------------
base = evaluate(RF, X0)
# cross-check my loop against eval_protocol on the untransformed path
chk, _ = repeated_group_kfold(RF, X0, y, groups)
assert np.allclose(chk, base["r"]), "own loop != eval_protocol"
print(f"exp033 reproduction: rkf {base['rkf_mean']:.4f} (std {base['rkf_std']:.4f}) CI [{base['ci_lo']:.3f}, {base['ci_hi']:.3f}] "
      f"GSS {base['gss_mean']:.4f}")
ok = (len(df) == EXP033["rows"] and df.material.nunique() == EXP033["materials"]
      and abs(base["rkf_mean"] - EXP033["rkf_mean"]) <= 0.002
      and abs(base["ci_lo"] - EXP033["ci_lo"]) <= 0.002 and abs(base["ci_hi"] - EXP033["ci_hi"]) <= 0.002)
if not ok:
    msg = (f"exp033 did not reproduce: rows {len(df)}/{df.material.nunique()}, rkf {base['rkf_mean']:.4f}, "
           f"CI [{base['ci_lo']:.3f}, {base['ci_hi']:.3f}] vs 0.538 [0.112, 0.713] 497/55")
    print("BLOCKED:", msg)
    for cid, short in [("c1", "rf_inorg_flag"), ("c2", "hgb_inorg_flag"), ("c3", "hgb_logit")]:
        write_json(cid, short, status="blocked", blocked_reason=msg, note="Baseline reproduction failed; no candidates run.")
    sys.exit(2)
print("REPRODUCTION OK")

# ---------------- Step 2: candidates ----------------
cat_l = df.categories.fillna("").str.lower()
FLAGS = []
for c in MINERAL_CATS:
    col = "cat_" + c.replace(" ", "_")
    df[col] = cat_l.apply(lambda s, c=c: float(c in [t.strip() for t in s.split(";")]))
    FLAGS.append(col)
X1 = df[BASE + FLAGS].values
print("flag prevalence (materials):", {f: int(df.groupby("material")[f].first().sum()) for f in FLAGS})

cands = {}
cands["c1"] = dict(short="rf_inorg_flag", model=RF, tf=False, desc="RandomForestRegressor", **evaluate(RF, X1))
cands["c2"] = dict(short="hgb_inorg_flag", model=HGB, tf=False, desc="HistGradientBoostingRegressor", **evaluate(HGB, X1))
best12 = max(["c1", "c2"], key=lambda k: cands[k]["rkf_mean"])
print(f"best of c1/c2 by GKF mean: {best12}")
cands["c3"] = dict(short=f"{best12}_logit", model=cands[best12]["model"], tf=True,
                   desc=cands[best12]["desc"] + " + logit target (clip [0.005,0.995])", **evaluate(cands[best12]["model"], X1, tf=True))

print("\n=== Summary (repeated GroupKFold 5x10, identical splits; groups = material) ===")
print(f"{'id':5s}{'rkf_mean':>9s}{'rkf_std':>9s}{'ci_lo':>8s}{'ci_hi':>8s}{'gss':>8s}{'wins':>7s}{'d_mean':>9s}")
print(f"{'exp033':5s}{base['rkf_mean']:9.4f}{base['rkf_std']:9.4f}{base['ci_lo']:8.3f}{base['ci_hi']:8.3f}{base['gss_mean']:8.4f}{'-':>7s}{'-':>9s}")
for k, c in cands.items():
    c["wins"] = int((c["r"] > base["r"]).sum())
    c["delta"] = float((c["r"] - base["r"]).mean())
    print(f"{k:5s}{c['rkf_mean']:9.4f}{c['rkf_std']:9.4f}{c['ci_lo']:8.3f}{c['ci_hi']:8.3f}{c['gss_mean']:8.4f}{c['wins']:>4d}/10{c['delta']:+9.4f}")

# ---------------- selection rule (mechanical) ----------------
sel = max(cands, key=lambda k: cands[k]["rkf_mean"])
s = cands[sel]
promote = s["wins"] >= 8 and s["ci_lo"] >= EXP033["ci_lo"]
print(f"\nSELECTED (highest rkf mean): {sel}; wins {s['wins']}/10 (need >=8), ci_lo {s['ci_lo']:.3f} (need >=0.112) "
      f"-> {'PROMOTE' if promote else 'NO PROMOTION'}")
trig = [k for k, c in cands.items() if c["ci_lo"] > SOTA]
if trig:
    print(f"TRIGGER: ci_lo > {SOTA} for {trig} -> exit condition 5")

# ---------------- diagnostic: per-category error table ----------------
prim = df.categories.fillna("unknown").str.split(";").str[0].str.strip().values


def cat_table(name, oof0):
    err = oof0 - y
    sse = (err ** 2).sum()
    rows = []
    for c in sorted(set(prim)):
        m = prim == c
        rows.append((c, len(set(groups[m])), np.abs(err[m]).mean(), (err[m] ** 2).sum() / sse))
    rows.sort(key=lambda t: -t[3])
    print(f"\nPer-category error ({name}; repeat-0 OOF; category = first listed; not a selection input)")
    print(f"{'category':28s}{'n_mat':>6s}{'mean|err|':>11s}{'sq-err share':>14s}")
    for c, n, mae, sh in rows:
        print(f"{c:28s}{n:6d}{mae:11.2f}{sh:14.3f}")
    return rows


t033 = cat_table("exp033", base["oof"][0])
tsel = cat_table(f"selected {sel}", s["oof"][0])
print("\nTop 3 by error share, exp033:", [(c, round(sh, 3)) for c, _, _, sh in t033[:3]])
print(f"Top 3 by error share, {sel}:", [(c, round(sh, 3)) for c, _, _, sh in tsel[:3]])

# ---------------- result JSONs ----------------
feat_names = BASE + FLAGS
for k, c in cands.items():
    hp = ("RF(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)" if "Random" in c["desc"] else
          "HGB(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)")
    if c["tf"]:
        hp += "; target logit(clip(y/100,0.005,0.995)), predictions expit*100"
    tag = " SELECTED." if k == sel else ""
    write_json(k, c["short"], dataset="exp033 rows (char_residue_cone_fsri x STA/MCC/props), 497 rows / 55 materials; no STA air data so category flags",
               features=feat_names, model=c["desc"], hyperparameters=hp, n_rows=len(df), n_groups=int(df.material.nunique()),
               rkf_mean=round(float(c["rkf_mean"]), 4), rkf_std=round(float(c["rkf_std"]), 4), ci_lo=round(float(c["ci_lo"]), 4),
               ci_hi=round(float(c["ci_hi"]), 4), gss_mean=round(float(c["gss_mean"]), 4), paired_delta=round(c["delta"], 4),
               paired_wins=f"{c['wins']}/10", status="ok", blocked_reason="",
               note=(f"exp033 reproduced (0.538 [0.112,0.713]).{tag} Promote={'yes' if (k == sel and promote) else 'no'} "
                     f"(rule: >=8/10 wins and ci_lo>=0.112). Inorganic proxy = 5 category flags (STA is N2 only).")[:300])
print("\nresult JSONs written to", RES_DIR)
