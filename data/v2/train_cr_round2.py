#!/usr/bin/env python3
"""CR round 2 (B1) -- experiments/configs/cr_round2.json, followed literally (approved D2, defaults).

  OMP_NUM_THREADS=1 python data/v2/train_cr_round2.py 2>&1 | tee experiments/runs/cr_round2.log
  python data/v2/train_cr_round2.py --assemble-gate      # after round 9's CR members (train_round9_tabpfn.py --target CR)

Step 1 reproduces exp117 (else BLOCKED), the n1 harness self-test checks the custom fold loop, then o1, o2 (report-only),
n1. Adoption: one adoption_gate_v2 run over the joint CR family [o1, o2, n1, p1, p2] (K = 5), assembled by --assemble-gate.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS, RESULTS = ROOT / "experiments" / "runs", ROOT / "experiments" / "results"
sys.path.insert(0, str(HERE))
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402
from train_blend_round1 import Avg, data_cr, et, ridge  # noqa: E402

CONFIG = "experiments/configs/cr_round2.json"
REC = (0.6504, 0.4389, 0.7893)
STA = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
CHAIR = "Overstuffed_Chair_Assembly"
LINKED = {CHAIR: {"Overstuffed_Chair_Polyester_Batting", "Overstuffed_Chair_Polyester_Fabric", "Overstuffed_Chair_Polyurethane_Foam"}}
EXTRA_COUNTS = {"Engineered_Wood_Table": 11, "Fiberglass_Insulation_R13_Paper_Faced": 9, "Fiberglass_Insulation_R30": 9,
                "Fiberglass_Insulation_R30_Paper_Faced": 9, "Fiberglass_Skylight": 9, "Gypsum_Wallboard": 9,
                "Lightweight_Gypsum_Wallboard": 9, CHAIR: 9, "PEVA": 9, "PMMA": 8, "Pallet_Wood": 3,
                "Plastic_Laminate_Countertop": 9, "Polyisocyanurate_Spray_Foam_Insulation": 6, "Roof_Felt": 9, "Rubber_Band": 5,
                "Vinyl_Plank_Flooring": 9, "XPS": 4}
VERSIONS = {"scikit-learn": (sklearn.__version__, "1.7.1"), "numpy": (np.__version__, "2.2.6"),
            "pandas": (pd.__version__, "2.3.3"), "scipy": (scipy.__version__, "1.16.1")}
HGB = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)


def base(X):
    """Offset base: row-wise nanmean of sta_res_3K, sta_res_10K, sta_res_30K (columns 3, 4, 5); label-free."""
    return np.nanmean(X[:, [3, 4, 5]], axis=1)


class Offset(BaseEstimator, RegressorMixin):
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        b = base(X)
        if not np.isfinite(b).all():
            print("BLOCKED: NaN offset base")
            sys.exit(2)
        self.m_ = clone(self.est).fit(X, y - b)
        return self

    def predict(self, X):
        return self.m_.predict(X) + base(X)


class Clip0_100(BaseEstimator, RegressorMixin):
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.m_ = clone(self.est).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.m_.predict(X), 0.0, 100.0)


def hgb():
    return HistGradientBoostingRegressor(**HGB)


def first_cat(s):
    return str(s).split(";")[0].strip().lower() if isinstance(s, str) and s.strip() else "none"


def build_703():
    """feature_build_703: direct left joins on material; 0-based CSV row_id; cat_* exactly as data_cr."""
    cone = pd.read_csv(HERE / "char_residue_cone_fsri.csv")
    cone["row_id"] = np.arange(len(cone))
    sta = pd.read_csv(HERE / "char_yield_sta.csv")
    mcc = pd.read_csv(HERE / "mcc_hrr.csv")
    props = pd.read_csv(HERE / "material_props_fsri.csv")
    sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct", aggfunc="mean")
                .rename(columns=lambda r: f"sta_res_{int(r)}K").reset_index())
    mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                          mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean")).reset_index()
    df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()]
    n0 = len(df)
    df = (df.merge(sta_wide, on="material", how="left").merge(mcc_agg, on="material", how="left")
          .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left").reset_index(drop=True))
    assert len(df) == n0 == 703 and df.material.nunique() == 79, (n0, len(df), df.material.nunique())
    cat_l = df.categories.fillna("").str.lower()
    flags = []
    for c in ["interior finish", "exterior finish", "insulation", "roofing", "structural"]:
        col = "cat_" + c.replace(" ", "_")
        df[col] = cat_l.apply(lambda s, c=c: float(c in [t.strip() for t in s.split(";")]))
        flags.append(col)
    feats = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct"] + STA + flags
    groups = np.where(df.material.values == "PlasticC", "HIPS", df.material.values)
    score = (df[STA].notna().any(axis=1) & df.mcc_char_yield_pct.notna()).to_numpy()
    return dict(X=df[feats].to_numpy(float), y=df.cone_residue_pct.to_numpy(float), g=groups, score=score, feats=feats,
                row_id=df.row_id.to_numpy(), cat=df.categories.map(first_cat).to_numpy(), mat=df.material.to_numpy())


def check_links(scoring_groups, extra_mats):
    """Label-free: material.json parent/component keys (raw clone a432697) give exactly the declared link table."""
    d01 = HERE / "raw" / "fsri_materials_database" / "01_Data"
    meta = {p.parent.name: json.loads(p.read_text()) for p in d01.glob("*/material.json")}
    children = {}
    for name, m in meta.items():
        par = (m.get("parent material") or "").replace(" ", "_")
        if par:
            children.setdefault(par, set()).add(name)
    derived = {}
    for comp in extra_mats:
        if meta.get(comp, {}).get("component material"):
            derived[comp] = children.get(comp, set()) & set(scoring_groups)
    linked = {k: v for k, v in derived.items() if v}
    assert linked == LINKED, (derived, LINKED)
    print(f"link table check: composites among extra materials {sorted(derived)}; scoring-group links {linked}")


def n1_loop(model, X5, y5, g5, Xe, ye, ge, split_groups, cat5=None, cate=None):
    """n1 fold loop: extra rows train-only; chair assembly removed when a linked layer is in the test fold;
    under the category protocol, extra rows whose first category is a test category are removed too."""
    oof, nlink = np.empty((10, len(y5))), 0
    for r in range(10):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X5, y5, split_groups):
            assert set(split_groups[tr]).isdisjoint(split_groups[te])
            keep = np.ones(len(ye), bool)
            if set(g5[te]) & LINKED[CHAIR]:
                keep &= ge != CHAIR
                nlink += int((ge == CHAIR).any())
            if cat5 is not None:
                keep &= ~np.isin(cate, list(set(cat5[te])))
            m = clone(model).fit(np.vstack([X5[tr], Xe[keep]]), np.concatenate([y5[tr], ye[keep]]))
            oof[r, te] = m.predict(X5[te])
    return oof, nlink


def topk(y, g, oof_inc, oof_c):
    """reported_only.top_k_removal. Diagnostic only; excluding data is banned."""
    row = ((y - oof_inc) ** 2 - (y - oof_c) ** 2).mean(axis=0)
    con = {str(m): float(row[g == m].sum()) for m in np.unique(g)}
    ranked = sorted(con, key=lambda m: (-con[m], m))
    tot = sum(con.values())
    out = dict(top5=[[m, con[m]] for m in ranked[:5]], top5_share_of_net=sum(con[m] for m in ranked[:5]) / tot if tot else None)
    for k in (1, 3, 5):
        keep = ~np.isin(g.astype(str), ranked[:k])
        d = np.array([r2_score(y[keep], oof_c[r, keep]) - r2_score(y[keep], oof_inc[r, keep]) for r in range(10)])
        out[f"drop_top{k}"] = dict(mean_gain=float(d.mean()), wins=int((d > 0).sum()))
    out["not_established"] = bool(out["drop_top5"]["mean_gain"] <= 0 or out["drop_top5"]["wins"] < 8)
    return out


def mineral_wool(y, g, oof):
    m = g == "Mineral_Wool_Insulation"
    sse = (y - oof) ** 2
    return dict(mean_pred=float(oof[:, m].mean()), sse_share=float(sse[:, m].sum() / sse.sum()))


def blocked_all(reason):
    for cid in ("o1", "o2", "n1"):
        (RESULTS / f"cr_r2_{cid}.json").write_text(json.dumps(dict(slug=f"cr_r2_{cid}", round_config=CONFIG, status="blocked",
                                                                   blocked_reason=reason), indent=2))
    print(f"BLOCKED: {reason}")
    sys.exit(2)


def main():
    t0 = time.time()
    print(f"python {sys.version.split()[0]} ({sys.executable}); " + ", ".join(f"{k} {v[0]}" for k, v in VERSIONS.items())
          + f"; OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}; LOKY_MAX_CPU_COUNT {os.environ.get('LOKY_MAX_CPU_COUNT', 'unset')}")
    bad = [k for k, (have, want) in VERSIONS.items() if have != want]
    if bad:
        blocked_all(f"library version mismatch: {bad}")
    D = data_cr()
    X, y, g = D["X"], D["y"], D["g"]
    assert D["feats"][3:6] == STA and D["inc"].get_params() == hgb().get_params()
    assert (len(y), len(np.unique(g))) == (567, 61)
    inc_model = Avg([hgb(), et(), ridge()])

    # ---- step 1: exp117 ----
    t = time.time()
    rk_inc, oof_inc = repeated_group_kfold(inc_model, X, y, g)
    lo, hi = bootstrap_ci(y, oof_inc[0], g)
    print(f"exp117 GKF {rk_inc.mean():.4f} +- {rk_inc.std():.4f} CI [{lo:.4f}, {hi:.4f}]  ({time.time() - t:.0f}s)", flush=True)
    if not all(abs(a - b) <= 0.002 for a, b in zip((rk_inc.mean(), lo, hi), REC)):
        blocked_all(f"exp117 not reproduced within 0.002 (mean, ci_lo, ci_hi = {rk_inc.mean():.4f}, {lo:.4f}, {hi:.4f})")
    print(f"REPRO exp117 ok (record {REC})")
    step1 = dict(rkf_mean_full=float(rk_inc.mean()), rkf_std_full=float(rk_inc.std()), ci_lo_full=float(lo), ci_hi_full=float(hi),
                 per_repeat_r2=[float(v) for v in rk_inc], n_rows=567, n_groups=61, config=CONFIG, code="data/v2/train_cr_round2.py")
    (RUNS / "cr_round2_step1_exp117.json").write_text(json.dumps(step1, indent=2))

    # ---- 703-row build and asserts ----
    B = build_703()
    sc = B["score"]
    assert sc.sum() == 567 and np.array_equal(B["X"][sc], X, equal_nan=True) and np.array_equal(B["y"][sc], y) \
        and np.array_equal(B["g"][sc], g), "567-row subset differs from data_cr()"
    ex = ~sc
    Xe, ye, ge, cate = B["X"][ex], B["y"][ex], B["g"][ex], B["cat"][ex]
    counts = pd.Series(B["mat"][ex]).value_counts().to_dict()
    assert counts == EXTRA_COUNTS, counts
    assert ex.sum() == 136 and len(np.unique(ge)) == 17 and len(np.unique(B["g"])) == 78 and not set(ge) & set(g)
    if not np.isfinite(base(X)).all():
        blocked_all("NaN offset base on a scoring row")
    cat5 = B["cat"][sc]
    assert len(np.unique(cat5)) == 11, np.unique(cat5)
    check_links(np.unique(g), np.unique(ge))
    print(f"703 build OK: 567 scoring rows identical to data_cr(); extra rows {ex.sum()} / {len(np.unique(ge))} materials")

    # ---- n1 harness self-test ----
    oof_st, _ = n1_loop(inc_model, X, y, g, Xe[:0], ye[:0], ge[:0], g)
    rk_st = np.array([r2_score(y, p) for p in oof_st])
    print(f"n1 harness self-test: max |dR2| vs step 1 = {np.abs(rk_st - rk_inc).max():.2e}")
    if not np.all(np.abs(rk_st - rk_inc) <= 1e-12):
        blocked_all("n1 harness self-test failed (custom loop folds differ from eval_protocol)")

    # ---- candidates ----
    o1_unc = Avg([Offset(hgb()), Offset(et()), ridge()])
    o2_unc = Offset(hgb())
    res, diag = {}, {}
    for cid, unc in (("o1", o1_unc), ("o2", o2_unc)):
        t = time.time()
        rk, oof = repeated_group_kfold(Clip0_100(unc), X, y, g)
        _, oof_u = repeated_group_kfold(unc, X, y, g)
        assert np.allclose(np.clip(oof_u, 0, 100), oof, rtol=0, atol=1e-12), "clip diagnostic does not match the candidate"
        res[cid], diag[cid] = dict(rk=rk, oof=oof, wall=time.time() - t), oof_u
    t = time.time()
    oof_n1, nlink = n1_loop(inc_model, X, y, g, Xe, ye, ge, g)
    res["n1"] = dict(rk=np.array([r2_score(y, p) for p in oof_n1]), oof=oof_n1, wall=time.time() - t)

    # ---- report-only: category GKF ----
    cat_mean = {"exp117": float(repeated_group_kfold(inc_model, X, y, cat5)[0].mean()),
                "o1": float(repeated_group_kfold(Clip0_100(o1_unc), X, y, cat5)[0].mean()),
                "o2": float(repeated_group_kfold(Clip0_100(o2_unc), X, y, cat5)[0].mean())}
    oof_n1c, nlink_c = n1_loop(inc_model, X, y, g, Xe, ye, ge, cat5, cat5=cat5, cate=cate)
    cat_mean["n1"] = float(np.mean([r2_score(y, p) for p in oof_n1c]))

    mw = {"exp117": mineral_wool(y, g, oof_inc)}
    for cid in ("o1", "o2", "n1"):
        r = res[cid]
        r["lo"], r["hi"] = (float(v) for v in bootstrap_ci(y, r["oof"][0], g))
        mw[cid] = mineral_wool(y, g, r["oof"])
        if cid in diag:
            mw[cid + "_unclipped"] = mineral_wool(y, g, diag[cid])
        d = r["rk"] - rk_inc
        print(f"{cid}   GKF {r['rk'].mean():.4f} +- {r['rk'].std():.4f} CI [{r['lo']:.4f}, {r['hi']:.4f}]  d {d.mean():+.4f} "
              f"wins {(d > 0).sum()}/10  category-GKF {cat_mean[cid]:.4f}  ({r['wall']:.0f}s)", flush=True)
    print(f"exp117 category-GKF {cat_mean['exp117']:.4f}; n1 chair-link removals: {nlink}/50 primary folds, {nlink_c}/50 category folds")
    clip = {cid: dict(share_above_100=float((diag[cid] > 100).mean()), share_below_0=float((diag[cid] < 0).mean()),
                      unclipped_rkf_mean=float(np.mean([r2_score(y, p) for p in diag[cid]]))) for cid in diag}
    print("clip effect:", json.dumps(clip), "\nmineral wool:", json.dumps(mw))
    print("Triggers:", [c for c in res if res[c]["lo"] > 0.82] or "none")

    DESC = {"o1": "Clip0_100(Avg([Offset(HGB exp081 params), Offset(ExtraTrees 500/leaf 2), Ridge pipeline]))",
            "o2": "Clip0_100(Offset(HGB exp081 params)) alone",
            "n1": "exp117 recipe verbatim trained per fold on the 567-row training part plus the 136 extra spark rows; scored on the 567 rows"}
    HP = {"o1": "offset base = row nanmean of sta_res_3K/10K/30K, coefficient 1, HGB and ET only; Ridge unchanged on raw y; final clip [0, 100]",
          "o2": "offset base = row nanmean of sta_res_3K/10K/30K, coefficient 1; final clip [0, 100]",
          "n1": "exp117 members unchanged (HGB exp081 params, ET 500/leaf 2 NaN-native, Ridge median impute + indicators); chair assembly group-linked to its 3 layers"}
    for cid in ("o1", "o2", "n1"):
        r = res[cid]
        d = r["rk"] - rk_inc
        tk = topk(y, g, oof_inc, r["oof"])
        note = ("REPORT-ONLY, never adopted; " if cid == "o2" else "") + f"CR round 2 {cid}; paired vs exp117 on identical folds; " \
            f"category-GKF {cat_mean[cid]:.3f} (exp117 {cat_mean['exp117']:.3f})"
        if tk["not_established"]:
            note += "; gain concentrated in <= 5 materials, not established"
        out = dict(slug=f"cr_r2_{cid}", target="Char Residue (cone residue %, FSRI)", round_config=CONFIG,
                   dataset="char_residue_cone_fsri x STA/MCC/props (567 rows / 61 merged groups), exp117 rows, order and groups"
                           + ("; training adds the 136 extra spark rows (17 materials), never scored" if cid == "n1" else ""),
                   features=D["feats"], model=DESC[cid], hyperparameters=HP[cid] + "; fixed a priori", code="data/v2/train_cr_round2.py",
                   n_rows=567, n_groups=61, protocol="gkf_group5x10", rkf_mean=round(float(r["rk"].mean()), 4),
                   rkf_std=round(float(r["rk"].std()), 4), ci_lo=round(r["lo"], 4), ci_hi=round(r["hi"], 4), paired_vs="exp117",
                   paired_delta=round(float(d.mean()), 4), paired_wins=f"{int((d > 0).sum())}/10", sota=0.82, sota_source="pomazi_2026",
                   status="ok", blocked_reason="", note=note, rkf_mean_full=float(r["rk"].mean()), ci_lo_full=r["lo"], ci_hi_full=r["hi"],
                   per_repeat_r2=[float(v) for v in r["rk"]], top_k_removal=tk, category_gkf_mean=cat_mean[cid],
                   mineral_wool={k: v for k, v in mw.items() if k.startswith(cid)}, wall_time_s=round(r["wall"], 1))
        if cid in clip:
            out["clip_effect"] = clip[cid]
        if cid == "n1":
            out.update(train_rows=703, extra_rows=136, n1_link_count=dict(primary=nlink, category=nlink_c))
        (RESULTS / f"cr_r2_{cid}.json").write_text(json.dumps(out, indent=2))
        print(f"wrote experiments/results/cr_r2_{cid}.json  top-k: {json.dumps({k: tk[k] for k in ('drop_top1', 'drop_top3', 'drop_top5')})}")
    np.savez(RUNS / "cr_round2_oof.npz", y=y, groups=np.asarray(g).astype(str), row_id=B["row_id"][sc], oof_incumbent=oof_inc,
             oof_o1=res["o1"]["oof"], oof_o2=res["o2"]["oof"], oof_n1=res["n1"]["oof"],
             diag_o1_unclipped=diag["o1"], diag_o2_unclipped=diag["o2"])
    rep = dict(exp117=dict(category_gkf_mean=cat_mean["exp117"], mineral_wool=mw["exp117"]))
    (RUNS / "cr_round2_report.json").write_text(json.dumps(rep, indent=2))
    print(f"wrote experiments/runs/cr_round2_oof.npz, cr_round2_step1_exp117.json, cr_round2_report.json; runtime {time.time() - t0:.0f}s")


def assemble_gate():
    """outputs.oof: joint round-9 CR family [o1, o2, n1, p1, p2] (K = 5), one gate run."""
    import adoption_gate
    A = np.load(RUNS / "cr_round2_oof.npz")
    Bz = np.load(RUNS / "round9_cr_b2_oof.npz")
    s1 = json.loads((RUNS / "cr_round2_step1_exp117.json").read_text())
    ok = np.array_equal(A["y"], Bz["y"]) and np.array_equal(A["groups"], Bz["groups"])
    dmax = float(np.abs(A["oof_incumbent"] - Bz["oof_incumbent"]).max())
    print(f"B2 y/groups identical: {ok}; max |B2 step-1 exp117 OOF - cr_round2 OOF| = {dmax:.2e}")
    if not ok or dmax > 1e-12:
        print("BLOCKED: B2 CR arrays not aligned with cr_round2 (no CR decision)")
        sys.exit(2)
    order, sel = ["o1", "o2", "n1", "p1", "p2"], ["o1", "n1", "p1", "p2"]
    npz = RUNS / "round9_cr_oof.npz"
    np.savez(npz, y=A["y"], groups=A["groups"], row_id=A["row_id"], oof_incumbent=A["oof_incumbent"],
             inc_per_repeat_r2=np.array(s1["per_repeat_r2"]), inc_ci=np.array([s1["ci_lo_full"], s1["ci_hi_full"]]),
             order=np.array(order), selectable=np.array(sel), oof_o1=A["oof_o1"], oof_o2=A["oof_o2"], oof_n1=A["oof_n1"],
             oof_p1=Bz["oof_p1"], oof_p2=Bz["oof_p2"])
    paths = {c: str(RESULTS / f"cr_r2_{c}.json") for c in ("o1", "o2", "n1")}
    paths.update({c: str(RESULTS / f"r9_tabpfn_cr_{c}.json") for c in ("p1", "p2")})
    gate = adoption_gate.run_npz(str(npz), paths, str(RUNS / "round9_cr_gate_v2.json"))
    for c, p in paths.items():
        o = json.loads(Path(p).read_text())
        o["gate_v2"] = dict(gate["per_candidate"][c], K=gate["K"], q=gate["q"], selected=(c == gate["best"]))
        if c == gate["best"]:
            o["note"] += (f"; gate v2 selected (K={gate['K']}, q={gate['q']:.3f}, LB={gate['per_candidate'][c]['lb_maxT']:+.4f}): "
                          f"G1 {gate['G1']}, G2 {gate['G2']}, G3 {gate['G3']}")
        Path(p).write_text(json.dumps(o, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble-gate", action="store_true")
    if ap.parse_args().assemble_gate:
        assemble_gate()
    else:
        main()
