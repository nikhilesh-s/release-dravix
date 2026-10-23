#!/usr/bin/env python3
"""C1 FSRI composite snapshot -- experiments/configs/fsri_composite_snapshot.json, followed literally (D5: pHRR and CR).

  OMP_NUM_THREADS=1 python data/v2/rescore_fsri_composite_snapshot.py --target pHRR|CR 2>&1 \
      | tee experiments/runs/fsri_composite_snapshot_<phrr|cr>.log
  python data/v2/rescore_fsri_composite_snapshot.py --target pHRR|CR --selftest-only   # label-free builder asserts, no fit

Per target: environment + clone check, layer map from material.json (+ side CSV), S0 (record config on its own data,
else BLOCKED), builder self-test (else BLOCKED), S1 (old rows, C1 groups; report-only), S2 (record config on the C1
snapshot; the number of record after a critic, whichever way it moves), report-only diagnostics.
build_phrr_c1() / build_cr_c1() are importable so later rounds reuse them verbatim.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import train_blend_round1 as B  # noqa: E402
import train_phrr_v3_robust as T  # noqa: E402
from eval_protocol import bootstrap_ci, grouped_shuffle_r2, repeated_group_kfold  # noqa: E402

CONFIG = "experiments/configs/fsri_composite_snapshot.json"
CFG = json.loads((ROOT / CONFIG).read_text())
CLONE = HERE / "raw" / "fsri_materials_database"
RUNS, RESULTS = ROOT / "experiments" / "runs", ROOT / "experiments" / "results"
LAYERS = CFG["composites"]["layers"]
NOT_MAPPED = {"Bamboo_Rug", "Overstuffed_Chair", "Overstuffed_Sofa"}
CHAIR = "Overstuffed_Chair_Assembly"
CHAIR_GROUP = [CHAIR] + LAYERS[CHAIR]
MV = CFG["rule"]["mapped_values (label-free, computed while drafting; the builder asserts each to abs diff <= 1e-4, NaN where shown)"]
STA = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
REC = {"pHRR": ("exp074", 0.6113, 0.5155, 0.8376), "CR": ("exp117", 0.6504, 0.4389, 0.7893)}
SNAP = {"pHRR": "exp158", "CR": "exp157"}  # the S2 results of this script, numbers of record since 2026-10-08
VERSIONS = {"scikit-learn": (sklearn.__version__, "1.7.1"), "numpy": (np.__version__, "2.2.6"),
            "pandas": (pd.__version__, "2.3.3"), "scipy": (scipy.__version__, "1.16.1")}


# ---------------------------------------------------------------- layer map and the one aggregation rule
def layer_map():
    """Recompute product -> layer material_dirs from every material.json; must equal the config table exactly."""
    meta = {p.parent.name: json.loads(p.read_text()) for p in (CLONE / "01_Data").glob("*/material.json")}
    name2dir = {m["material"]: m["material_dir"] for m in meta.values()}
    assert len(name2dir) == len(meta), "duplicate display names"
    comp = {k: [name2dir[c] for c in m["component material"]] for k, m in meta.items() if m.get("component material")}
    assert set(comp) == set(LAYERS) | NOT_MAPPED, sorted(comp)
    assert {k: comp[k] for k in LAYERS} == LAYERS, comp
    return comp


def write_layers_csv():
    with open(HERE / "fsri_composite_layers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["product", "layer", "layer_order"])
        n = 0
        for p, ls in LAYERS.items():
            for i, layer in enumerate(ls, 1):
                w.writerow([p, layer, i])
                n += 1
    assert n == 12


def map_layers(agg):
    """Stage 2 (rule.aggregation): per product, unweighted mean over its layers' stage-1 values that are non-NaN, per
    column. A product row is appended only if some layer has a value somewhere in this table."""
    assert not set(LAYERS) & set(agg.index), "a product is already in the table"
    rows = {p: agg.reindex(ls).mean(axis=0, skipna=True) for p, ls in LAYERS.items() if agg.reindex(ls).notna().any().any()}
    if not rows:
        return agg
    return pd.concat([agg, pd.DataFrame(rows).T.rename_axis(agg.index.name)])


def check_mapped(tables, cols):
    """rule.mapped_values within 1e-4 (NaN where shown; a product with no row in that table counts as NaN)."""
    for p in LAYERS:
        exp = dict(zip(MV["columns"], MV[p]))
        for c in cols:
            t = next(t for t in tables if c in t.columns)
            got = float(t.loc[p, c]) if p in t.index else np.nan
            e = exp[c]
            if e == "NaN":
                assert np.isnan(got), (p, c, got)
            else:
                assert abs(got - e) <= 1e-4, (p, c, got, e)


def chair_groups(mat, grp):
    return np.where(np.isin(mat, CHAIR_GROUP), CHAIR, grp)


# ---------------------------------------------------------------- builders
def build_phrr_c1(mapping=True, chair=True):
    """T.load(replicate_target=True) with the --bc extras, verbatim except mcc_agg / x_agg pass through map_layers()."""
    T.EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")
    D = T.DATA_DIR
    phrr = T.replicate_mean_target()
    mcc = pd.read_csv(f"{D}/mcc_hrr.csv")
    mcc_agg = mcc.groupby("material").agg(mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
                                          mcc_peak_norm_Wg=("rate_normalized_peak_Wg", "mean"),
                                          mcc_char_yield_pct=("char_yield_pct", "mean"))
    x = pd.read_csv(f"{D}/mcc_extras_fsri.csv")
    assert np.allclose(x["hrc_JgK"], x["peak_hrr_Wg"] / x["heating_rate_C_s"], atol=0.01), "HRC no longer redundant"
    x_agg = x.groupby("material").agg(mcc_total_hr_kJg=(T.EXTRAS["hr"], "mean"), mcc_t_peak_C=("t_peak_C", "mean"),
                                      mcc_t_onset_C=(T.EXTRAS["onset"], "mean"))
    # complete-layer entry rule (stage 1): every layer has mcc_peak_hrr_Wg and mcc_char_yield_pct
    ok = {p for p, ls in LAYERS.items() if mcc_agg.reindex(ls)[["mcc_peak_hrr_Wg", "mcc_char_yield_pct"]].notna().all().all()}
    if mapping:
        mcc_agg, x_agg = map_layers(mcc_agg), map_layers(x_agg)
    props = pd.read_csv(f"{D}/material_props_fsri.csv")
    props["category"] = props["categories"].fillna("").str.split(";").str[0].str.strip().replace("", "uncategorized")
    props = props[["material", "category", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
    k = pd.read_csv(f"{D}/thermal_conductivity_fsri.csv")
    c = pd.read_csv(f"{D}/specific_heat_fsri.csv")
    kpc = np.log(T.ambient_dry(k, "Average_Cond") * T.ambient_dry(c, "Spc_Heat")).dropna()
    thermal = kpc.rename("log_kpc").rename_axis("material").reset_index()
    df = (phrr.merge(mcc_agg.reset_index(), on="material", how="left").merge(x_agg.reset_index(), on="material", how="left")
          .merge(props, on="material", how="left").merge(thermal, on="material", how="left"))
    assert df["category"].notna().all()
    df["has_mcc"] = df[T.BASE].notna().all(axis=1)
    df = df.dropna(subset=[T.TARGET]).reset_index(drop=True)
    df = df[df["has_mcc"]].reset_index(drop=True)
    rule = ~df["material"].isin(set(LAYERS) - ok)
    assert rule.all(), "complete-layer rule changed the row set"
    df = df[rule].reset_index(drop=True)
    mat = df["material"].to_numpy()
    grp = np.where(mat == "PlasticC", "HIPS", mat)
    cat = df["category"].to_numpy(object)
    if chair:
        grp = chair_groups(mat, grp)
        ac = df.loc[df.material == CHAIR, "category"].unique()
        if len(ac):
            assert list(ac) == ["upholstered furniture"], ac
            cat = np.where(np.isin(mat, CHAIR_GROUP), ac[0], cat)
    feats = T.BASE + T.PHYS
    return dict(df=df, X=df[feats].to_numpy(float), y=df[T.TARGET].to_numpy(float), g=grp, mat=mat,
                hf=df["heat_flux_kWm2"].to_numpy(float), cat=cat, feats=feats, is_new=np.isin(mat, list(LAYERS)),
                tables=(mcc_agg, x_agg))


def first_cat(s):  # as cr_round2.json
    return str(s).split(";")[0].strip().lower() if isinstance(s, str) and s.strip() else "none"


def build_cr_c1(mapping=True, chair=True):
    """train_blend_round1.data_cr() verbatim except sta_wide / mcc_agg pass through map_layers() before their merge."""
    cone = pd.read_csv(f"{HERE}/char_residue_cone_fsri.csv")
    sta = pd.read_csv(f"{HERE}/char_yield_sta.csv")
    mcc = pd.read_csv(f"{HERE}/mcc_hrr.csv")
    props = pd.read_csv(f"{HERE}/material_props_fsri.csv")
    sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct", aggfunc="mean")
                .rename(columns=lambda r: f"sta_res_{int(r)}K"))
    mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                          mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"))
    # complete-layer entry rule (stage 1): every layer has STA at >= 1 rate AND mcc_char_yield_pct
    ok = {p for p, ls in LAYERS.items()
          if sta_wide.reindex(ls)[STA].notna().any(axis=1).all() and mcc_agg.reindex(ls)["mcc_char_yield_pct"].notna().all()}
    if mapping:
        sta_wide, mcc_agg = map_layers(sta_wide), map_layers(mcc_agg)
    tables = (sta_wide, mcc_agg)
    sta_wide, mcc_agg = sta_wide.reset_index(), mcc_agg.reset_index()
    feat = (sta_wide.merge(mcc_agg, on="material", how="outer")
            .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left"))
    df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()].merge(feat, on="material", how="left")
    df = df[df[STA].notna().any(axis=1) & df.mcc_char_yield_pct.notna()].reset_index(drop=True)
    rule = ~df["material"].isin(set(LAYERS) - ok)
    assert rule.all(), "complete-layer rule changed the row set"
    df = df[rule].reset_index(drop=True)
    mat = df.material.values
    groups = np.where(mat == "PlasticC", "HIPS", mat)
    base = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct"] + STA
    cat_l = df.categories.fillna("").str.lower()
    flags = []
    for c in ["interior finish", "exterior finish", "insulation", "roofing", "structural"]:
        col = "cat_" + c.replace(" ", "_")
        df[col] = cat_l.apply(lambda s, c=c: float(c in [t.strip() for t in s.split(";")]))
        flags.append(col)
    cat = df.categories.map(first_cat).to_numpy(object)
    if chair:
        groups = chair_groups(mat, groups)
        ac = np.unique(cat[mat == CHAIR])
        if len(ac):
            assert list(ac) == ["upholstered furniture"], ac
            cat = np.where(np.isin(mat, CHAIR_GROUP), ac[0], cat)
    feats = base + flags
    return dict(df=df, X=df[feats].values.astype(float), y=df.cone_residue_pct.values.astype(float), g=groups, mat=mat,
                hf=df.heat_flux_kWm2.to_numpy(float), cat=cat, feats=feats, is_new=np.isin(mat, list(LAYERS)), tables=tables)


# ---------------------------------------------------------------- self-test (config data.<target>.asserts)
SPEC = {
    "pHRR": dict(counts=(205, 70, 66), s1_groups=62, n_old=191,
                 pos=list(range(56, 61)) + list(range(81, 84)) + list(range(109, 112)) + list(range(150, 153)),
                 new={"Fiberglass_Insulation_R30_Paper_Faced": [25, 50, 75], "Gypsum_Wallboard": [50, 75],
                      "Lightweight_Gypsum_Wallboard": 3, CHAIR: 3, "Plastic_Laminate_Countertop": 3},
                 nan={"cone_thickness_mm": 21, "density_kgm3": 24, "log_kpc": 135},
                 cols=["mcc_peak_hrr_Wg", "mcc_char_yield_pct", "mcc_total_hr_kJg", "mcc_t_peak_C", "mcc_t_onset_C"], absent=()),
    "CR": dict(counts=(603, 66, 62), s1_groups=59, n_old=567,
               pos=list(range(180, 189)) + list(range(255, 264)) + list(range(353, 362)) + list(range(468, 477)),
               new={"Gypsum_Wallboard": 9, "Lightweight_Gypsum_Wallboard": 9, "Plastic_Laminate_Countertop": 9, CHAIR: 9},
               nan={"density_kgm3": 62, "sta_res_3K": 173, "sta_res_30K": 161},
               cols=["mcc_peak_hrr_Wg", "mcc_char_yield_pct"] + STA, absent=("Fiberglass_Insulation_R30_Paper_Faced",)),
}
BUILD = {"pHRR": build_phrr_c1, "CR": build_cr_c1}
OLD = {"pHRR": B.data_phrr, "CR": B.data_cr}


def self_test(tname):
    sp, build = SPEC[tname], BUILD[tname]
    D = OLD[tname]()
    # (a) mapping off, chair merge off -> bit-identical to the record data
    off = build(mapping=False, chair=False)
    assert np.array_equal(off["X"], D["X"], equal_nan=True) and np.array_equal(off["y"], D["y"]) \
        and np.array_equal(off["g"], D["g"]), "(a) mapping-off build differs from record data"
    # (b) mapping on: old rows identical except chair-layer groups
    S = build()
    old = ~S["is_new"]
    assert old.sum() == sp["n_old"]
    assert np.array_equal(S["X"][old], D["X"], equal_nan=True) and np.array_equal(S["y"][old], D["y"]), "(b) X/y"
    assert np.array_equal(S["mat"][old], off["mat"]) and np.array_equal(S["hf"][old], D["X"][:, 0]), "(b) material/flux"
    assert np.array_equal(S["g"][old], chair_groups(off["mat"], D["g"])), "(b) groups"
    assert len(np.unique(chair_groups(off["mat"], D["g"]))) == sp["s1_groups"], "S1 group count"
    # (c) counts, new rows, NaN counts, mapped values
    n, m, g = len(S["y"]), len(np.unique(S["mat"])), len(np.unique(S["g"]))
    assert (n, m, g) == sp["counts"], (n, m, g)
    assert np.flatnonzero(S["is_new"]).tolist() == sp["pos"], np.flatnonzero(S["is_new"]).tolist()
    for p, v in sp["new"].items():
        hf = sorted(S["hf"][S["mat"] == p].tolist())
        assert (hf == v) if isinstance(v, list) else (len(hf) == v), (p, hf)
    assert set(S["mat"][S["is_new"]]) == set(sp["new"]), set(S["mat"][S["is_new"]])
    nan = dict(zip(S["feats"], np.isnan(S["X"]).sum(axis=0).tolist()))
    assert {k: v for k, v in nan.items() if v} == sp["nan"], nan
    check_mapped(S["tables"], sp["cols"])
    for p in sp["absent"]:  # R30_PF: no sta_wide row (no layer STA), no CR rows
        assert p not in S["tables"][0].index and p not in set(S["mat"]), p
    assert len(np.unique(S["cat"])) == 11, np.unique(S["cat"])
    print(f"self-test {tname} OK: {n} rows / {m} materials / {g} groups; new rows at {sp['pos'][0]}..{sp['pos'][-1]} "
          f"({int(S['is_new'].sum())}); NaN counts {sp['nan']}; mapped values within 1e-4; 11 categories; "
          f"S1 groups {sp['s1_groups']}")
    return D, S


# ---------------------------------------------------------------- scoring + outputs
def rgkf(model, X, y, g, logt, tname):
    rk, oof = repeated_group_kfold(model, X, y, g, n_splits=5, repeats=10, log_target=logt)
    if not np.isfinite(oof).all():
        block(tname, "non-finite OOF")
    lo, hi = bootstrap_ci(y, oof[0], g, n_boot=2000, seed=0)
    return dict(rk=rk, mean=float(rk.mean()), std=float(rk.std()), lo=float(lo), hi=float(hi), oof=oof)


def result_path(tname):
    return RESULTS / f"fsri_composite_snapshot_{tname.lower()}.json"


def base_result(tname):
    exp = REC[tname][0]
    return dict(slug=f"fsri_composite_snapshot_{tname.lower()}",
                target={"pHRR": "Peak HRR (kW/m2)", "CR": "Char Residue (cone residue %, FSRI)"}[tname],
                round_config=CONFIG, code="data/v2/rescore_fsri_composite_snapshot.py",
                hyperparameters=f"{exp} config; unchanged", protocol={"pHRR": "rgkf5x10", "CR": "gkf_group5x10"}[tname],
                paired_vs="", paired_delta=0.0, paired_wins="n/a (snapshot re-score)",
                sota={"pHRR": 0.30, "CR": 0.82}[tname], sota_source="pomazi_2026", snapshot="C1")


def block(tname, reason):
    print(f"BLOCKED: {reason}")
    out = base_result(tname) | dict(status="blocked", blocked_reason=reason, note=f"C1 BLOCKED: {reason}"[:300])
    result_path(tname).write_text(json.dumps(out, indent=2))
    print(f"wrote {result_path(tname).relative_to(ROOT)}")
    sys.exit(2)


def check_env(tname):
    bad = {k: a for k, (a, b) in VERSIONS.items() if a != b}
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"python {sys.version.split()[0]} ({sys.executable}); {', '.join(f'{k} {a}' for k, (a, _) in VERSIONS.items())}; "
          f"FSRI clone {head[:7]}; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"LOKY_MAX_CPU_COUNT={os.environ.get('LOKY_MAX_CPU_COUNT')}")
    if bad or not head.startswith("a432697") or sys.version_info[:2] != (3, 12):
        block(tname, f"environment mismatch: {bad}, clone {head[:7]}, python {sys.version.split()[0]}")
    nor = json.loads((ROOT / "experiments" / "numbers_of_record.json").read_text())["targets"]
    rec = nor[{"pHRR": "pHRR", "CR": "Char Residue"}[tname]]
    if rec["exp_id"] not in (REC[tname][0], SNAP[tname]):  # SNAP: this script's own S2, promoted (numbers-of-record repro)
        block(tname, f"uncovered record config {rec['exp_id']} (incumbent_at_run_time)")
    return dict(python=sys.version.split()[0], executable=sys.executable, clone=head,
                **{k: a for k, (a, _) in VERSIONS.items()}, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                LOKY_MAX_CPU_COUNT=os.environ.get("LOKY_MAX_CPU_COUNT"))


def fmt(r):
    return f"{r['mean']:.4f} +/- {r['std']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["pHRR", "CR"])
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    tname, t0 = a.target, time.time()
    print(f"Config: {CONFIG} (status: {CFG['status']}); target {tname}")
    env = check_env(tname)
    try:
        layer_map()
    except AssertionError as e:
        block(tname, f"layer map differs from composites.layers: {e}")
    write_layers_csv()
    print(f"layer map == composites.layers (5 products, 12 layers); wrote data/v2/fsri_composite_layers.csv")
    if a.selftest_only:
        try:
            self_test(tname)
        except AssertionError as e:
            block(tname, f"self-test: {e}")
        print(f"selftest-only: no model fit; {time.time() - t0:.0f}s")
        return

    exp, m0, l0, h0 = REC[tname]
    logt = tname == "pHRR"
    D0 = OLD[tname]()
    model = T.hgb() if tname == "pHRR" else B.Avg([D0["inc"], B.et(), B.ridge()])
    print(f"model: {model}")

    # S0: reproduce the record on its own data
    t = time.time()
    s0 = rgkf(model, D0["X"], D0["y"], D0["g"], logt, tname)
    print(f"S0 {exp} on its own data ({len(D0['y'])} rows / {len(np.unique(D0['g']))} groups): {fmt(s0)}  "
          f"(record {m0} [{l0}, {h0}]; {time.time() - t:.0f}s)", flush=True)
    if not (abs(s0["mean"] - m0) <= 0.002 and abs(s0["lo"] - l0) <= 0.002 and abs(s0["hi"] - h0) <= 0.002):
        block(tname, f"step 1: {exp} not reproduced within 0.002 ({s0['mean']:.4f} [{s0['lo']:.4f}, {s0['hi']:.4f}])")
    print(f"REPRO {exp} ok")

    try:
        D, S = self_test(tname)
    except AssertionError as e:
        block(tname, f"self-test: {e}")

    # S1: old rows, old features, C1 groups (report-only)
    old = ~S["is_new"]
    g1 = S["g"][old]
    t = time.time()
    s1 = rgkf(model, D["X"], D["y"], g1, logt, tname)
    print(f"S1 merge-only (old rows, C1 groups: {len(D['y'])} rows / {len(np.unique(g1))} groups; report-only): {fmt(s1)} "
          f"({time.time() - t:.0f}s)", flush=True)

    # S2: the record config on the C1 snapshot
    X, y, g = S["X"], S["y"], S["g"]
    t = time.time()
    s2 = rgkf(model, X, y, g, logt, tname)
    print(f"S2 snapshot ({len(y)} rows / {len(np.unique(S['mat']))} materials / {len(np.unique(g))} groups): {fmt(s2)} "
          f"({time.time() - t:.0f}s)", flush=True)
    print(f"  per-repeat R2: {np.round(s2['rk'], 4).tolist()}")

    # reported_only
    plo, phi = bootstrap_ci(y, s2["oof"].mean(axis=0), g, n_boot=2000, seed=0)
    gss = float(grouped_shuffle_r2(model, X, y, g, log_target=True).mean()) if logt else None
    t = time.time()
    crk, coof = repeated_group_kfold(model, X, y, S["cat"], n_splits=5, repeats=10, log_target=logt)
    assert np.isfinite(coof).all()
    clo, chi = bootstrap_ci(y, coof[0], g, n_boot=2000, seed=0)
    print(f"  pooled-repeat CI [{plo:.4f}, {phi:.4f}]" + (f"; GroupShuffleSplit x10 mean {gss:.4f}" if logt else "")
          + f"; category-GKF ({len(np.unique(S['cat']))} categories) {crk.mean():.4f} +/- {crk.std():.4f} "
          f"CI [{clo:.4f}, {chi:.4f}] ({time.time() - t:.0f}s)")
    new = S["is_new"]
    P = s2["oof"]
    sse = ((P - y) ** 2)
    dec = dict(label="not paired; folds differ",
               S0=dict(rkf_mean=s0["mean"], ci_lo=s0["lo"], ci_hi=s0["hi"], n_rows=len(D0["y"]), n_groups=len(np.unique(D0["g"]))),
               S1=dict(rkf_mean=s1["mean"], ci_lo=s1["lo"], ci_hi=s1["hi"], n_rows=len(D["y"]), n_groups=len(np.unique(g1))),
               S2=dict(rkf_mean=s2["mean"], ci_lo=s2["lo"], ci_hi=s2["hi"], n_rows=len(y), n_groups=len(np.unique(g))),
               S2_old_rows_r2=float(np.mean([r2_score(y[old], p[old]) for p in P])),
               S2_new_rows_r2=float(np.mean([r2_score(y[new], p[new]) for p in P])),
               S2_new_rows_sse_share=float(sse[:, new].sum() / sse.sum()))
    print(f"decomposition (not paired; folds differ): S0 {s0['mean']:.4f} [{s0['lo']:.4f}, {s0['hi']:.4f}] | "
          f"S1 {s1['mean']:.4f} [{s1['lo']:.4f}, {s1['hi']:.4f}] | S2 {s2['mean']:.4f} [{s2['lo']:.4f}, {s2['hi']:.4f}]; "
          f"S2 old rows R2 {dec['S2_old_rows_r2']:.4f} ({int(old.sum())}), new rows R2 {dec['S2_new_rows_r2']:.4f} "
          f"({int(new.sum())}), new rows' SSE share {dec['S2_new_rows_sse_share']:.4f}")
    table = {}
    for p in [q for q in LAYERS if q in set(S["mat"])] + LAYERS[CHAIR]:
        r = S["mat"] == p
        table[p] = dict(n=int(r.sum()), mean_y=float(y[r].mean()), mean_oof=float(P[:, r].mean()),
                        mae=float(np.abs(P[:, r] - y[r]).mean()), sse_share=float(sse[:, r].sum() / sse.sum()))
        print(f"  {p:42s} n={table[p]['n']:2d} mean y {table[p]['mean_y']:8.2f}  mean OOF {table[p]['mean_oof']:8.2f}  "
              f"MAE {table[p]['mae']:8.2f}  SSE share {table[p]['sse_share']:.4f}")
    diag = dict(config=CONFIG, environment=env, precondition="round 9, cr_round2 and round 10 pHRR/CR closed with no "
                "adoption; numbers_of_record still exp074 / exp117 (checked by the script)",
                S0=dict(dec["S0"], per_repeat_r2=s0["rk"].tolist(), reproduced=True, record=[m0, l0, h0]),
                S1=dict(dec["S1"], rkf_std=s1["std"], per_repeat_r2=s1["rk"].tolist()),
                reported_only=dict(pooled_ci=[float(plo), float(phi)], gss_mean=gss,
                                   category_gkf=dict(mean=float(crk.mean()), std=float(crk.std()), ci_lo=float(clo),
                                                     ci_hi=float(chi), n_categories=int(len(np.unique(S["cat"]))),
                                                     per_repeat_r2=crk.tolist()),
                                   decomposition=dec, per_composite_table=table))
    if logt:
        r = pd.read_csv(HERE / "cone_replicates_fsri.csv")
        r = r[~r["did_not_ignite"] & ~r["early_stop"] & r["material"].isin(list(LAYERS))]
        tr = r.groupby("material")["truncated_test"].sum().astype(int).to_dict()
        diag["reported_only"]["pHRR_truncation_note"] = dict(n_replicates=int(len(r)), n_truncated=int(r["truncated_test"].sum()),
                                                             by_product=tr)
        print(f"  truncation note: {int(r['truncated_test'].sum())} of {len(r)} composite replicates behind the new rows "
              f"carry truncated_test: {tr}")
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"fsri_composite_snapshot_{tname.lower()}_diag.json").write_text(json.dumps(diag, indent=2))
    np.savez_compressed(RUNS / f"fsri_composite_snapshot_{tname.lower()}_oof.npz", y=y, groups=np.asarray(g, dtype=str),
                        material=np.asarray(S["mat"], dtype=str), heat_flux=S["hf"], is_new=new, row_id=np.arange(len(y)),
                        category=np.asarray(S["cat"], dtype=str), oof_S2=P, oof_catS2=coof, oof_S1=s1["oof"], oof_S0=s0["oof"])

    n_mat = len(np.unique(S["mat"]))
    k = {"pHRR": 5, "CR": 4}[tname]
    trig = (s2["lo"] <= 0.30) if logt else (s2["lo"] > 0.82)
    note = (f"C1 snapshot re-score, different population (not comparable to {exp} {m0}); {exp} config unchanged; "
            f"unweighted layer-mean MCC/STA for {k} composites; chair+3 layers one group; S1 {s1['mean']:.4f}; "
            f"category-GKF {crk.mean():.3f}" + ("; EXIT trigger fired" if trig else ""))
    out = base_result(tname) | dict(
        dataset=(f"C1 snapshot (FSRI composite layer mapping, {CONFIG}): {len(y)} rows / {n_mat} materials / "
                 f"{len(np.unique(g))} groups; unweighted layer-mean MCC/STA for {k} composite products; chair assembly + "
                 f"3 layers one group; PlasticC -> HIPS"),
        features=S["feats"], model={"pHRR": "HistGradientBoostingRegressor (log1p target)",
                                    "CR": "Avg(HGB, ExtraTrees, Ridge)"}[tname],
        n_rows=int(len(y)), n_groups=int(len(np.unique(g))), n_materials=int(n_mat),
        rkf_mean=round(s2["mean"], 4), rkf_std=round(s2["std"], 4), ci_lo=round(s2["lo"], 4), ci_hi=round(s2["hi"], 4),
        status="ok", blocked_reason="", note=note[:300],
        pooled_ci_lo=round(float(plo), 4), pooled_ci_hi=round(float(phi), 4),
        category_grouped_mean=round(float(crk.mean()), 4), category_grouped_ci_lo=round(float(clo), 4),
        category_grouped_ci_hi=round(float(chi), 4), rkf_mean_full=s2["mean"], ci_lo_full=s2["lo"], ci_hi_full=s2["hi"],
        per_repeat_r2=[float(v) for v in s2["rk"]])
    if logt:
        out["gss_mean"] = round(gss, 4)
    assert len(note) <= 300
    result_path(tname).write_text(json.dumps(out, indent=2))
    print(f"wrote {result_path(tname).relative_to(ROOT)}, runs/fsri_composite_snapshot_{tname.lower()}_diag.json, _oof.npz")
    print("exit trigger: " + ({"pHRR": "S2 ci_lo <= 0.30 -> EXIT with blocker", "CR": "S2 ci_lo > 0.82 -> EXIT"}[tname]
                              + (" FIRED" if trig else " not fired")))
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
