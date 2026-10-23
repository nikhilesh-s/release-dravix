#!/usr/bin/env python3
"""Dravix V2 -- pHRR: how robust is "beats SOTA", and can it be made robust? (sprint 2026-09-29)

WHY: the promoted pHRR model's "beats SOTA" claim (exp006, R2=0.5744 vs Pomazi's real held-out
0.30) rests on ONE GroupShuffleSplit (seed 42, 15 test materials). The same config's 10-seed mean
was 0.346 with std 0.198 (exp002), so a single 15-material split can land almost anywhere. This
script:

1. Re-scores the promoted config (models/v2/phrr_rf_v1.joblib's recipe) with
   eval_protocol.report: repeated 5-fold GroupKFold x10 with pooled OOF R2 and a material-bootstrap
   CI. It also re-runs exp006's seed 42 to check nothing drifted, and shows where seed 42 falls
   among 200 GroupShuffleSplit seeds.
2. Repeats the repeated GroupKFold grouped by FSRI material CATEGORY. That's the family-grouped
   split V2_TARGET_DEFINITION.md calls ideal. Category = the FIRST category listed in material.json
   (via material_props_fsri.csv). For every multi-category material in this set, FSRI lists the
   composition-type category (general polymer / engineered wood) before the use-type one
   (interior finish / upholstered furniture), so "first listed" works as "composition first".
   Materials with no category form one "uncategorized" group.
3. Scores a small, PRE-DECLARED set of variants on the SAME 175 rows and the SAME splits, and
   picks the winner by repeated-GroupKFold mean alone:
     V0 rf_base      promoted config, the reference
     V1 rf_base_log  V0 trained on log1p(pHRR), scored in kW/m2
     V2 rf_phys      V0 + PHYS features (RF handles NaN natively in sklearn>=1.4)
     V3 rf_phys_log  V2 trained on log1p(pHRR)
     V4 hgb_phys     HistGradientBoostingRegressor on V2's features, same rows
   Also V5 hgb_phys_nojoin: V4 on every pHRR row, with no MCC inner join (220 rows / 76
   materials). Its row set differs from the others, so it isn't a candidate for "winner". Instead
   it's compared with V0 on one shared 76-material partition, scored on the rows both can predict.
   PHYS = cone fuel load (mass per area), cone thickness, density (material_props_fsri.csv); MCC
   total heat release, T_peak, T_onset (mcc_extras_fsri.csv, per-material means); thermal inertia
   log(k*rho*c) (HFM, Dry, lowest setpoint, same recipe as train_tti_v4_thermal.py).
   Heat release capacity is left out on purpose: it's peak HRR divided by the heating rate,
   so at MCC's fixed rate it just duplicates mcc_peak_hrr_Wg (asserted below).
   All hyperparameters are fixed a priori. None were tuned.
   Promotion bar, declared before running: the winner must beat V0 in >= 8/10 repeats, AND the
   material-bootstrap 95% CI of its mean paired delta-R2 must exclude 0.
   After the winner is picked, a drop-one-feature-group run explains where its gain comes from.
   That run is labelled post-hoc and isn't used to choose anything.
4. Cross-lab check of the winner on MaCFP's 10-lab PMMA set, the same way
   validate_phrr_crosslab.py does it (FSRI "PMMA" material-level features on every row). It's also
   run with FSRI "Black_PMMA", the closer product, and with MaCFP's own nominal 6 mm specimen
   geometry, because the geometry features describe the specimen as well as the material.
5. --save PATH retrains the rule-picked winner on all its rows and writes a joblib bundle in
   the same format as models/v2/phrr_rf_v1.joblib. predict() returns kW/m2 even for log-target
   winners. Only the lead should point PATH at models/v2.

6. --replicate-check runs only a target-sensitivity check, added after the FIGRA stream found that
   FSRI's averaged HRRPUA curves end early and smear peaks. It rescores V0 and V3 on per-replicate
   pHRR from figra_fsri.csv.

7. --no-extras-variant (added 2026-09-29, reports/sprint_0929/mcc_fix.md) also scores "V3 minus the MCC
   extras" (mcc_total_hr_kJg, mcc_t_peak_C, mcc_t_onset_C) on the same rows and splits, paired against V3.
   It was pre-declared there as a secondary hypothesis; it is never a candidate for the V0-V4 rule winner.

8. --replicate-target (added 2026-09-29 round 3, reports/sprint_0929/replicate_targets.md) swaps the target for
   the canonical per-replicate one: per material x flux, the mean phrr_kWm2 in cone_replicates_fsri.csv over
   replicates that are neither did_not_ignite nor early_stop. Rows = those pairs with MCC features (same rule as
   the default). It runs ONLY the pre-declared set: A1 = V3 rf_phys_log (must be run with --bc), A2 = A1 with
   the MCC peak rate-normalised (material mean of rate_normalized_peak_Wg), and V0 as the paired reference.
   Everything else (V1/V2/V4/V5, seed 42, error shares, cross-lab) is skipped to save time.

Row sets: the MCC ingest fix of 2026-09-29 (mcc_fix.md) recovered MCC files for 9 materials, 7 of which have
pHRR rows, so the inner join grew from 175 rows / 60 materials (every result up to exp036, branch commit
cb53979) to 196 / 67. DATA_VERSIONS below accepts either; exp006's seed-42 value is only asserted on the old one.

Rerun: python3 data/v2/train_phrr_v3_robust.py  [--bc] [--no-extras-variant] [--save PATH | --replicate-check]
       python3 data/v2/train_phrr_v3_robust.py --bc --replicate-target
"""
import argparse
import os

# ~150-row fits: OpenMP threads (HGB) only add overhead, and they oversubscribe the machine badly when other
# jobs are running (a first run with default threads was still going after 11 min; single-threaded it takes ~5)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from eval_protocol import bootstrap_ci, grouped_shuffle_r2, repeated_group_kfold, report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA = 0.30
TARGET = "peak_hrrpua_kWm2"
BASE = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct"]
PHYS = ["cone_mass_per_area_kgm2", "cone_thickness_mm", "density_kgm3",
        "mcc_total_hr_kJg", "mcc_t_peak_C", "mcc_t_onset_C", "log_kpc"]
# mcc_extras_fsri.csv columns behind mcc_total_hr_kJg / mcc_t_onset_C. --bc switches to the
# baseline-corrected ones added after this stream ran (raw ones picked up end-of-run MCC drift).
EXTRAS = {"hr": "total_hr_kJg", "onset": "t_onset_C"}
# (rows, materials) of the pHRR x MCC inner join, per mcc_hrr.csv version
DATA_VERSIONS = {(175, 60): "pre-fix mcc_hrr.csv (branch commit cb53979)",
                 (196, 67): "mcc_hrr.csv after the 2026-09-29 MCC ingest fix"}


def rf():
    return RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)


def hgb():
    # ponytail: fixed a priori for ~140-175 training rows (default min_samples_leaf=20 is too coarse), not tuned
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                         min_samples_leaf=10, random_state=0)


def ambient_dry(df, col):
    df = df[df["condition"] == "Dry"]
    lowest = df.groupby("material")["Mean_Temp"].transform("min")
    return df[df["Mean_Temp"] == lowest].groupby("material")[col].mean()


def replicate_mean_target():
    """Canonical pHRR target (round 3): per material x flux, mean phrr_kWm2 over replicates that ignited and were
    not stopped early. truncated_test replicates are kept (pre-declared definition)."""
    r = pd.read_csv(f"{DATA_DIR}/cone_replicates_fsri.csv")
    r = r[~r["did_not_ignite"] & ~r["early_stop"]]
    return r.groupby(["material", "heat_flux_kWm2"], as_index=False).agg(
        **{TARGET: ("phrr_kWm2", "mean"), "n_reps": ("phrr_kWm2", "size"), "n_trunc": ("truncated_test", "sum"),
           "n_tags": ("hf_tag", "nunique")})


def load(replicate_target=False):
    phrr = replicate_mean_target() if replicate_target else pd.read_csv(f"{DATA_DIR}/phrr_fsri.csv")
    mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv")
    mcc_agg = mcc.groupby("material").agg(mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"),
                                          mcc_peak_norm_Wg=("rate_normalized_peak_Wg", "mean"),
                                          mcc_char_yield_pct=("char_yield_pct", "mean")).reset_index()
    x = pd.read_csv(f"{DATA_DIR}/mcc_extras_fsri.csv")
    assert np.allclose(x["hrc_JgK"], x["peak_hrr_Wg"] / x["heating_rate_C_s"], atol=0.01), "HRC no longer redundant"
    x_agg = x.groupby("material").agg(mcc_total_hr_kJg=(EXTRAS["hr"], "mean"), mcc_t_peak_C=("t_peak_C", "mean"),
                                      mcc_t_onset_C=(EXTRAS["onset"], "mean")).reset_index()
    props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv")
    props["category"] = props["categories"].fillna("").str.split(";").str[0].str.strip().replace("", "uncategorized")
    props = props[["material", "category", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
    k = pd.read_csv(f"{DATA_DIR}/thermal_conductivity_fsri.csv")
    c = pd.read_csv(f"{DATA_DIR}/specific_heat_fsri.csv")
    kpc = np.log(ambient_dry(k, "Average_Cond") * ambient_dry(c, "Spc_Heat")).dropna()
    thermal = kpc.rename("log_kpc").rename_axis("material").reset_index()
    df = (phrr.merge(mcc_agg, on="material", how="left").merge(x_agg, on="material", how="left")
          .merge(props, on="material", how="left").merge(thermal, on="material", how="left"))
    assert df["category"].notna().all()
    df["has_mcc"] = df[BASE].notna().all(axis=1)
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def r2_rows(y, P):
    """R2 of each row of P (repeats x n) against y, vectorised."""
    return 1 - ((P - y) ** 2).sum(axis=1) / ((y - y.mean()) ** 2).sum()


def delta_ci(y, P_new, P_ref, groups, n_boot=2000, seed=0):
    """Material-bootstrap 95% CI of the repeat-averaged paired R2 gain of P_new over P_ref."""
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))])
        stats.append((r2_rows(y[rows], P_new[:, rows]) - r2_rows(y[rows], P_ref[:, rows])).mean())
    return np.percentile(stats, [2.5, 97.5])


def evaluate(name, model, df, feats, log_target, check_report=False):
    """eval_protocol's two protocols + CI (same numbers report() prints), keeping the OOF predictions
    for paired comparisons, plus the same repeated GroupKFold grouped by category."""
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    mat, cat = df["material"].to_numpy(), df["category"].to_numpy()
    gss = grouped_shuffle_r2(model, X, y, mat, log_target=log_target)
    rkf, oof = repeated_group_kfold(model, X, y, mat, log_target=log_target)
    lo, hi = bootstrap_ci(y, oof[0], mat)
    out = dict(gss_mean=gss.mean(), gss_std=gss.std(), gss_min=gss.min(), gss_max=gss.max(),
               rkf_mean=rkf.mean(), rkf_std=rkf.std(), ci_lo=lo, ci_hi=hi)
    if check_report:  # the literal protocol printout; must match the components above
        rep = report(name, model, X, y, mat, log_target=log_target, sota=SOTA)
        assert all(np.isclose(rep[k], out[k]) for k in out), "components must reproduce report()"
    else:
        print(f"{name}: {len(y)} rows / {len(np.unique(mat))} materials, features={feats}, log_target={log_target}")
        print(f"  GroupShuffleSplit x10 (leaderboard protocol): mean R2={gss.mean():.4f} std={gss.std():.4f} "
              f"min={gss.min():.4f} max={gss.max():.4f}")
        print(f"  Repeated GroupKFold 5x10 (pooled OOF):        mean R2={rkf.mean():.4f} std={rkf.std():.4f}; "
              f"material-bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  vs SOTA {SOTA}: delta (GKF mean) {rkf.mean() - SOTA:+.4f}; CI excludes SOTA from above: {lo > SOTA}")
    crkf, coof = repeated_group_kfold(model, X, y, cat, log_target=log_target)
    clo, chi = bootstrap_ci(y, coof[0], mat)
    klo, khi = bootstrap_ci(y, coof[0], cat)
    print(f"  CATEGORY-grouped GroupKFold 5x10 ({len(np.unique(cat))} categories): mean R2={crkf.mean():.4f} "
          f"std={crkf.std():.4f}; material-bootstrap CI [{clo:.3f}, {chi:.3f}], category-bootstrap CI [{klo:.3f}, {khi:.3f}]")
    return dict(out, name=name, feats=feats, log_target=log_target, model=model, y=y, mat=mat,
                rkf=rkf, oof=oof, crkf=crkf, coof=coof, cat_mean=crkf.mean(), cat_std=crkf.std(),
                cat_ci_lo=clo, cat_ci_hi=chi, cat_kci_lo=klo, cat_kci_hi=khi)


def seed42_check(df, exp006_rows=True):
    """Reproduce exp006 (seed 42 -> 0.5744; only on its own pre-fix 175 rows), then place seed 42 among
    200 GroupShuffleSplit seeds."""
    X, y, g = df[BASE].to_numpy(float), df[TARGET].to_numpy(float), df["material"].to_numpy()
    scores, bad = [], []
    for seed in range(200):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, g))
        scores.append(r2_score(y[te], clone(rf()).fit(X[tr], y[tr]).predict(X[te])))
        bad += list(set(g[te])) if scores[-1] < -0.5 else []
    scores = np.array(scores)
    n_bad = int((scores < -0.5).sum())
    print(f"\n{n_bad}/200 seeds score R2 < -0.5; materials most often in those test sets: "
          f"{pd.Series(bad).value_counts().head(3).to_dict()}")
    if exp006_rows:
        assert abs(scores[42] - 0.5744) < 5e-4, f"exp006 not reproduced: {scores[42]:.4f}"
    print(f"\nexp006 {'reproduced' if exp006_rows else 'config, NOT its rows'}: seed 42 R2={scores[42]:.4f}. Over GroupShuffleSplit seeds 0-199 (same model): "
          f"mean={scores.mean():.4f} median={np.median(scores):.4f} std={scores.std():.4f} "
          f"min={scores.min():.4f} max={scores.max():.4f}")
    print(f"  seed 42 is at the {100 * (scores < scores[42]).mean():.0f}th percentile; "
          f"{100 * (scores > SOTA).mean():.0f}% of seeds beat 0.30; 5th-95th pct "
          f"[{np.percentile(scores, 5):.3f}, {np.percentile(scores, 95):.3f}]")


def error_shares(r, df, top=5):
    """Which materials carry the pooled OOF squared error (repeat mean), and R2 without the top 3."""
    se = ((r["oof"] - r["y"]) ** 2).mean(axis=0)
    d = df.assign(sq_err=se, pred=r["oof"].mean(axis=0)).groupby("material").agg(
        share_pct=("sq_err", "sum"), true_max=(TARGET, "max"), pred_max=("pred", "max"),
        thickness_mm=("cone_thickness_mm", "first"), mass_per_area=("cone_mass_per_area_kgm2", "first"))
    d = d.assign(share_pct=100 * d["share_pct"] / se.sum()).sort_values("share_pct", ascending=False)
    keep = ~np.isin(r["mat"], d.index[:3])
    print(f"  {r['name']}: largest shares of pooled OOF squared error; R2 without the top 3 = "
          f"{r2_rows(r['y'][keep], r['oof'][:, keep]).mean():.4f}\n" + d.head(top).round(1).to_string())


def nojoin_paired(df):
    """V5 vs V0 on one 76-material partition: V0 trains/predicts only rows with MCC (its inner join),
    V5 trains on every training-fold row. Both scored on the shared MCC rows; V5 also on all rows."""
    y, g, m = df[TARGET].to_numpy(float), df["material"].to_numpy(), df["has_mcc"].to_numpy()
    Xb, Xa = df[BASE].to_numpy(float), df[BASE + PHYS].to_numpy(float)
    P0, P5 = np.full((10, len(y)), np.nan), np.full((10, len(y)), np.nan)
    for r in range(10):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(Xa, y, g):
            trm, tem = tr[m[tr]], te[m[te]]
            P0[r, tem] = clone(rf()).fit(Xb[trm], y[trm]).predict(Xb[tem])
            P5[r, te] = clone(hgb()).fit(Xa[tr], y[tr]).predict(Xa[te])
    r0, r5c, r5a = r2_rows(y[m], P0[:, m]), r2_rows(y[m], P5[:, m]), r2_rows(y, P5)
    d = r5c - r0
    lo, hi = delta_ci(y[m], P5[:, m], P0[:, m], g[m])
    print(f"\nV5 vs V0 on one shared {len(np.unique(g))}-material partition (GroupKFold 5x10):")
    print(f"  scored on the {m.sum()} MCC rows / {len(np.unique(g[m]))} materials: V0 rf_base={r0.mean():.4f} "
          f"(std {r0.std():.4f}), V5 hgb_phys_nojoin={r5c.mean():.4f} (std {r5c.std():.4f}); paired delta "
          f"{d.mean():+.4f}, improved {int((d > 0).sum())}/10, delta CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  V5 on all {len(y)} rows / {len(np.unique(g))} materials: {r5a.mean():.4f} (std {r5a.std():.4f}); "
          f"no-MCC rows only ({(~m).sum()} rows / {len(np.unique(g[~m]))} materials): "
          f"{r2_rows(y[~m], P5[:, ~m]).mean():.4f}")


def replicate_check(df):
    """Target sensitivity (added after the FIGRA stream found that FSRI's averaged HRRPUA curves, which
    phrr_fsri.csv is taken from, end early and smear peaks). The unchanged V0 and V3 are scored on per-replicate pHRR
    from figra_fsri.csv (raw Scan -> FSRI's HRR formula), over the same 175 material x flux pairs:
    (a) the replicate mean per pair, i.e. the same rows and splits as the main run with only the target
    changed; (b) one row per replicate, still grouped by material."""
    f = pd.read_csv(f"{DATA_DIR}/figra_fsri.csv")
    print(f"figra_fsri.csv: {len(f)} replicate rows; test_status counts {f['test_status'].value_counts().to_dict()}, "
          f"peak <= 0: {int((f['peak_hrrpua_kWm2'] <= 0).sum())}")
    f = f[(f["test_status"] == 1) & (f["peak_hrrpua_kWm2"] > 0)]
    keys = ["material", "heat_flux_kWm2"]
    rep = df[keys].merge(f[keys + [TARGET]], on=keys)
    mean = rep.groupby(keys, as_index=False, sort=False)[TARGET].mean()
    d_mean = df.drop(columns=TARGET).merge(mean, on=keys)
    old = df.set_index(keys)[TARGET].reindex(pd.MultiIndex.from_frame(d_mean[keys])).to_numpy()
    rel = d_mean[TARGET].to_numpy() / old - 1
    print(f"(a) replicate-mean target: {len(d_mean)} of {len(df)} pairs / {d_mean['material'].nunique()} materials; "
          f"vs averaged-curve peak: median {100 * np.median(rel):+.1f}%, >10% higher in {int((rel > 0.1).sum())} pairs, "
          f">10% lower in {int((rel < -0.1).sum())}")
    d_rep = df.drop(columns=TARGET).merge(rep, on=keys)
    print(f"(b) per-replicate target: {len(d_rep)} rows / {d_rep['material'].nunique()} materials")
    for label, d in (("(a) replicate mean", d_mean), ("(b) per replicate", d_rep)):
        print(f"\n--- {label} ---")
        r0 = evaluate("V0 rf_base", rf(), d, BASE, False)
        r3 = evaluate("V3 rf_phys_log", rf(), d, BASE + PHYS, True)
        dd, dc = r3["rkf"] - r0["rkf"], r3["crkf"] - r0["crkf"]
        lo, hi = delta_ci(r3["y"], r3["oof"], r0["oof"], r3["mat"])
        print(f"  V3 vs V0 paired: material-grouped {dd.mean():+.4f} (improved {int((dd > 0).sum())}/10, CI "
              f"[{lo:+.3f}, {hi:+.3f}]); category-grouped {dc.mean():+.4f} (improved {int((dc > 0).sum())}/10)")


def replicate_target_run(df, full):
    """Round 3 pre-declared runs (reports/sprint_0929/replicate_targets.md) on the replicate-mean target."""
    assert EXTRAS["hr"] == "total_hr_bc_kJg", "A1 is exp022/exp037's config: run with --bc"
    keys = ["material", "heat_flux_kWm2"]
    old = load()
    old = old[old["has_mcc"]]
    assert (len(old), old["material"].nunique()) == (196, 67), "expected the fixed MCC data"
    both = df[keys + [TARGET]].merge(old[keys + [TARGET]], on=keys, suffixes=("", "_old"))
    rel = both[TARGET] / both[f"{TARGET}_old"] - 1
    print(f"Replicate-mean target: {len(full)} material x flux pairs / {full['material'].nunique()} materials; "
          f"with MCC features (the rows used): {len(df)} / {df['material'].nunique()}")
    print(f"  replicates per pair {df['n_reps'].value_counts().sort_index().to_dict()}; pairs with a truncated_test "
          f"replicate kept: {int((df['n_trunc'] > 0).sum())}; pairs pooling >1 hf_tag: "
          f"{df.loc[df['n_tags'] > 1, keys].values.tolist()}")
    print(f"  vs the default 196-row set: {len(both)} pairs shared, {len(df) - len(both)} only here, "
          f"{len(old) - len(both)} only there; on shared pairs new/old - 1: median {100 * rel.median():+.1f}%, "
          f"> +5% on {int((rel > 0.05).sum())}, < -5% on {int((rel < -0.05).sum())}")
    v0 = evaluate("V0 rf_base", rf(), df, BASE, False)
    a1 = evaluate("A1 V3 rf_phys_log", rf(), df, BASE + PHYS, True, check_report=True)
    a2 = evaluate("A2 V3 rf_phys_log, rate-normalised MCC peak", rf(), df,
                  [BASE[0], "mcc_peak_norm_Wg", BASE[2]] + PHYS, True)
    print("\n=== Paired (same rows, same splits; delta CI = material bootstrap of the repeat-mean gain) ===")
    for new, ref in ((a1, v0), (a2, a1)):
        d, dc = new["rkf"] - ref["rkf"], new["crkf"] - ref["crkf"]
        new["lo"], new["hi"] = delta_ci(new["y"], new["oof"], ref["oof"], new["mat"])
        new["n_up"] = int((d > 0).sum())
        print(f"  {new['name']} vs {ref['name']}: material-grouped delta {d.mean():+.4f} (improved {new['n_up']}/10, "
              f"CI [{new['lo']:+.3f}, {new['hi']:+.3f}]); category-grouped delta {dc.mean():+.4f} "
              f"(improved {int((dc > 0).sum())}/10)")
    print(f"\nA1 rule (beats pomazi_2026's {SOTA} only if CI lower bound > {SOTA}): lower bound {a1['ci_lo']:.3f} -> "
          f"{'BEATS' if a1['ci_lo'] > SOTA else 'does NOT beat'}")
    ok = a2["n_up"] >= 8 and a2["lo"] > 0
    print(f"A2 rule (adopt over A1 only if better in >= 8/10 paired repeats AND gain CI excludes 0): "
          f"{a2['n_up']}/10, CI [{a2['lo']:+.3f}, {a2['hi']:+.3f}] -> {'ADOPT A2' if ok else 'not met, A1 stays'}")
    print(f"\n{'variant':46s} {'GKF mean':>9s} {'std':>7s} {'95% CI':>17s} {'GSS mean':>9s} {'cat mean':>9s} {'cat CI':>17s}")
    for r in (v0, a1, a2):
        print(f"{r['name']:46s} {r['rkf_mean']:9.4f} {r['rkf_std']:7.4f} [{r['ci_lo']:6.3f}, {r['ci_hi']:6.3f}] "
              f"{r['gss_mean']:9.4f} {r['cat_mean']:9.4f} [{r['cat_ci_lo']:6.3f}, {r['cat_ci_hi']:6.3f}]")
    return a1


def fit_final(res, df):
    X, y = df[res["feats"]].to_numpy(float), df[TARGET].to_numpy(float)
    model = clone(res["model"])
    if res["log_target"]:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model.fit(X, y)


def crosslab(res, df, fsri_material="PMMA", override=None):
    """validate_phrr_crosslab.py's check, for any variant: FSRI's own material-level features for PMMA
    on every MaCFP row. Refit on all rows here, so V0 should reproduce exp008's 0.1986. `override`
    replaces specimen-geometry features with MaCFP's own (their specimens aren't FSRI's)."""
    macfp = pd.read_csv(f"{DATA_DIR}/phrr_macfp_pmma.csv")
    row = df[df["material"] == fsri_material].iloc[0]
    for f in res["feats"]:
        if f != "heat_flux_kWm2":
            macfp[f] = (override or {}).get(f, row[f])
    fsri_material += " + MaCFP geometry" if override else ""
    pred = fit_final(res, df).predict(macfp[res["feats"]].to_numpy(float))
    y = macfp["peak_hrr_kWm2"].to_numpy(float)
    per_hf = {hf: r2_score(y[s], pred[s]) for hf, s in ((hf, macfp["heat_flux_kWm2"].to_numpy() == hf)
                                                        for hf in sorted(macfp["heat_flux_kWm2"].unique())) if s.sum() > 1}
    print(f"  {res['name']} (FSRI {fsri_material} features): cross-lab R2={r2_score(y, pred):.4f}, "
          f"MAE={np.mean(np.abs(y - pred)):.1f} kW/m2; per heat flux " +
          ", ".join(f"{hf:g}: pred {pred[macfp['heat_flux_kWm2'].to_numpy() == hf][0]:.0f} vs true mean "
                    f"{y[macfp['heat_flux_kWm2'].to_numpy() == hf].mean():.0f} (R2 {r:.2f})" for hf, r in per_hf.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="write the rule-picked winner's joblib bundle here (lead only for models/v2)")
    ap.add_argument("--replicate-check", action="store_true",
                    help="only run the per-replicate pHRR target sensitivity check for V0 and V3")
    ap.add_argument("--save-variant", help="save this variant (e.g. 'V3 rf_phys_log') instead of the rule winner. "
                    "The lead committed to V3 before the --bc rerun, whose rule winner became V4 by 0.001")
    ap.add_argument("--no-extras-variant", action="store_true",
                    help="also score V3 minus the MCC extras, paired against V3 (pre-declared in mcc_fix.md)")
    ap.add_argument("--bc", action="store_true",
                    help="use mcc_extras' baseline-corrected total_hr_bc_kJg / t_onset_bc_C (the promoted v2 model)")
    ap.add_argument("--replicate-target", action="store_true",
                    help="only the round-3 pre-declared A1/A2 (+V0) on the cone_replicates_fsri.csv replicate-mean target")
    args = ap.parse_args()
    if args.bc:
        EXTRAS.update(hr="total_hr_bc_kJg", onset="t_onset_bc_C")
        print("Using baseline-corrected MCC extras (total_hr_bc_kJg, t_onset_bc_C)")

    full = load(args.replicate_target)
    df = full[full["has_mcc"]].reset_index(drop=True)  # the promoted config's inner-join rows
    if args.replicate_target:
        a1 = replicate_target_run(df, full)
        if args.save:  # the lead's refit of the promoted config on the canonical target (models/v2/phrr_rf_v3.joblib)
            joblib.dump(dict(model=fit_final(a1, df), features=a1["feats"], target=TARGET,
                             trained_on=f"cone_replicates_fsri.csv replicate-mean pHRR (no did_not_ignite / early_stop) "
                                        f"+ fixed MCC ({EXTRAS['hr']}, {EXTRAS['onset']}) + props + HFM, "
                                        f"all {len(df)} rows / {df['material'].nunique()} materials",
                             validation=f"train_phrr_v3_robust.py --bc --replicate-target (exp043): repeated GroupKFold "
                                        f"5x10 pooled OOF R2={a1['rkf_mean']:.4f} (material-bootstrap CI "
                                        f"[{a1['ci_lo']:.3f}, {a1['ci_hi']:.3f}]); does NOT beat SOTA 0.30 (CI lower bound)",
                             sota_reference="pomazi_2026 Peak HRR real held-out RFR R2=0.30 (benchmarks/sota_board.csv)"),
                        args.save)
            p = joblib.load(args.save)["model"].predict(df[a1["feats"]].to_numpy(float))
            assert p.max() > 100, "saved model must predict kW/m2, not log units"
            print(f"\nSaved A1 refit to {args.save} (in-sample R2 {r2_score(df[TARGET], p):.4f}, sanity only)")
        return
    shape = (len(df), df["material"].nunique())
    assert shape in DATA_VERSIONS, shape
    print(f"Data version: {DATA_VERSIONS[shape]}")
    if args.replicate_check:
        return replicate_check(df)
    print(f"Same-rows set (pHRR x MCC inner join, as promoted): {len(df)} rows / {df['material'].nunique()} materials")
    print(f"No-join set (every pHRR row): {len(full)} rows / {full['material'].nunique()} materials")
    print("PHYS coverage on the same-rows set (non-NaN rows / materials):",
          {f: (int(df[f].notna().sum()), df.loc[df[f].notna(), "material"].nunique()) for f in PHYS})
    print("Category groups (first-listed FSRI category), materials per group:",
          df.groupby("category")["material"].nunique().sort_values(ascending=False).to_dict())

    seed42_check(df, exp006_rows=shape == (175, 60))

    print(f"\n=== Pre-declared variants, same {len(df)} rows, same splits ===")
    variants = [("V0 rf_base", rf(), BASE, False), ("V1 rf_base_log", rf(), BASE, True),
                ("V2 rf_phys", rf(), BASE + PHYS, False), ("V3 rf_phys_log", rf(), BASE + PHYS, True),
                ("V4 hgb_phys", hgb(), BASE + PHYS, False)]
    res = [evaluate(n, m, df, f, lt, check_report=(i == 0)) for i, (n, m, f, lt) in enumerate(variants)]
    v0 = res[0]

    print("\n=== Paired vs V0 (same rows, same splits; delta CI = material bootstrap of the repeat-mean gain) ===")
    for r in res[1:]:
        d, dc = r["rkf"] - v0["rkf"], r["crkf"] - v0["crkf"]
        lo, hi = delta_ci(r["y"], r["oof"], v0["oof"], r["mat"])
        print(f"  {r['name']:16s} material-grouped delta {d.mean():+.4f} (improved {int((d > 0).sum())}/10, "
              f"CI [{lo:+.3f}, {hi:+.3f}]); category-grouped delta {dc.mean():+.4f} (improved {int((dc > 0).sum())}/10)")
        r.update(delta=d.mean(), n_improved=int((d > 0).sum()), delta_lo=lo, delta_hi=hi,
                 cat_delta=dc.mean(), cat_n_improved=int((dc > 0).sum()))

    extra = []
    if args.no_extras_variant:
        v3 = next(r for r in res if r["name"] == "V3 rf_phys_log")
        print(f"\n=== Pre-declared secondary (mcc_fix.md): V3 minus MCC extras, same {len(df)} rows, same splits ===")
        r = evaluate("V3b phys_noextras", rf(), df, BASE + PHYS[:3] + PHYS[6:], True)
        for ref in (v3, v0):
            d, dc = r["rkf"] - ref["rkf"], r["crkf"] - ref["crkf"]
            lo, hi = delta_ci(r["y"], r["oof"], ref["oof"], r["mat"])
            print(f"  vs {ref['name']:16s} material-grouped delta {d.mean():+.4f} (improved {int((d > 0).sum())}/10, "
                  f"CI [{lo:+.3f}, {hi:+.3f}]); category-grouped delta {dc.mean():+.4f} (improved {int((dc > 0).sum())}/10)")
            if ref is v3:
                ok = int((d > 0).sum()) >= 8 and lo > 0
        print(f"  Rule (prefer over V3 only if better in >= 8/10 paired repeats AND gain CI excludes 0): "
              f"{'PASSES' if ok else 'not met'}")
        extra = [r]

    print("\n=== V5: HGB, NaN-native, no inner join ===")
    v5 = evaluate("V5 hgb_phys_nojoin", hgb(), full, BASE + PHYS, False)
    nojoin_paired(full)

    print(f"\n=== Summary (same {len(df)} rows; winner = highest material-grouped GKF mean among V0-V4) ===")
    print(f"{'variant':16s} {'GKF mean':>9s} {'std':>7s} {'95% CI':>17s} {'GSS mean':>9s} {'cat mean':>9s} {'cat CI':>17s}")
    for r in res + extra + [v5]:
        print(f"{r['name']:16s} {r['rkf_mean']:9.4f} {r['rkf_std']:7.4f} [{r['ci_lo']:6.3f}, {r['ci_hi']:6.3f}] "
              f"{r['gss_mean']:9.4f} {r['cat_mean']:9.4f} [{r['cat_ci_lo']:6.3f}, {r['cat_ci_hi']:6.3f}]"
              + ("  (220 rows / 76 materials, not a candidate)" if r is v5 else "")
              + ("  (secondary, not a rule candidate)" if r in extra else ""))
    win = max(res, key=lambda r: r["rkf_mean"])
    passes = win is not v0 and win["n_improved"] >= 8 and win["delta_lo"] > 0
    print(f"Winner by rule: {win['name']}. Promotion bar (>=8/10 paired repeats AND delta CI excludes 0): "
          f"{'PASSES' if passes else 'not met'}")

    print("\n=== Where the error is (repeat-mean OOF, material-grouped) ===")
    for r in dict.fromkeys([v0["name"], win["name"]]):
        error_shares(next(x for x in res if x["name"] == r), df)

    if win is not v0:
        print(f"\n=== Post-hoc diagnostics for {win['name']} (explain the gain; NOT selection candidates) ===")
        X, y, g = df[win["feats"]].to_numpy(float), df[TARGET].to_numpy(float), df["material"].to_numpy()
        drops = {"geometry (mass/area, thickness, density)": PHYS[:3],
                 "MCC extras (total HR, T_peak, T_onset)": PHYS[3:6], "thermal inertia (log_kpc)": PHYS[6:]}
        for label, drop in drops.items():
            keep = [i for i, f in enumerate(win["feats"]) if f not in drop]
            rkf, _ = repeated_group_kfold(win["model"], X[:, keep], y, g, log_target=win["log_target"])
            d = rkf - win["rkf"]
            print(f"  minus {label:42s} GKF mean {rkf.mean():.4f} (paired vs winner {d.mean():+.4f}, "
                  f"worse in {int((d < 0).sum())}/10)")
        imp = fit_final(win, df)
        imp = getattr(imp, "regressor_", imp)
        if hasattr(imp, "feature_importances_"):
            print("  Impurity importances, full-data fit (interpretation only): " + ", ".join(
                f"{f}={v:.3f}" for f, v in sorted(zip(win["feats"], imp.feature_importances_), key=lambda t: -t[1])))

    print(f"\n=== Cross-lab: MaCFP PMMA, 67 tests / 10 labs (refit on all {len(df)} rows) ===")
    # MaCFP specimens: nominally 6 mm black cast PMMA (raw/matl-db/PMMA/README.md); density from FSRI Black_PMMA
    rho = df.loc[df["material"] == "Black_PMMA", "density_kgm3"].iloc[0]
    macfp_geom = {"cone_thickness_mm": 6.0, "cone_mass_per_area_kgm2": 6.0e-3 * rho, "density_kgm3": rho}
    for r in dict.fromkeys([v0["name"], win["name"]]):
        r = next(x for x in res if x["name"] == r)
        crosslab(r, df)  # exp008's choice: FSRI clear PMMA
        crosslab(r, df, "Black_PMMA")  # closer product: FSRI black cast PMMA (but ~8.4 mm cone specimens)
        if r is not v0:
            crosslab(r, df, "Black_PMMA", override=macfp_geom)

    if args.save:
        if args.save_variant:
            win = next(r for r in res + extra if r["name"] == args.save_variant)
        bundle = dict(model=fit_final(win, df), features=win["feats"], target=TARGET,
                      trained_on=f"data/v2/phrr_fsri.csv + mcc_hrr.csv (inner join) + PHYS tables "
                                 f"(MCC extras columns: {EXTRAS['hr']}, {EXTRAS['onset']}), "
                                 f"all {len(df)} rows / {df['material'].nunique()} materials",
                      validation=f"train_phrr_v3_robust.py: repeated GroupKFold 5x10 pooled OOF R2="
                                 f"{win['rkf_mean']:.4f} (material-bootstrap CI [{win['ci_lo']:.3f}, {win['ci_hi']:.3f}])",
                      sota_reference="pomazi_2026 Peak HRR real held-out RFR R2=0.30 (benchmarks/sota_board.csv)")
        joblib.dump(bundle, args.save)
        p = joblib.load(args.save)["model"].predict(df[win["feats"]].to_numpy(float))
        assert p.max() > 100, "saved model must predict kW/m2, not log units"
        print(f"\nSaved {win['name']} bundle to {args.save} (in-sample R2 {r2_score(df[TARGET], p):.4f}, "
              f"sanity only)")


if __name__ == "__main__":
    main()
