#!/usr/bin/env python3
"""Dravix V2 -- FIGRA round 1b (configs: stale_rebaseline_round1b.json + figra_round1.json, pre-declared).
Human decision 2026-10-02 (option A): re-baseline exp012 config on current (post-f063d48) data; c1 vs that baseline.
(Derived from train_figra_round1.py.)

1. Reproduce exp012 (train_figra_v1.py, row set A, F2, HGB, log1p target): rkf mean 0.503,
   CI [0.324, 0.663] within 0.002, else status=blocked and stop.
2. c1 two-stage: same HGB settings and F2 features, one model on log(pHRR), one on log(t_pHRR),
   FIGRA_pred = exp(a) / exp(b); both fit in-fold on identical repeated GroupKFold splits
   (groups = material). Scored in raw kW/m2/s against exp012's target (replicate-mean FIGRA).
   pHRR and t_pHRR are the replicate means over the same spark-ignited, ignition-recorded replicates.
3. Rule (mechanical): c1 replaces exp012 only if >= 8/10 paired wins AND ci_lo >= 0.324.
4. Writes experiments/results/figra_r1b_c1_twostage.json and reports/figra_same_polymer_table.md
   (OOF repeat 0 of the selected model).

Run: OMP_NUM_THREADS=1 python train_figra_round1.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from eval_protocol import bootstrap_ci, repeated_group_kfold, report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
SOTA = 0.9724
EXP012 = dict(rkf_mean=0.4563, ci_lo=0.245, ci_hi=0.639)  # re-baseline expectation (197 rows / 67 materials)
TOL = 0.002

# ---- rows: identical to train_figra_v1.py row set A (all fluxes, materials with MCC) ----
reps = pd.read_csv(f"{DATA_DIR}/figra_fsri.csv")
reps = reps[reps.spark_ignitor & reps.ignition_recorded]
df = (reps.groupby(["material", "heat_flux_kWm2"])
      .agg(figra_kWm2s=("figra_kWm2s", "mean"), phrr_kWm2=("peak_hrrpua_kWm2", "mean"),
           t_phrr_s=("time_to_peak_s", "mean"), n_reps=("figra_kWm2s", "size")).reset_index())
mcc = pd.read_csv(f"{DATA_DIR}/mcc_hrr.csv").groupby("material").agg(
    mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean"))
extras = pd.read_csv(f"{DATA_DIR}/mcc_extras_fsri.csv").groupby("material").agg(
    mcc_hrc_JgK=("hrc_JgK", "mean"), mcc_total_hr_kJg=("total_hr_kJg", "mean"), mcc_t_peak_C=("t_peak_C", "mean"))
props = pd.read_csv(f"{DATA_DIR}/material_props_fsri.csv").set_index("material")[
    ["categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
df = df.join(mcc, on="material").join(extras, on="material").join(props, on="material")
F2 = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "density_kgm3", "cone_thickness_mm",
      "cone_mass_per_area_kgm2", "mcc_hrc_JgK", "mcc_total_hr_kJg", "mcc_t_peak_C"]
d = df[df.mcc_peak_hrr_Wg.notna()].reset_index(drop=True)
X, y, groups = d[F2].to_numpy(float), d.figra_kWm2s.to_numpy(), d.material.to_numpy()
lphrr, ltp = np.log(d.phrr_kWm2.to_numpy()), np.log(d.t_phrr_s.to_numpy())
assert np.isfinite(lphrr).all() and np.isfinite(ltp).all()
print(f"FIGRA round1 rows: {len(d)} / {d.material.nunique()} materials")

HGB = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                    min_samples_leaf=5, random_state=0)
slug = "figra_r1b_c1_twostage"
result = dict(
    slug=slug, target="Fire Growth Rate FIGRA = pHRR / t_pHRR from test start (kW/m2/s)",
    round_config="experiments/configs/stale_rebaseline_round1b.json + figra_round1.json",
    dataset="data/v2/figra_fsri.csv replicate mean per material x flux (spark-ignited, ignition recorded) "
            "+ mcc_hrr + mcc_extras_fsri + material_props_fsri; materials with MCC (exp012 rows)",
    features=F2, model="HistGradientBoostingRegressor x2 (log pHRR, log t_pHRR); FIGRA = exp(a)/exp(b)",
    hyperparameters="max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=0",
    code="data/v2/train_figra_round1b.py", n_rows=len(d), n_groups=int(d.material.nunique()),
    protocol="rgkf5x10", paired_vs="FIGRA re-baseline", sota=SOTA, sota_source="phan_morgan")


def blocked(reason):
    result.update(rkf_mean=None, rkf_std=None, ci_lo=None, ci_hi=None, gss_mean=None, paired_delta=None,
                  paired_wins=None, status="blocked", blocked_reason=reason, note=reason[:300])
    with open(f"{ROOT}/experiments/results/{slug}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("BLOCKED:", reason)
    raise SystemExit(2)


# ---- 1. reproduce exp012 ----
base = report("exp012 reproduction (A, F2, HGB, log1p)", HGB, X, y, groups, log_target=True, sota=SOTA)
rkf0, oof0 = repeated_group_kfold(HGB, X, y, groups, log_target=True)
ok = (abs(rkf0.mean() - EXP012["rkf_mean"]) <= TOL and abs(base["ci_lo"] - EXP012["ci_lo"]) <= TOL
      and abs(base["ci_hi"] - EXP012["ci_hi"]) <= TOL)
print(f"reproduction check: mean {rkf0.mean():.4f} (want {EXP012['rkf_mean']}), "
      f"CI [{base['ci_lo']:.4f}, {base['ci_hi']:.4f}] (want [{EXP012['ci_lo']}, {EXP012['ci_hi']}]) -> {'OK' if ok else 'FAIL'}")
if not ok:
    blocked(f"FIGRA re-baseline did not reproduce: rkf {rkf0.mean():.4f}, CI [{base['ci_lo']:.4f}, {base['ci_hi']:.4f}]")


base_res = dict(result)
base_res.update(slug="figra_rebaseline_exp012cfg", model="HistGradientBoostingRegressor (direct, log1p target)",
                paired_vs="FIGRA re-baseline", rkf_mean=round(float(rkf0.mean()), 4), rkf_std=round(float(rkf0.std()), 4),
                ci_lo=round(float(base["ci_lo"]), 4), ci_hi=round(float(base["ci_hi"]), 4), gss_mean=round(float(base["gss_mean"]), 4),
                paired_delta=0.0, paired_wins="n/a", status="ok", blocked_reason="",
                note="exp012 config re-run on post-f063d48 data; human decision 2026-10-02")
with open(f"{ROOT}/experiments/results/figra_rebaseline_exp012cfg.json", "w") as f:
    json.dump(base_res, f, indent=2)

# ---- 2. c1 two-stage ----
def two_stage(tr, te):
    a = clone(HGB).fit(X[tr], lphrr[tr]).predict(X[te])
    b = clone(HGB).fit(X[tr], ltp[tr]).predict(X[te])
    return np.exp(a) / np.exp(b)


oof_c1 = np.empty_like(oof0)
for r in range(10):
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, groups):
        assert set(groups[tr]).isdisjoint(groups[te])
        oof_c1[r, te] = two_stage(tr, te)
rkf_c1 = np.array([r2_score(y, p) for p in oof_c1])
lo, hi = bootstrap_ci(y, oof_c1[0], groups)
gss = []
for seed in range(10):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    gss.append(r2_score(y[te], two_stage(tr, te)))
gss = np.array(gss)
wins = int((rkf_c1 > rkf0).sum())
delta = float(rkf_c1.mean() - rkf0.mean())
adopt = wins >= 8 and lo >= base["ci_lo"]
print(f"\nc1 two-stage: rkf mean={rkf_c1.mean():.4f} std={rkf_c1.std():.4f} CI [{lo:.3f}, {hi:.3f}] "
      f"GSS mean={gss.mean():.4f}")
print(f"per-repeat exp012: {np.round(rkf0, 3).tolist()}\nper-repeat c1:     {np.round(rkf_c1, 3).tolist()}")
print(f"paired wins vs exp012: {wins}/10; delta {delta:+.4f}; ci_lo {lo:.3f} vs {base["ci_lo"]:.3f}")
print(f"RULE (>=8/10 wins AND ci_lo >= baseline ci_lo): {'c1 ADOPTED' if adopt else 'c1 NOT adopted, re-baseline stays'}")

result.update(rkf_mean=round(float(rkf_c1.mean()), 4), rkf_std=round(float(rkf_c1.std()), 4),
              ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), gss_mean=round(float(gss.mean()), 4),
              paired_delta=round(delta, 4), paired_wins=f"{wins}/10", status="ok", blocked_reason="",
              note=(f"re-baseline reproduced ({rkf0.mean():.4f}). c1 two-stage (log pHRR / log t_pHRR, HGB) "
                    f"{'meets' if adopt else 'does not meet'} the 8/10 and ci_lo>=baseline rule. "
                    f"Different population from phan_morgan; no claim vs 0.9724."))
with open(f"{ROOT}/experiments/results/{slug}.json", "w") as f:
    json.dump(result, f, indent=2)

# ---- 4. same-polymer table (selected model, OOF repeat 0) ----
sel_name = "c1 two-stage" if adopt else "FIGRA re-baseline (exp012 config, HGB direct, log1p)"
sel_oof = (oof_c1 if adopt else oof0)[0]
d["oof"] = sel_oof
# Phan/Morgan Table 2 values as transcribed in reports/sprint_0929/figra.md section 3.1:
# (polymer, TIG s, pHRR, FIGRA, implied t_pHRR s, [FSRI materials], match note)
PHAN = [
    ("PE", 40.3, 2584.7, 19.09, 135.4, ["LDPE", "HDPE"], "PE grade/density not given; LDPE and HDPE both listed"),
    ("PP", 33.3, 2255.9, 17.40, 129.7, ["PP"], "name match"),
    ("PMMA", 7.6, 866.4, 9.25, 93.7, ["PMMA"], "name match; FSRI PMMA is 2.8 mm sheet (Black_PMMA excluded: pigmented)"),
    ("PS", 31.1, 1419.7, 14.54, 97.6, ["HIPS"], "approximate: FSRI HIPS is high-impact PS, Phan is PS"),
    ("Nylon 6", 60.8, 1082.7, 5.96, 181.7, ["Nylon"], "approximate: FSRI 'Nylon' grade (6 vs 66) unspecified"),
    ("Nylon 66", 58.8, 986.6, 6.64, 148.6, ["Nylon"], "approximate: FSRI 'Nylon' grade (6 vs 66) unspecified"),
    ("PVC", 66.7, 274.0, 2.59, 106.0, ["PVC"], "name match; formulation (plasticizer, filler) unknown on both sides"),
    ("PET", 30.4, 472.7, 5.96, 79.4, ["PET"], "name match; FSRI PET is 6.7 mm thick"),
]
f50 = reps[reps.heat_flux_kWm2 == 50].groupby("material").figra_kWm2s.mean()
oof50 = d[d.heat_flux_kWm2 == 50].set_index("material").oof
rows, unmatched = [], []
for poly, tig, phrr, fig, tp, mats, note in PHAN:
    for m in mats:
        if m not in f50.index:
            unmatched.append(f"{poly} ({m} absent at 50 kW/m2)")
            continue
        o = f"{oof50[m]:.2f}" if m in oof50.index else "n/a (no MCC row, not in model)"
        rows.append(f"| {poly} | {fig:.2f} | {m} | {f50[m]:.2f} | {o} | {note} |")

# The other 7 Table 2 rows, names from the arXiv preprint 2504.00223 Table 2 (checked 2026-10-02). FSRI tested none of them.
UNL = [("PAN", 13.64), ("PCTFE", 0.34), ("PTFE", 0.19), ("PVF", 3.22), ("PVA", 7.50), ("PBT", 12.83), ("PCL", 19.08)]
rows += [f"| {n} | {v:.2f} | not tested by FSRI | — | — | no FSRI counterpart |" for n, v in UNL]
md = f"""# FIGRA same-polymer table: Phan/Morgan Table 2 vs FSRI vs Dravix, 50 kW/m2

Generated by `data/v2/train_figra_round1b.py` (round `figra_round1` on the 2026-10-02 re-baseline). Dravix model shown: **{sel_name}**
(selected by the pre-declared rule; c1 two-stage rkf {rkf_c1.mean():.3f}, {wins}/10 paired wins,
ci_lo {lo:.3f}; re-baseline rkf {rkf0.mean():.3f} [{base["ci_lo"]:.3f}, {base["ci_hi"]:.3f}]).

**Units.** All FIGRA columns are kW/m2/s (pHRR in kW/m2 divided by t_pHRR in s from test start). Phan/Morgan
values are the Table 2 FIGRA entries as transcribed in `reports/sprint_0929/figra.md` section 3.1, checked against
the arXiv preprint 2504.00223 Table 2 on 2026-10-02 (all 8 match); that report treats them as kW/m2/s, and the check pHRR/FIGRA gives 79-182 s, a plausible
t_pHRR, which only works if pHRR is kW/m2 and FIGRA is kW/m2/s. No unit conversion was applied.
FSRI = replicate mean of spark-ignited replicates at 50 kW/m2. Dravix = out-of-fold prediction, repeat 0 of
repeated GroupKFold 5x10 (material held out; the material was not in its own training fold), model trained on
all heat fluxes.

| Polymer (Phan/Morgan Table 2) | Phan/Morgan measured FIGRA | FSRI material | FSRI measured FIGRA | Dravix OOF prediction | Match note |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

**Matched:** 8 of Phan/Morgan's 15 Table 2 rows (PE, PP, PMMA, PS, Nylon 6, Nylon 66, PVC, PET) against {len(rows)-len(UNL)} polymer-FSRI pairings.
**No FSRI counterpart:** the other 7 of Phan/Morgan's 15 Table 2 rows (PAN, PCTFE, PTFE, PVF, PVA, PBT, PCL; names from
arXiv 2504.00223 Table 2). FSRI tested none of these polymers, so they cannot be
matched. {('Dropped for no FSRI row at 50 kW/m2: ' + '; '.join(unmatched)) if unmatched else 'Every FSRI candidate for the 8 listed rows has a 50 kW/m2 measurement.'}

Caveats:
1. Different populations and protocols: Phan/Morgan tested neat Sigma-Aldrich powders, pellets or film normalized by weight; FSRI tested thick sheets and products by area. Specimen geometry drives t_pHRR, so the same polymer name gives different FIGRA (e.g. PET 5.96 vs 2.98, PVC 2.59 vs 4.38).
2. The t_pHRR clock in Phan/Morgan (test start vs ignition) is inferred, not stated, and polymer-name matches (PS/HIPS, Nylon 6/66, PE/LDPE/HDPE) are approximate.
3. These are 8 single comparisons at one condition, not a like-for-like benchmark. No claim is made against Phan/Morgan's R2 of 0.9724.
"""
with open(f"{ROOT}/reports/figra_same_polymer_table.md", "w") as f:
    f.write(md)
print(f"\nwrote reports/figra_same_polymer_table.md ({len(rows)} pairings, model={sel_name})")
