#!/usr/bin/env python3
"""Dravix V2 -- THR: exp040's pick re-scored on the canonical per-replicate cone target (sprint 2026-09-29,
round 3, run A3 in reports/sprint_0929/replicate_targets.md; report only, nothing is selected here).

WHY: exp040 (train_thr_v1.py V1: THR = chi * m''_flux * h_c, h_c = MCC total_hr_bc_kJg, chi fitted on training
folds only) was scored on per-replicate raw-scan THR built inside train_thr_v1.py, which still averages in
non-igniting replicates (integrated over the whole test). cone_replicates_fsri.csv is now the canonical cone
target table. The pre-declared A3 target is its thr_MJm2, mean per material x flux over replicates that are
neither did_not_ignite nor early_stop (truncated_test replicates are kept). Features and model are exp040's,
imported from train_thr_v1.py unchanged, so this is a thin wrapper and train_thr_v1.py's default run is untouched.

Run: OMP_NUM_THREADS=1 python3 data/v2/train_thr_v1_replicates.py  (~1 min)
"""
import os

import pandas as pd

from eval_protocol import report
from train_thr_v1 import LIN, SOTA, build_features, scalar_table

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

r = pd.read_csv(f"{DATA_DIR}/cone_replicates_fsri.csv")
r = r[~r["did_not_ignite"] & ~r["early_stop"]]
thr = r.groupby(["material", "heat_flux_kWm2"], as_index=False).agg(
    total_heat_release_MJm2=("thr_MJm2", "mean"), n_reps=("thr_MJm2", "size"), n_trunc=("truncated_test", "sum"))
df = build_features(thr, scalar_table())
df["phys_MJm2"] = df.mpa_flux_kgm2 * df.hc_bc_kJg  # exp040: h_c = baseline-corrected MCC total heat release
core = df.dropna(subset=["phys_MJm2"]).reset_index(drop=True)
print(f"A3 target: {len(thr)} material x flux pairs / {thr.material.nunique()} materials; core rows (m''_flux and "
      f"MCC h_c): {len(core)} / {core.material.nunique()}; core pairs with a truncated_test replicate kept: "
      f"{int((core.n_trunc > 0).sum())}; replicates per core pair {core.n_reps.value_counts().sort_index().to_dict()}")
res = report("A3 V1 [phys_MJm2 = m''_flux * h_c_bc], replicate-mean thr_MJm2", LIN, core[["phys_MJm2"]].to_numpy(float),
             core.total_heat_release_MJm2.to_numpy(float), core.material.to_numpy(), sota=SOTA)
print(f"A3 headline: {res['rkf_mean']:.4f} [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}] on {len(core)} rows / "
      f"{core.material.nunique()} materials (exp040: 0.9531 [0.918, 0.975] on 190 / 65)")
