#!/usr/bin/env python3
"""Dravix V2 -- LOI config A (log10 eta_c, linear) on AR-05/14 + AR-01/53 rows.

WHY: train_loi_v1.py's config A (the report's LOI ~ eta_c^-1/4 physics, refit in each fold)
scored GKF 0.504 on the 32 AR-05/14 polymers. ingest_faa_loi_ar0153.py adds 6 cured epoxy
systems whose LOI and eta_c were measured by one group on the same formulations (unlike
AR-05/14's "generic" LOI values). This reruns config A, unchanged, on the union.

Fixed before running (nothing tuned, config A only, same code path as train_loi_v1.run):
  - Union = the 32 AR-05/14 rows with LOI + eta_c, plus the 6 AR-01/53 rows with LOI + eta_c.
    Dedup on exact polymer name; on a collision keep AR-05/14 and print the conflict.
    (AR-05/14's generic "EP" is the same chemical class as the DGEBA rows but not the same
    name, so it is not deduplicated; see the report.)
  - Family for the harder split: every AR-01/53 row is "thermoset", the family train_loi_v1
    already gives EP and CEA. So leave-one-family-out holds all epoxies/cyanates out together.
  - AR-01/53 LOI is the midpoint of the printed range.
  - Extra check: fit A on the 32 AR-05/14 rows only, predict the 6 AR-01/53 rows (a true
    external test: other report, other specimens, measured LOI). The fixed formula
    LOI = 125/eta_c^0.25 is scored on the same 6 rows.

Rerun: cd data/v2 && python3 ingest_faa_loi.py
       && python3 ingest_faa_loi_ar0153.py
       && python3 train_loi_v1_union.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from train_loi_v1 import CONFIGS, FAMILY, held_out, run

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
COLS = ["polymer", "LOI_pct", "HRC_J_gK", "source"]


def load():
    a = pd.read_csv(f"{DATA_DIR}/loi_faa_polymers.csv").dropna(subset=["LOI_pct", "HRC_J_gK"])
    a["source"], a["family"] = "AR-05/14", a["polymer"].map(FAMILY)
    b = pd.read_csv(f"{DATA_DIR}/loi_faa_ar0153.csv").dropna(subset=["LOI_pct", "HRC_J_gK"])
    b["source"], b["family"] = "AR-01/53", "thermoset"
    clash = sorted(set(a["polymer"]) & set(b["polymer"]))
    for p in clash:  # keep AR-05/14's value, report the other
        print(f"  CONFLICT {p}: AR-05/14 {a.loc[a.polymer == p, ['LOI_pct', 'HRC_J_gK']].values.tolist()} "
              f"vs AR-01/53 {b.loc[b.polymer == p, ['LOI_pct', 'HRC_J_gK']].values.tolist()}; kept AR-05/14")
    print(f"exact-name collisions between reports: {len(clash)}")
    u = pd.concat([a[COLS + ["family"]], b.loc[~b.polymer.isin(clash), COLS + ["family"]]], ignore_index=True)
    assert u["polymer"].is_unique and u["family"].notna().all()
    u["log10_HRC"] = np.log10(u["HRC_J_gK"])
    for c in ("char_yield_g_g", "hc0_kJ_g", "chi"):  # blank on purpose: run() then skips B/C/RF
        u[c] = np.nan
    return u


if __name__ == "__main__":
    u = load()
    ar0514, ar0153 = u[u.source == "AR-05/14"], u[u.source == "AR-01/53"]
    print(f"union: {len(u)} polymers ({len(ar0514)} AR-05/14 + {len(ar0153)} AR-01/53)")
    run(ar0514, "AR-05/14 only, config A (reproduces train_loi_v1)")
    run(u, "union AR-05/14 + AR-01/53, config A")

    feats, model = CONFIGS["A"]
    loo_r2, loo_mae, pred = held_out(model, u[feats].to_numpy(), u["LOI_pct"].to_numpy(), u["polymer"].to_numpy())
    new = u.source.eq("AR-01/53").to_numpy()
    print("\n  union leave-one-out predictions for the AR-01/53 rows:")
    print(pd.DataFrame({"polymer": u.polymer[new], "LOI": u.LOI_pct[new], "eta_c": u.HRC_J_gK[new],
                        "LOO_pred": pred[new].round(1)}).to_string(index=False))

    ext = clone(model).fit(ar0514[feats].to_numpy(), ar0514["LOI_pct"].to_numpy())
    y, p = ar0153["LOI_pct"].to_numpy(), ext.predict(ar0153[feats].to_numpy())
    ref = 125 / ar0153["HRC_J_gK"].to_numpy() ** 0.25
    print("\n  EXTERNAL: config A fit on 32 AR-05/14 rows -> 6 AR-01/53 rows")
    print(pd.DataFrame({"polymer": ar0153.polymer, "LOI": y, "pred_A": p.round(1), "formula": ref.round(1)})
          .to_string(index=False))
    for name, q in (("config A", p), ("formula 125/eta_c^0.25", ref)):
        print(f"  {name}: R2 {r2_score(y, q):.3f}  MAE {mean_absolute_error(y, q):.2f}  "
              f"RMSE {np.sqrt(np.mean((q - y) ** 2)):.2f}  (n=6, LOI sd {y.std(ddof=0):.2f})")
