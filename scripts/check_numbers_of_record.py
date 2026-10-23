#!/usr/bin/env python3
"""Compare a numbers-of-record repro run (data/v2/run_numbers_of_record.sh) with experiments/numbers_of_record.json.

Reads experiments/runs/repro_results/*.json (result JSONs the repro produced) and experiments/runs/repro_<stem>.log
(for TTI exp154, THR exp045 and the Pomazi pooled CIs, which are only printed). Every mean and CI bound must be within
TOL (0.001) of the recorded value. Prints a table; exit 1 on any mismatch or missing number, 0 if all match.
"""
import json
import re
import sys
from pathlib import Path

TOL = 0.001
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
RES = RUNS / "repro_results"
REC = json.load(open(ROOT / "experiments" / "numbers_of_record.json"))
NUM = r"(-?\d+\.\d+)"
rows = []  # (label, quantity, recorded, repro or None)


def add(label, qty, rec, got):
    rows.append((label, qty, rec, got))


def log(stem):
    p = RUNS / f"repro_{stem}.log"
    return p.read_text() if p.exists() else ""


def res(slug):
    p = RES / f"{slug}.json"
    return json.load(open(p)) if p.exists() else {}


def first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return [float(g) for g in m.groups()] if m else [None] * re.compile(pattern).groups


def pair(s):  # "0.448 [0.136, 0.578]" -> (0.448, 0.136, 0.578); "[0.382, 0.734]" -> (None, 0.382, 0.734)
    v = [float(x) for x in re.findall(NUM, s)]
    return (None, *v) if len(v) == 2 else tuple(v)


def check_triplet(label, rec, got):
    for q, r, g in zip(("mean", "ci_lo", "ci_hi"), rec, got):
        if r is not None:
            add(label, q, r, g)


# --- the six main targets -----------------------------------------------------------------------------
for tgt, v in REC["targets"].items():
    exp = v["exp_id"]
    rec = (v["value"], v["ci_lo"], v["ci_hi"])
    if exp in ("exp158", "exp157", "exp163", "exp119"):  # exp163 = 4-stage (chemfix + unitfix) rescore, slug keeps snap606
        d = res({"exp158": "fsri_composite_snapshot_phrr", "exp157": "fsri_composite_snapshot_cr",
                 "exp163": "loi_snap606_exp113cfg", "exp119": "blend_r1_figra_b1"}[exp])
        got = (d.get("rkf_mean"), d.get("ci_lo"), d.get("ci_hi"))
    elif exp == "exp154":  # train_tti_round3 --record-only: exp046 pipeline on S675 (STEP 2)
        got = first(r"STEP 2: exp046 on S675: rkf_mean " + NUM + r" \+- [\d.]+; repeat-0 CI \[" + NUM + r", " + NUM + r"\]",
                    log("train_tti_round3"))
    elif exp == "exp045":
        got = first(r"A3 headline: " + NUM + r" \[" + NUM + r", " + NUM + r"\]", log("train_thr_v1_replicates"))
    else:
        raise SystemExit(f"no repro mapping for {tgt} {exp}")
    check_triplet(f"{tgt} {exp}", rec, got)

# --- supplementary LOI numbers: proxy-free (snapshot 3 + chemfix, exp114 config) and given-control (round 7, snapshot 1)
SUPP = {"exp164": "loi_snap606_exp114cfg_proxyfree", "exp100": "loi_r7_d1_delta_ctrl"}
for key in ("proxy_free", "delta_from_control"):
    v = REC["targets"]["LOI"].get(key)
    if v:
        d = res(SUPP[v["exp_id"]])
        check_triplet(f"LOI {key} {v['exp_id']}", (v["value"], v["ci_lo"], v["ci_hi"]),
                      (d.get("rkf_mean"), d.get("ci_lo"), d.get("ci_hi")))

# --- Pomazi reconstruction (exp076-079): primary LOSO mean+CI, pooled GKF CI, protocol A mean -------------
PSLUG = {"pHRR": ("pomazi_r1b_phrr_kWm2_rf", "phrr_kWm2"), "Char Residue": ("pomazi_r1b_residue_pct_hgb", "residue_pct"),
         "THR": ("pomazi_r1b_thr_MJm2_gbr_theirs", "thr_MJm2"), "TTI": ("pomazi_r1b_tti_s_rf", "tti_s")}
plog = log("train_pomazi_round1b")
for tgt, v in REC["pomazi_recon"].items():
    if tgt.startswith("_"):
        continue
    slug, key = PSLUG[tgt]
    label = f"Pomazi {tgt} {v['exp_id']}"
    d = res(slug)
    check_triplet(label + " LOSO", pair(v["primary_resin_system_holdout"]), (d.get("rkf_mean"), d.get("ci_lo"), d.get("ci_hi")))
    add(label, "protocol A mean", v["their_protocol_A"], d.get("gss_mean"))
    # SELECTED table row: key cand rows/grp PRIMARY [lo, hi] SEC±std [rep0] [pooled lo, hi] A mean [...]
    row = re.search(rf"^{key}\s+\S+\s+\d+/\d+\s+{NUM} \[\s*{NUM},\s*{NUM}\]\s+{NUM}±{NUM}\s+\[\s*{NUM},\s*{NUM}\]\s+"
                    rf"\[\s*{NUM},\s*{NUM}\]", plog, re.M)
    plo, phi = (float(row.group(8)), float(row.group(9))) if row else (None, None)
    _, rlo, rhi = pair(v["formulation_grouped_pooled_ci"])
    add(label, "pooled GKF ci_lo", rlo, plo)
    add(label, "pooled GKF ci_hi", rhi, phi)

bad = 0
w = max(len(r[0]) for r in rows)
print(f"\nNumbers of record vs repro (tolerance {TOL})")
print(f"{'target':<{w}}  {'quantity':<17} {'record':>8} {'repro':>8} {'diff':>8}  result")
for label, qty, rec, got in rows:
    if got is None:
        ok, diff = False, "n/a"
    else:
        ok, diff = abs(got - rec) <= TOL + 1e-9, f"{got - rec:+.4f}"
    bad += not ok
    g = "missing" if got is None else f"{got:.4f}"
    print(f"{label:<{w}}  {qty:<17} {rec:>8.4f} {g:>8} {diff:>8}  {'ok' if ok else 'MISMATCH'}")
print(f"\n{len(rows) - bad}/{len(rows)} numbers match" + ("" if not bad else f"; {bad} MISMATCH/missing"))
sys.exit(1 if bad else 0)
