#!/usr/bin/env python3
"""Dravix V2 -- A2: MaCFP-4 Wood + Edinburgh PMMA external cross-lab check (report-only).

Follows experiments/configs/macfp_wood_crosslab.json exactly (committed dd9c9c6, before this first run).
Report-only external validation: no training, tuning, selection or adoption on any MaCFP row; no number of
record can change. pHRR record exp074 and TTI record exp046 are scored per test; Char Residue record exp117 is a
diagnostic only (absolute error, no R2).

Procedure (config 'procedure'):
  0. pin check, file and README-row counts (row_sets), MCC-algorithm self-test
  1. reproduce exp074, exp046, exp117 on FSRI (step1_reproduce)
  2. PS check and gate (label-free)
  3. derive and freeze every MaCFP and Edinburgh input; sanity gates; label-free reports (no label read yet)
  4. predict all variants for all rows; write the predictions
  5. read the labels (HRR max, README t_ign, mass channel); compute every metric and table; write outputs
  (6. fresh-context critic pass: orchestrator, before any number is quoted)

Output location: the config names experiments/results/macfp_wood_crosslab/. The orchestrator for this run forbids
writing into experiments/results/, so the four declared files (ps_check.json, features.json, tests.csv,
summary.json) go to experiments/runs/macfp_wood_crosslab/ instead. File names and contents are as declared.
The markdown report is reports/macfp_wood_crosslab.md (tables generated here; a hand-written summary between the
summary markers is preserved across reruns).

Run: OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 data/v2/validate_macfp_wood_crosslab.py \
       > experiments/runs/macfp_wood_crosslab.log
"""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from eval_protocol import bootstrap_ci, repeated_group_kfold  # noqa: E402

CONFIG = "experiments/configs/macfp_wood_crosslab.json"
PIN_COMMIT = "0e4ad8bf6ddb7c52eea4022aade810672adabfd0"
PIN = os.path.join(HERE, "raw", "matl-db-0e4ad8b")
WOOD = os.path.join(PIN, "Wood", "Calibration_Data")
EDIN = os.path.join(PIN, "PMMA", "Calibration_Data", "Edinburgh")
FSRI_SELFTEST = os.path.join(HERE, "raw", "fsri_materials_database", "01_Data", "Pine_Siding", "MCC",
                             "Pine_Siding_30K_min_210608_R1.txt")
OUT = os.path.join(ROOT, "experiments", "runs", "macfp_wood_crosslab")
REPORT = os.path.join(ROOT, "reports", "macfp_wood_crosslab.md")
MODEL_PHRR = os.path.join(ROOT, "models", "v2", "phrr_hgb_v4.joblib")

PHRR_FEATS = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "cone_mass_per_area_kgm2",
              "cone_thickness_mm", "density_kgm3", "mcc_total_hr_kJg", "mcc_t_peak_C", "mcc_t_onset_C", "log_kpc"]
TTI_FEATS = ["log_q", "log_dT", "log_kpc", "log_L", "log_rho"]
CR_FEATS = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct", "sta_res_3K", "sta_res_10K", "sta_res_30K",
            "cat_interior_finish", "cat_exterior_finish", "cat_insulation", "cat_roofing", "cat_structural"]
T0_C = 20.0

PRIMARY_LABS = ["Aalto", "FPL", "FSRI", "FZJ", "IMT", "TUBS", "UDRI", "UQ"]
# Per-lab cone metadata (config 'labs'); column names are README headers with backslashes removed.
LABS = {
    "Aalto": dict(glob="Aalto_Wood_Cone_*kW_hor_parallel_R*.csv", L=("Initial thickness (mm)", 1.0),
                  planform=("cross", "Planform (mm × mm)"), flameout="t_flameout (s)", A=0.01,
                  orientation="parallel", frame="no", sampling_s=5),
    "BUWfd": dict(glob="BUWfd_Wood_Cone_*kW_hor_R*.csv", exclude=["BUWfd_Wood_Cone_50kW_hor_R1"], A=None,
                  orientation="unknown", frame="n/a", sampling_s=None),
    "FPL": dict(glob="FPL_Wood_Cone_50kW_hor_R*.csv", L=("thickness (cm)", 10.0), planform=("nominal", 0.01),
                flameout="t_flameout (s)", A=0.00884, orientation="unknown", frame="yes", sampling_s=1),
    "FSRI": dict(glob="FSRI_Wood_Cone_*kW_hor_*_R*.csv", L=("Sample Thickness (mm)", 1.0),
                 planform=("two", "Sample Length (mm)"), flameout="Time to Flame Out (s)", A=0.01,
                 orientation="file", frame="no (tie wires)", sampling_s=1),
    "FZJ": dict(glob="FZJ_Wood_Cone_*kW_hor_*_R*.csv", L=("Thickness (mm)", 1.0),
                planform=("cm2", "Surface area (cm^2)"), flameout=None, A=0.01, orientation="file", frame="no",
                sampling_s=1),
    "IMT": dict(glob="IMT_Wood_Cone_*kW_hor_R*.csv", L=("Thickness (cm)", 10.0), planform=("nominal", 0.01),
                flameout="t_flameout (s)", A=0.00884, orientation="unknown", frame="yes", sampling_s=5),
    "TUBS": dict(glob="TUBS_Wood_Cone_30kW_hor_R*.csv", L=("Thickness (cm)", 10.0), planform=("nominal", 0.01),
                 flameout=None, A=0.00884, orientation="readme", frame="yes", sampling_s=1),
    "UDRI": dict(glob="UDRI_Wood_Cone_*kW_hor_R*.csv", L=("Thickness (cm)", 10.0), planform=("nominal", 0.01),
                 flameout="t_flameout (s)", A=0.01, orientation="unknown", frame="no", sampling_s=1),
    "UMET": dict(glob="UMET_Wood_Cone_*kW_hor_R*.csv", L=("Thickness (cm)", 10.0), planform=("umet", None),
                 flameout="t_flameout (s)", A=0.00884, orientation="unknown", frame="no", sampling_s=1),
    "UQ": dict(glob="UQ_Wood_Cone_60kW_hor_Parallel_R*.CSV", L=("uq_bullet", None), planform=("nominal", 0.01),
               flameout="t_flameout (s)", A=0.01, orientation="parallel", frame="no", sampling_s=1),
}
EXPECT_FILES = {"Aalto": 9, "BUWfd": 5, "FPL": 6, "FSRI": 18, "FZJ": 22, "IMT": 12, "TUBS": 3, "UDRI": 7, "UMET": 6,
                "UQ": 3}
EXPECT_PRIMARY = {"Aalto": 9, "FPL": 6, "FSRI": 18, "FZJ": 22, "IMT": 12, "TUBS": 3, "UDRI": 7, "UQ": 3}
EXPECT_FLUX = {25: 11, 30: 15, 45: 3, 50: 16, 60: 19, 75: 16}
EXPECT_ORIENT = {"parallel": 34, "perpendicular": 21, "unknown": 25}
EXPECT_CR = {"Aalto": 5, "FPL": 5, "FSRI": 18, "IMT": 12, "UDRI": 7, "UMET": 3, "UQ": 3}
EXPECT_EOT = {"Aalto": 9, "BUWfd": 5, "FPL": 6, "FSRI": 18, "FZJ": 20, "IMT": 12, "TUBS": 3, "UDRI": 7, "UMET": 6,
              "UQ": 3}
STOP_RULE = {  # what each README says about when a cone test ends (read Oct 6; none states a stop criterion)
    "Aalto": "not stated for cone-in-air tests; t_flameout reported (from start of DAQ); 'Test duration' filled only for N2 gasification",
    "BUWfd": "README has no cone metadata",
    "FPL": "not stated (ASTM E1354-25); t_flameout reported; R2 DAQ lost at 1400 s, flameout recorded by hand",
    "FSRI": "not stated (ASTM E1354, general accordance); flameout by operator button",
    "FZJ": "not stated (ISO 5660); no flameout column",
    "IMT": "not stated (ISO 5660); t_flameout reported",
    "TUBS": "not stated (ISO 5660); t_flameout '-'",
    "UDRI": "not stated (ASTM E1354-25); t_flameout reported",
    "UMET": "not stated (ISO 13927 mass-loss cone); t_flameout reported",
    "UQ": "not stated (ISO 5660); t_flameout reported; README: 60 kW/m2 mass readings fluctuated",
}

STATE = {"blocked": [], "flags": [], "targets_blocked": {}}
T_START = time.time()


def log(*a):
    print(*a, flush=True)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def jsonable(o):
    """NaN/inf -> None, numpy scalars -> Python, recursively (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2, allow_nan=False, default=str)


def write_summary(extra=None):
    os.makedirs(OUT, exist_ok=True)
    status = "blocked" if STATE["blocked"] else ("flag" if STATE["flags"] else "ok")
    s = dict(round="macfp_wood_crosslab", config=CONFIG, script="data/v2/validate_macfp_wood_crosslab.py",
             status=status, blocked=STATE["blocked"], flags=STATE["flags"], targets_blocked=STATE["targets_blocked"],
             report_only=True, note="External cross-lab check, MaCFP-4 pine (one material, 8 oxygen-consumption labs), "
                                    "report-only. No number of record changes; nothing is compared with a bar.")
    s.update(extra or {})
    dump_json(s, os.path.join(OUT, "summary.json"))
    return s


def block(msg):
    """Whole-item BLOCKED: write status 'blocked' and stop."""
    STATE["blocked"].append(msg)
    log(f"BLOCKED: {msg}")
    write_summary()
    sys.exit(2)


# ---------------------------------------------------------------------------------------------------------------
# README parsing. normalize_name / normalize_header copied verbatim from MaCFP Scripts/MaCFP-4/Cone_analysis.py @ pin
# ---------------------------------------------------------------------------------------------------------------
def normalize_header(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def normalize_name(text):
    text = text.replace('\\', '').replace('.csv', '').lower().strip()
    text = text.replace('-', '_')
    text = text.replace('imt_mines_ales', 'imt')

    if text.startswith('umet_mlc_'):
        text = text.replace('umet_mlc_', 'umet_cone_').replace('_r', '_hor_r')

    parts = [p for p in text.split('_') if p != 'wood']
    orientation = next((p for p in parts if p in ['parallel', 'perpendicular']), None)
    parts = [p for p in parts if p not in ['parallel', 'perpendicular']]

    if orientation is not None:
        parts.append(orientation)

    return '_'.join(parts)


def macfp_value(value):
    """MaCFP's cell rule: '', '-', en/em dash -> NaN; float() or NaN."""
    value = value.strip()
    if value in ['', '-', '–', '—']:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def readme_tables(path):
    """Markdown tables: [(line_no, header_cells, [row_cells, ...])]. A header is a '|' line followed by a separator."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    tables, i = [], 0
    sep = re.compile(r"^:?-+:?$")
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|") and i + 1 < len(lines):
            nxt = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            if nxt and all(sep.match(c) for c in nxt if c):
                hdr = [c.strip() for c in ln.strip("|").split("|")]
                rows, j = [], i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                    j += 1
                tables.append((i + 1, hdr, rows))
                i = j
                continue
        i += 1
    return tables, lines


def name_col(hdr):
    """Test-name column under either MaCFP rule (ignition or initial-mass function)."""
    nh = [normalize_header(h) for h in hdr]
    return next((j for j, h in enumerate(nh) if 'testname' in h or 'filetestname' in h or 'runlabel' in h
                 or h in ['experiment', 'experimentname', 'runlabel']), None)


def mass_col(hdr):  # get_reported_initial_mass header rule
    nh = [normalize_header(h) for h in hdr]
    return next((j for j, h in enumerate(nh) if h.startswith('initialmass') or h.startswith('initialsamplemass')
                 or h == 'massg'), None)


def ign_col(hdr):  # get_reported_ignition_time header rule
    nh = [normalize_header(h) for h in hdr]
    return next((j for j, h in enumerate(nh) if h.startswith('tign') or h.startswith('ignitiontime')
                 or h.startswith('timetoignition')), None)


def clean_hdr(h):
    return h.replace("\\", "").strip()


def match_readme_row(tables, stem):
    target = normalize_name(stem)
    hits = []
    for ln, hdr, rows in tables:
        nc = name_col(hdr)
        if nc is None:
            continue
        for r in rows:
            if len(r) > nc and normalize_name(r[nc]) == target:
                hits.append((ln, hdr, r))
    return hits


def cell(hdr, row, col_name):
    idx = [j for j, h in enumerate(hdr) if clean_hdr(h) == col_name]
    if len(idx) != 1:
        return None, f"column '{col_name}' found {len(idx)}x"
    j = idx[0]
    return (row[j] if j < len(row) else ""), None


# ---------------------------------------------------------------------------------------------------------------
# MCC curve algorithm: copied from data/v2/ingest_fsri_props.ingest_mcc_extras (baseline-corrected part + peak)
# ---------------------------------------------------------------------------------------------------------------
def to_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def read_mcc(path):
    """-> (t, T_C, q, header_rate_C_s or None). Time (s), Temperature (C or K; K->C), HRR (W/g); NaN rows dropped."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    head, start = {}, None
    if path.lower().endswith(".txt"):  # Deatak text export (FSRI '*' marker, NIST '@' marker)
        for i, line in enumerate(lines):
            if line.strip() in ("*", "@"):
                start = i + 1
                break
            if ":" in line:
                k, _, v = line.partition(":")
                head[k.strip()] = v.strip()
        if start is None:
            raise ValueError("no '*'/'@' marker")
        cols = [c.strip() for c in lines[start].split("\t")]
        body = [ln.split("\t") for ln in lines[start + 1:]]
    else:
        rdr = list(csv.reader(lines))
        cols = [c.strip() for c in rdr[0]]
        body = rdr[1:]
    try:
        it = cols.index("Time (s)")
        ih = cols.index("HRR (W/g)")
    except ValueError:
        raise ValueError(f"missing Time/HRR column in {cols}")
    if "Temperature (C)" in cols:
        iT, kelvin = cols.index("Temperature (C)"), False
    elif "Temperature (K)" in cols:
        iT, kelvin = cols.index("Temperature (K)"), True
    else:
        raise ValueError(f"missing Temperature column in {cols}")
    data = []
    for parts in body:
        if len(parts) > max(it, iT, ih):
            vals = [to_float(parts[j]) for j in (it, iT, ih)]
            if None not in vals:
                data.append(vals)
    if len(data) < 10:
        raise ValueError(f"only {len(data)} complete rows")
    t, T, q = np.array(data).T
    if kelvin:
        T = T - 273.15
    return t, T, q, to_float(head.get("Heating Rate (C/s)"))


def mcc_curve(t, T, q):
    ipk = int(np.argmax(q))
    edge = max(len(q) // 20, 1)
    qb = q - np.interp(t, [t[0], t[-1]], [np.median(q[:edge]), np.median(q[-edge:])])
    cum_bc = np.concatenate([[0.0], np.cumsum(np.diff(t) * (qb[1:] + qb[:-1]) / 2)])
    total_bc = cum_bc[-1] / 1000.0
    onset_bc = None
    if total_bc >= 0.5:
        onset_bc = float(T[np.argmax(cum_bc >= 0.05 * cum_bc[-1])])
    return dict(peak_raw=float(q[ipk]), peak_hrr_Wg=round(float(q[ipk]), 3), t_peak_C=round(float(T[ipk]), 1),
                total_hr_bc_kJg=round(total_bc, 3),
                t_onset_bc_C=round(onset_bc, 1) if onset_bc is not None else None, n_rows=len(q))


def nominal_beta(fname):
    m = re.search(r"_(\d+)K_", fname)
    return int(m.group(1)) / 60.0 if m else None


def mcc_run(path):
    t, T, q, rate = read_mcc(path)
    r = mcc_curve(t, T, q)
    beta = rate if rate else nominal_beta(os.path.basename(path))
    r.update(beta_C_s=beta, beta_src="header" if rate else "file name", hrc_JgK=round(r["peak_raw"] / beta, 2),
             file=os.path.relpath(path, PIN))
    return r


# ---------------------------------------------------------------------------------------------------------------
# helpers for metrics/tables
# ---------------------------------------------------------------------------------------------------------------
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(r2_score(y, p)) if len(y) >= 2 else float("nan")


def log_r2(y, p):  # natural log of seconds (as tti_secondary_metrics.json LOG_ALL)
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(r2_score(np.log(y), np.log(np.clip(p, 1e-6, None)))) if len(y) >= 2 else float("nan")


def lab_ci(y, p, labs):
    lo, hi = bootstrap_ci(np.asarray(y, float), np.asarray(p, float), np.asarray(labs), n_boot=2000, seed=0)
    return float(lo), float(hi)


def f1(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}"


def f4(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}"


def f2(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def lab_label(lab):
    return "UQ (flag: README mass fluctuations)" if lab == "UQ" else lab


def group_table(d, key, obs, pred, tti=False, order=None):
    rows = []
    keys = order if order is not None else sorted(d[key].unique(), key=lambda v: (str(type(v)), v))
    for k in keys:
        g = d[d[key] == k]
        if g.empty:
            continue
        o, p = g[obs].to_numpy(float), g[pred].to_numpy(float)
        samp = "/".join(str(int(s)) for s in sorted(g["sampling_s"].dropna().unique()))
        row = {key: lab_label(k) if key == "lab" else (f"{k:g}" if isinstance(k, (int, float, np.number)) else k),
               "n": len(g), "mean obs": f1(o.mean()), "mean pred": f1(p.mean()), "MAE": f1(np.mean(np.abs(p - o))),
               "bias (pred-obs)": f1(np.mean(p - o)), "mean obs / mean pred": f2(o.mean() / p.mean()),
               "sampling (s)": samp}
        if key != "lab" and "UQ" in set(g["lab"]):
            row[key] = f"{row[key]} (incl. UQ, flagged)"
        if tti:
            row["log R2"] = f4(log_r2(o, p))
            row["median abs(pred-obs)/obs (%)"] = f1(100 * np.median(np.abs(p - o) / o))
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_table(d, obs, pred):
    rows = []
    for lab in ["FSRI", "FZJ"]:
        for q in sorted(d.loc[d.lab == lab, "flux"].unique()):
            g = d[(d.lab == lab) & (d.flux == q)]
            pe, pa = g[g.orientation == "perpendicular"], g[g.orientation == "parallel"]
            if pe.empty or pa.empty:
                continue
            rows.append({"lab": lab, "flux": f"{q:g}", "n perp/par": f"{len(pe)}/{len(pa)}",
                         "obs perp-par": f1(pe[obs].mean() - pa[obs].mean()),
                         "pred perp-par": f1(pe[pred].mean() - pa[pred].mean())})
    return pd.DataFrame(rows)


# ===============================================================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"A2 MaCFP-4 Wood + Edinburgh PMMA cross-lab check; config {CONFIG}")
    log(f"python {sys.version.split()[0]}; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}; "
        f"PYTHONDONTWRITEBYTECODE={os.environ.get('PYTHONDONTWRITEBYTECODE')}")
    import sklearn
    import scipy
    log(f"sklearn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}")
    env_ok = (sys.version.split()[0], sklearn.__version__, np.__version__, pd.__version__, scipy.__version__) == \
             ("3.12.4", "1.7.1", "2.2.6", "2.3.3", "1.16.1")
    log(f"environment matches the config: {env_ok}")
    try:
        cfg_commit = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%H %ci", "--", CONFIG],
                                    capture_output=True, text=True).stdout.strip()
        log(f"config last committed: {cfg_commit}")
    except Exception as e:  # informational only
        log(f"(could not read config commit: {e})")

    # =========================================== STEP 0 ===========================================
    log("\n========== STEP 0: pin check, file and README-row counts, MCC self-test ==========")
    head = subprocess.run(["git", "-C", PIN, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", PIN, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    log(f"pin HEAD {head}; tree clean: {dirty == ''}")
    if head != PIN_COMMIT or dirty:
        block(f"pin mismatch (HEAD {head}, dirty={bool(dirty)})")

    # ---- cone files
    files = {}
    for lab, spec in LABS.items():
        fl = sorted(glob.glob(os.path.join(WOOD, lab, spec["glob"])))
        fl = [f for f in fl if "Gasification" not in f and
              not any(os.path.splitext(os.path.basename(f))[0] == x for x in spec.get("exclude", []))]
        files[lab] = fl
    counts = {k: len(v) for k, v in files.items()}
    log(f"cone files per lab: {counts}")
    if counts != EXPECT_FILES:
        block(f"cone file count mismatch: {counts} (expected {EXPECT_FILES})")
    assert not any("BUWfd_Wood_Cone_50kW_hor_R1" in f for f in files["BUWfd"])
    log(f"UCantabria: excluded everywhere (README-only rows, no CSV); BUWfd_Wood_Cone_50kW_hor_R1 excluded "
        f"(non-physical mass; MaCFP's script excludes it)")

    # ---- README rows
    tests, labels_readme = [], {}
    for lab in LABS:
        spec = LABS[lab]
        readme = os.path.join(WOOD, lab, "README.md")
        tables, lines = readme_tables(readme)
        uq_L = umet_planform = None
        if lab == "UQ":
            vals = re.findall(r"Sample Initial Thickness:\s*([\d.]+)\s*cm", "\n".join(lines))
            if not vals or any(float(v) != 2.53 for v in vals):
                block(f"UQ README 'Sample Initial Thickness' not 2.53 cm: {vals}")
            uq_L = 25.3
        if lab == "UMET":
            txt = "\n".join(lines)
            sec = txt[txt.index("Experimental Conditions: Mass Loss Cone"):]
            m = re.search(r"Solid size\s*:\s*([\d.]+)mm\\?\*([\d.]+)mm", sec)
            if not m or (float(m.group(1)), float(m.group(2))) != (100.0, 100.0):
                block(f"UMET README 'Solid size' not 100mm*100mm: {m.group(0) if m else None}")
            umet_planform = float(m.group(1)) * float(m.group(2)) * 1e-6
        for path in files[lab]:
            stem = os.path.splitext(os.path.basename(path))[0]
            flux = int(re.search(r"_(\d+)kW_", stem).group(1))
            if spec["orientation"] == "file":
                orient = "perpendicular" if "perpendicular" in stem.lower() else "parallel"
                assert "parallel" in stem.lower() or "perpendicular" in stem.lower()
            else:
                orient = spec["orientation"]
            t = dict(lab=lab, file=os.path.relpath(path, PIN), stem=stem, flux=flux, orientation=orient,
                     stratum="UMET" if lab == "UMET" else ("BUWfd_end_of_test_only" if lab == "BUWfd" else "primary"),
                     frame=spec["frame"], declared_sampling_s=spec["sampling_s"], A_norm_m2=spec["A"])
            if lab == "BUWfd":
                tests.append(t)
                continue
            hits = match_readme_row(tables, stem)
            if len(hits) != 1:
                block(f"{stem}: matched {len(hits)} README rows (need exactly 1)")
            ln, hdr, row = hits[0]
            mi, ii = mass_col(hdr), ign_col(hdr)
            if mi is None or ii is None:
                block(f"{stem}: README table (line {ln}) lacks a MaCFP initial-mass or t_ign column")
            m0 = macfp_value(row[mi]) if mi < len(row) else np.nan
            tign = macfp_value(row[ii]) if ii < len(row) else np.nan
            Lname, Lfac = spec["L"]
            if Lname == "uq_bullet":
                L = uq_L
            else:
                v, err = cell(hdr, row, Lname)
                if err:
                    block(f"{stem}: {err}")
                L = macfp_value(v) * Lfac
            kind, arg = spec["planform"]
            if kind == "nominal":
                plan = arg
            elif kind == "umet":
                plan = umet_planform
            elif kind == "cm2":
                v, err = cell(hdr, row, arg)
                if err:
                    block(f"{stem}: {err}")
                plan = macfp_value(v) * 1e-4
            elif kind == "two":
                idx = [j for j, h in enumerate(hdr) if clean_hdr(h) == arg]
                if len(idx) != 2:
                    block(f"{stem}: expected two '{arg}' columns, found {len(idx)}")
                plan = macfp_value(row[idx[0]]) * macfp_value(row[idx[1]]) * 1e-6
            elif kind == "cross":
                v, err = cell(hdr, row, arg)
                if err:
                    block(f"{stem}: {err}")
                m = re.fullmatch(r"\s*([\d.]+)\s*×\s*([\d.]+)\s*", v)
                plan = float(m.group(1)) * float(m.group(2)) * 1e-6 if m else np.nan
            if lab == "TUBS":
                v, err = cell(hdr, row, "Grain orientation (-)")
                if err or v.strip().lower() != "parallel":
                    block(f"{stem}: TUBS README grain orientation is not 'parallel' ({v}, {err})")
                orient = t["orientation"] = "parallel"
            tfo = np.nan
            if spec["flameout"]:
                v, err = cell(hdr, row, spec["flameout"])
                if err:
                    block(f"{stem}: {err}")
                tfo = macfp_value(v)
            missing = [n for n, v in (("m0", m0), ("L", L), ("planform", plan), ("t_ign", tign)) if not np.isfinite(v)]
            if missing:
                block(f"{stem}: README row (line {ln}) lacks required values {missing}")
            t.update(m0_g=m0, L_mm=L, planform_m2=plan, readme_mass_header=hdr[mi], readme_tign_header=hdr[ii],
                     readme_line=ln)
            labels_readme[stem] = dict(t_ign_s=tign, t_flameout_s=tfo)
            tests.append(t)
    T = pd.DataFrame(tests)
    log(f"README rows matched: {len(labels_readme)} cone tests, each to exactly one README row; "
        f"mass headers used: {sorted(set(T.readme_mass_header.dropna()))}; "
        f"t_ign headers used: {sorted(set(T.readme_tign_header.dropna()))}")

    # Edinburgh README (12 rows)
    etab, _ = readme_tables(os.path.join(EDIN, "README.md"))
    ed = []
    for ln, hdr, rows in etab:
        nc, ic = name_col(hdr), ign_col(hdr)
        if nc is None or ic is None:
            continue
        fc = [j for j, h in enumerate(hdr) if clean_hdr(h) == "Heat Flux (kW/m2)"]
        assert len(fc) == 1
        for r in rows:
            ed.append(dict(lab="Edinburgh_PMMA", stem=r[nc].strip(), flux=int(macfp_value(r[fc[0]])),
                           stratum="Edinburgh_PMMA", t_ign_s=macfp_value(r[ic])))
    E = pd.DataFrame(ed)
    if len(E) != 12 or E.flux.value_counts().to_dict() != {25: 6, 65: 6} or not np.isfinite(E.t_ign_s).all():
        block(f"Edinburgh README rows: {len(E)} ({E.flux.value_counts().to_dict() if len(E) else {}})")
    log(f"Edinburgh PMMA README rows: {len(E)} (25 kW/m2: 6, 65 kW/m2: 6)")

    # ---- cone CSV records (finiteness / time only here; no HRR or mass value is used until step 5)
    data = {}
    for _, t in T.iterrows():
        df = pd.read_csv(os.path.join(PIN, t.file), encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        for c in ("Time (s)", "Mass (g)", "HRR (kW/m2)"):
            if c not in df.columns:
                block(f"{t.stem}: missing column {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        data[t.stem] = df
    fin_mass, last_mass_t, last_t, samp = {}, {}, {}, {}
    for stem, df in data.items():
        tt = df["Time (s)"].to_numpy(float)
        ok = np.isfinite(tt) & np.isfinite(df["Mass (g)"].to_numpy(float))
        fin_mass[stem] = int(ok.sum())
        last_mass_t[stem] = float(tt[ok][-1]) if ok.any() else np.nan
        tf = tt[np.isfinite(tt)]
        last_t[stem] = float(tf[-1])
        samp[stem] = float(np.median(np.diff(tf)))
    T["sampling_s"] = T.stem.map(samp).round(3)
    T["has_finite_mass"] = T.stem.map(lambda s: fin_mass[s] > 0)
    bad_samp = T[(T.declared_sampling_s.notna()) & (np.abs(T.sampling_s - T.declared_sampling_s) > 0.01)]
    log(f"measured median sampling interval per lab: "
        f"{T.groupby('lab').sampling_s.agg(lambda s: sorted(set(s))).to_dict()}; mismatches with declared: {len(bad_samp)}")

    # ---- row sets and count rule
    prim = T[T.stratum == "primary"]
    c_lab = prim.lab.value_counts().to_dict()
    c_flux = prim.flux.value_counts().to_dict()
    c_or = prim.orientation.value_counts().to_dict()
    log(f"primary_80: {len(prim)}; by lab {dict(sorted(c_lab.items()))}; by flux {dict(sorted(c_flux.items()))}; "
        f"by orientation {c_or}")
    tti25 = prim[prim.flux == 25]
    T["in_cr_flameout"] = False
    for i, t in T.iterrows():
        if t.lab == "BUWfd":
            continue
        tfo = labels_readme[t.stem]["t_flameout_s"]
        T.loc[i, "in_cr_flameout"] = bool(np.isfinite(tfo) and np.isfinite(last_mass_t[t.stem])
                                          and tfo <= last_mass_t[t.stem])
    c_cr = T[T.in_cr_flameout].lab.value_counts().to_dict()
    c_eot = T[T.has_finite_mass].lab.value_counts().to_dict()
    log(f"split: FSRI-lab {int((prim.lab == 'FSRI').sum())} vs other labs {int((prim.lab != 'FSRI').sum())} "
        f"({prim[prim.lab != 'FSRI'].lab.nunique()} labs)")
    log(f"tti_25: {len(tti25)} {tti25.lab.value_counts().to_dict()}; umet_stratum: {int((T.lab == 'UMET').sum())}; "
        f"tti_86_secondary: {len(prim) + int((T.lab == 'UMET').sum())}; edinburgh_pmma: {len(E)}")
    log(f"cr_flameout: {sum(c_cr.values())} {dict(sorted(c_cr.items()))}")
    log(f"end_of_test: {sum(c_eot.values())} {dict(sorted(c_eot.items()))}")
    problems = []
    if c_lab != EXPECT_PRIMARY:
        problems.append(f"primary per lab {c_lab}")
    if c_flux != EXPECT_FLUX:
        problems.append(f"primary per flux {c_flux}")
    if c_or != EXPECT_ORIENT:
        problems.append(f"primary per orientation {c_or}")
    if (prim.lab == "FSRI").sum() != 18 or (prim.lab != "FSRI").sum() != 62:
        problems.append("FSRI/other split")
    if tti25.lab.value_counts().to_dict() != {"FSRI": 6, "IMT": 5}:
        problems.append(f"tti_25 {tti25.lab.value_counts().to_dict()}")
    if (T.lab == "UMET").sum() != 6:
        problems.append("UMET stratum")
    if c_cr != EXPECT_CR:
        problems.append(f"cr_flameout {c_cr}")
    if c_eot != EXPECT_EOT:
        problems.append(f"end_of_test {c_eot}")
    if problems:
        block("count mismatch: " + "; ".join(problems))
    log("count rule: all row-set counts match the config")

    # ---- MCC self-test
    st = mcc_curve(*read_mcc(FSRI_SELFTEST)[:3])
    ref = pd.read_csv(os.path.join(HERE, "mcc_extras_fsri.csv"))
    ref = ref[(ref.material == "Pine_Siding") & (ref.replicate == 1) & (ref.test_date == 210608)].iloc[0]
    cmp_ = {k: (st[k], float(ref[k])) for k in ("peak_hrr_Wg", "t_peak_C", "total_hr_bc_kJg", "t_onset_bc_C")}
    log(f"MCC self-test on Pine_Siding_30K_min_210608_R1 (computed, stored): {cmp_}")
    if any(abs(a - b) > 1e-9 for a, b in cmp_.values()):
        block(f"MCC algorithm self-test failed: {cmp_}")
    log("MCC self-test: reproduced at stored rounding")

    # =========================================== STEP 1 ===========================================
    log("\n========== STEP 1: reproduce exp074, exp046, exp117 on FSRI ==========")
    import train_blend_round1 as B
    step1 = {}

    def repro(name, model, X, y, g, logt, ref3):
        t0 = time.time()
        rk, oof = repeated_group_kfold(model, X, y, g, log_target=logt)
        lo, hi = bootstrap_ci(y, oof[0], g)
        ok = abs(rk.mean() - ref3[0]) <= 0.002 and abs(lo - ref3[1]) <= 0.002 and abs(hi - ref3[2]) <= 0.002
        log(f"{name}: {len(y)} rows / {len(np.unique(g))} groups; GKF 5x10 {rk.mean():.4f} CI [{lo:.4f}, {hi:.4f}] "
            f"(record {ref3[0]} [{ref3[1]}, {ref3[2]}]) -> {'REPRODUCED' if ok else 'NOT REPRODUCED'} ({time.time()-t0:.0f}s)")
        step1[name] = dict(mean=round(float(rk.mean()), 4), ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                           record=list(ref3), reproduced=bool(ok), n_rows=int(len(y)), n_groups=int(len(np.unique(g))))
        return ok

    Dp = B.data_phrr()
    assert Dp["feats"] == PHRR_FEATS and len(np.unique(Dp["g"])) == 64
    ok_p = repro("exp074", Dp["inc"], Dp["X"], Dp["y"], Dp["g"], True, (0.6113, 0.5155, 0.8376))
    Dt = B.data_tti()
    assert Dt["feats"] == TTI_FEATS and len(np.unique(Dt["g"])) == 76 and len(Dt["y"]) == 681
    ok_t = repro("exp046", Dt["inc"], Dt["X"], Dt["y"], Dt["g"], False, (0.3133, 0.0715, 0.4958))
    Dc = B.data_cr()
    assert Dc["feats"] == CR_FEATS and len(Dc["y"]) == 567 and len(np.unique(Dc["g"])) == 61
    cr_model = B.Avg([Dc["inc"], B.et(), B.ridge()])
    ok_c = repro("exp117", cr_model, Dc["X"], Dc["y"], Dc["g"], False, (0.6504, 0.4389, 0.7893))

    bundle = joblib.load(MODEL_PHRR)
    if list(bundle["features"]) != PHRR_FEATS:
        ok_p = False
        log(f"exp074 joblib features differ: {bundle['features']}")
    import train_phrr_v3_robust as TV3
    refit = TransformedTargetRegressor(regressor=TV3.hgb(), func=np.log1p, inverse_func=np.expm1).fit(Dp["X"], Dp["y"])
    dmax = float(np.max(np.abs(refit.predict(Dp["X"]) - bundle["model"].predict(Dp["X"]))))
    log(f"exp074 refit vs saved phrr_hgb_v4.joblib on the 191 FSRI rows: max |diff| {dmax:.3g} kW/m2 (limit 1e-6)")
    step1["exp074"]["joblib_refit_max_abs_diff_kWm2"] = dmax
    if dmax > 1e-6:
        ok_p = False
    for name, ok, tgt in (("exp074", ok_p, "pHRR"), ("exp046", ok_t, "TTI"), ("exp117", ok_c, "CR")):
        if not ok:
            STATE["targets_blocked"][tgt] = f"{name} step-1 reproduction failed"
            log(f"BLOCKED ({tgt} only): {name} not reproduced")

    # fit the frozen records on all FSRI rows (no MaCFP row anywhere near a fit)
    m_phrr = bundle["model"]
    m_tti = Dt["inc"].fit(Dt["X"], Dt["y"])
    imp = m_tti.named_steps["simpleimputer"]
    ind = list(imp.indicator_.features_)
    log(f"exp046 fitted on all 681 rows; imputer medians {dict(zip(TTI_FEATS, np.round(imp.statistics_, 4)))}; "
        f"indicator features {[TTI_FEATS[i] for i in ind]}")
    if 2 not in ind:
        STATE["targets_blocked"]["TTI"] = "exp046 imputer carries no log_kpc indicator"
        log("BLOCKED (TTI only): exp046 imputer has no log_kpc indicator")
    m_cr = cr_model.fit(Dc["X"], Dc["y"])
    log("exp117 fitted on all 567 rows (Avg of HGB, ExtraTrees, Ridge; predictions not clipped)")

    # =========================================== STEP 2 ===========================================
    log("\n========== STEP 2: PS check and gate (label-free) ==========")
    ps_files = {
        "FZJ": [os.path.join(WOOD, "FZJ", f"FZJ_PS_MCC-raw_N2_60K_R{i}.csv") for i in range(1, 7)],
        "Lund": [os.path.join(WOOD, "Lund", "Lund_PS_MCC-raw_N2_60K_R1.csv")],
        "NIST": [os.path.join(WOOD, "NIST", "NIST_Polystyrene_MCC_N2_60K_R1.txt")],
        "TUBS": [os.path.join(WOOD, "TUBS", "TUBS_PS_MCC-raw_N2_30K_R1.csv")],
        "UDRI": [os.path.join(WOOD, "UDRI", f"UDRI_PS_MCC_N2_60K_R{i}.csv") for i in range(1, 4)],
    }
    ps_runs, ps_lab = [], {}
    for lab, fl in ps_files.items():
        for f in fl:
            try:
                r = mcc_run(f)
            except Exception as e:
                block(f"PS MCC file {os.path.relpath(f, PIN)} unparseable: {e}")
            r["lab"] = lab
            ps_runs.append(r)
        rr = [r for r in ps_runs if r["lab"] == lab]
        ps_lab[lab] = dict(n=len(rr), hrc_JgK=float(np.median([r["hrc_JgK"] for r in rr])),
                           h_c_kJg=float(np.median([r["total_hr_bc_kJg"] for r in rr])))
    ref_hrc = float(np.median([v["hrc_JgK"] for v in ps_lab.values()]))
    ref_hc = float(np.median([v["h_c_kJg"] for v in ps_lab.values()]))
    for lab, v in ps_lab.items():
        v["hrc_ratio"] = v["hrc_JgK"] / ref_hrc
        v["h_c_ratio"] = v["h_c_kJg"] / ref_hc
        v["pass"] = bool(0.90 <= v["hrc_ratio"] <= 1.10 and 0.90 <= v["h_c_ratio"] <= 1.10)
    log("per run: lab | file | beta (src) | peak W/g | HRC J/gK | h_c (bc) kJ/g | rows")
    for r in ps_runs:
        log(f"  {r['lab']:5s} {r['file']:60s} {r['beta_C_s']:.3f} ({r['beta_src']}) {r['peak_hrr_Wg']:8.1f} "
            f"{r['hrc_JgK']:8.1f} {r['total_hr_bc_kJg']:7.2f} {r['n_rows']}")
    log(f"reference (median of 5 lab medians): HRC {ref_hrc:.1f} J/gK, h_c {ref_hc:.2f} kJ/g")
    for lab, v in ps_lab.items():
        log(f"  {lab:5s} n={v['n']} HRC {v['hrc_JgK']:.1f} ({v['hrc_ratio']:.3f}) h_c {v['h_c_kJg']:.2f} "
            f"({v['h_c_ratio']:.3f}) -> {'PASS' if v['pass'] else 'FAIL'}")
    log("context only (never part of the gate): TUBS self-report 1099 J/gK, 38.8 kJ/g (long-run 1108 +- 15 J/gK, "
        "38.7 +- 0.2 kJ/g, 30 K/min); UDRI self-report 1099-1166 W/g, 41.2-41.8 kJ/g (60 K/min)")
    log("no PS-derived factor is applied to any input")
    use_tubs_peak = ps_lab["TUBS"]["pass"]
    if not use_tubs_peak:
        log("TUBS fails the PS gate: TUBS dropped from mcc_peak_hrr_Wg (FZJ 30K runs only)")
    for lab in ("Lund", "NIST", "UDRI"):
        if not ps_lab[lab]["pass"]:
            log(f"{lab} fails the PS gate: reported only (its wood MCC is never an input)")
    ps_out = dict(runs=ps_runs, per_lab=ps_lab, reference=dict(hrc_JgK=ref_hrc, h_c_kJg=ref_hc),
                  pass_rule="0.90 <= lab/reference <= 1.10 for both hrc and h_c", tubs_peak_used=use_tubs_peak,
                  context_only="TUBS self-report 1099 J/gK, 38.8 kJ/g; UDRI self-report 1099-1166 W/g, 41.2-41.8 kJ/g")
    dump_json(ps_out, os.path.join(OUT, "ps_check.json"))
    fzj_fail = not ps_lab["FZJ"]["pass"]
    if fzj_fail:
        STATE["flags"].append("FZJ fails the PS gate: no pHRR, TTI or CR prediction is run")
        log("FLAG: FZJ fails the PS gate -> no pHRR, TTI or CR prediction; PS table and stop-time table still written")

    # =========================================== STEP 3 ===========================================
    log("\n========== STEP 3: derive and freeze inputs (no label read yet) ==========")
    # ---- label-free report of every Wood MCC file in the pin
    readme_purpose = {}
    for lab in ("FZJ",):
        tabs, _ = readme_tables(os.path.join(WOOD, lab, "README.md"))
        for ln, hdr, rows in tabs:
            pc = [j for j, h in enumerate(hdr) if clean_hdr(h) == "Purpose"]
            nc = name_col(hdr)
            if pc and nc is not None:
                for r in rows:
                    readme_purpose[r[nc].strip()] = r[pc[0]].strip()
    wood_mcc = []
    for f in sorted(glob.glob(os.path.join(WOOD, "*", "*Wood*MCC*"))):
        if "TEMPLATE" in f:
            continue
        b = os.path.basename(f)
        lab = b.split("_")[0]
        m = re.search(r"MCC(-raw)?_([A-Za-z0-9-]+)_(\d+)K_R(\d+)", b)
        key = re.sub(r"\.csv$", "", b).replace("MCC-raw", "MCC")
        row = dict(lab=lab, file=b, kind="raw" if "-raw" in b else "processed", atmosphere=m.group(2) if m else "",
                   rate_K_min=int(m.group(3)) if m else None, purpose=readme_purpose.get(key, ""))
        try:
            r = mcc_run(f)
            row.update(peak_Wg=r["peak_hrr_Wg"], hrc_JgK=r["hrc_JgK"], t_peak_C=r["t_peak_C"],
                       total_hr_bc_kJg=r["total_hr_bc_kJg"], t_onset_bc_C=r["t_onset_bc_C"], rows=r["n_rows"])
        except Exception as e:
            row.update(error=str(e))
        wood_mcc.append(row)
    WM = pd.DataFrame(wood_mcc)
    log(f"label-free report: {len(WM)} Wood MCC files (never inputs except FZJ raw 30K R1-R2 and TUBS 30K R1-R3)")
    log(WM.to_string(index=False))

    # ---- MCC inputs (30 K/min N2)
    fzj_in = [mcc_run(os.path.join(WOOD, "FZJ", f"FZJ_Wood_MCC-raw_N2_30K_R{i}.csv")) for i in (1, 2)]
    tubs_in = [mcc_run(os.path.join(WOOD, "TUBS", f"TUBS_Wood_MCC_N2_30K_R{i}.csv")) for i in (1, 2, 3)]
    peak_runs = [r["peak_hrr_Wg"] for r in fzj_in] + ([r["peak_hrr_Wg"] for r in tubs_in] if use_tubs_peak else [])
    onsets = [r["t_onset_bc_C"] for r in fzj_in]
    if any(o is None for o in onsets):
        block("FZJ 30K onset undefined (total_hr_bc < 0.5)")
    # FZJ README MCC masses for char yield
    tabs, _ = readme_tables(os.path.join(WOOD, "FZJ", "README.md"))
    cy = {}
    for ln, hdr, rows in tabs:
        nc = name_col(hdr)
        ic = [j for j, h in enumerate(hdr) if clean_hdr(h) == "Initial Sample Mass (mg)"]
        fc = [j for j, h in enumerate(hdr) if clean_hdr(h) == "Final Sample Mass (mg)"]
        if nc is None or not ic or not fc:
            continue
        for r in rows:
            if r[nc].strip() in ("FZJ_Wood_MCC_N2_30K_R1", "FZJ_Wood_MCC_N2_30K_R2"):
                cy[r[nc].strip()] = (float(r[ic[0]]), float(r[fc[0]]))
    if cy != {"FZJ_Wood_MCC_N2_30K_R1": (4.0, 0.64), "FZJ_Wood_MCC_N2_30K_R2": (4.01, 0.54)}:
        block(f"FZJ README MCC masses not reproduced: {cy}")
    mat = dict(
        mcc_peak_hrr_Wg=float(np.mean(peak_runs)),
        mcc_t_peak_C=float(np.mean([r["t_peak_C"] for r in fzj_in])),
        mcc_total_hr_kJg=float(np.mean([r["total_hr_bc_kJg"] for r in fzj_in])),
        mcc_t_onset_C_mean=float(np.mean(onsets)),
        mcc_t_onset_C_median=float(np.median(onsets)),
        mcc_char_yield_pct=float(np.mean([100 * f / i for i, f in cy.values()])),
    )
    log(f"FZJ raw 30K inputs: {[{k: r[k] for k in ('file', 'peak_hrr_Wg', 't_peak_C', 'total_hr_bc_kJg', 't_onset_bc_C')} for r in fzj_in]}")
    log(f"TUBS 30K inputs (peak only): {[(r['file'], r['peak_hrr_Wg']) for r in tubs_in]}")
    fzj_proc = [mcc_run(os.path.join(WOOD, "FZJ", f"FZJ_Wood_MCC_N2_30K_R{i}.csv")) for i in (1, 2)]
    log(f"FZJ processed 30K (report only, never inputs): "
        f"{[{k: r[k] for k in ('file', 'peak_hrr_Wg', 't_peak_C', 'total_hr_bc_kJg', 't_onset_bc_C')} for r in fzj_proc]}")
    log(f"FZJ README MCC char yield: R1 {100*0.64/4.0:.2f}%, R2 {100*0.54/4.01:.2f}%")

    # ---- TPS (S1)
    tps = []
    for lab, names in (("UMET", ("Wood_TPS_Sample1", "Wood_TPS_Sample2")), ("UQ", ("UQ_Wood_TPS_R1", "UQ_Wood_TPS_R2"))):
        tabs, _ = readme_tables(os.path.join(WOOD, lab, "README.md"))
        for ln, hdr, rows in tabs:
            kc = [j for j, h in enumerate(hdr) if "Thermal Conductivity" in h]
            cc = [j for j, h in enumerate(hdr) if "Volumetric Specific Heat" in h]
            if not kc or not cc:
                continue
            for r in rows:
                if r[0].strip() in names:
                    tps.append((r[0].strip(), float(r[kc[0]]), float(r[cc[0]])))
    declared_tps = [("Wood_TPS_Sample1", 0.147, 0.362), ("Wood_TPS_Sample2", 0.144, 0.410),
                    ("UQ_Wood_TPS_R1", 0.142, 0.406), ("UQ_Wood_TPS_R2", 0.134, 0.430)]
    if tps != declared_tps:
        block(f"TPS readings differ from the config: {tps}")
    log_kpc_S1 = float(np.log(np.mean([k for _, k, _ in tps]) * np.mean([c for _, _, c in tps]) * 1e6))
    mat["log_kpc_S1"] = log_kpc_S1
    log(f"TPS: mean k {np.mean([k for _, k, _ in tps]):.5f} W/mK, mean rho*c {np.mean([c for _, _, c in tps]):.4f} MJ/m3K "
        f"-> log_kpc_S1 {log_kpc_S1:.4f}")

    # ---- STA (FSRI MaCFP files) for CR
    sta = {}
    sta750 = {}
    for rate, reps in ((3, (1,)), (10, (1, 2, 3)), (30, (1, 2, 3))):
        vals, v750 = [], []
        for rep in reps:
            f = os.path.join(WOOD, "FSRI", f"FSRI_Wood_STA_N2_{rate}K_R{rep}.csv")
            with open(f, newline="", encoding="utf-8-sig", errors="replace") as fh:
                rdr = csv.reader(fh)
                header = next(rdr)
                ti = next(i for i, h in enumerate(header) if "Temp" in h)
                mi = next(i for i, h in enumerate(header) if "Mass" in h)
                temps, masses = [], []
                for row in rdr:
                    if len(row) > max(ti, mi):
                        a, b = to_float(row[ti]), to_float(row[mi])
                        if a is not None and b is not None:
                            temps.append(a)
                            masses.append(b)
            pct = [100.0 * m / masses[0] for m in masses]
            if not -5.0 <= pct[-1] <= 105.0:
                log(f"STA {os.path.basename(f)} fails the [-5, 105] QA gate ({pct[-1]:.2f}); not used")
                continue
            vals.append(pct[-1])
            i750 = int(np.argmax(np.asarray(temps) >= 1023.15))
            v750.append(pct[i750])
            log(f"  STA {os.path.basename(f)}: unit {header[mi]}, final T {temps[-1]:.1f} K, residue {pct[-1]:.3f}% "
                f"(at first T >= 1023.15 K: {pct[i750]:.3f}%)")
        sta[f"sta_res_{rate}K"] = float(np.mean(vals))
        sta750[f"sta_res_{rate}K_at_750C"] = float(np.mean(v750))
    mat.update(sta)
    log(f"STA residues (inputs): {sta}; at 750 C (label-free report only): {sta750}")

    flags_primary = dict(cat_interior_finish=0.0, cat_exterior_finish=1.0, cat_insulation=0.0, cat_roofing=0.0,
                         cat_structural=1.0)
    flags_S3 = {k: 0.0 for k in flags_primary}

    # ---- geometry
    W = T[T.stratum.isin(["primary", "UMET"])].copy()
    W["cone_thickness_mm"] = W.L_mm
    W["density_kgm3"] = W.m0_g / 1000 / (W.planform_m2 * W.L_mm / 1000)
    W["cone_mass_per_area_kgm2"] = W.m0_g / 1000 / W.A_norm_m2
    P = W[W.stratum == "primary"]
    G2 = dict(cone_thickness_mm=float(P.cone_thickness_mm.median()), density_kgm3=float(P.density_kgm3.median()),
              cone_mass_per_area_kgm2=float(P.cone_mass_per_area_kgm2.median()))
    mat["G2"] = G2
    log(f"G2 (medians over primary_80): {G2}")

    # ---- sanity gates
    gates = []

    def gate(name, v, lo, hi):
        ok = lo <= v <= hi
        gates.append(dict(name=name, value=v, lo=lo, hi=hi, ok=bool(ok)))
        return ok

    gate("mcc_peak_hrr_Wg", mat["mcc_peak_hrr_Wg"], 40, 120)
    gate("mcc_total_hr_kJg", mat["mcc_total_hr_kJg"], 8, 16)
    gate("mcc_t_onset_C (mean)", mat["mcc_t_onset_C_mean"], 200, 400)
    gate("mcc_t_onset_C (median)", mat["mcc_t_onset_C_median"], 200, 400)
    gate("mcc_t_peak_C", mat["mcc_t_peak_C"], 330, 420)
    gate("mcc_char_yield_pct", mat["mcc_char_yield_pct"], 5, 30)
    for k, v in sta.items():
        gate(k, v, 5, 40)
    for col, lo, hi in (("density_kgm3", 250, 600), ("cone_thickness_mm", 20, 30), ("cone_mass_per_area_kgm2", 6, 16)):
        gate(f"per-test {col} min (n={len(W)})", float(W[col].min()), lo, hi)
        gate(f"per-test {col} max (n={len(W)})", float(W[col].max()), lo, hi)
    for g in gates:
        log(f"  gate {g['name']:42s} {g['value']:10.3f} in [{g['lo']}, {g['hi']}]: {'ok' if g['ok'] else 'FAIL'}")
    if not all(g["ok"] for g in gates):
        block("sanity gate failed: " + ", ".join(g["name"] for g in gates if not g["ok"]))

    # ---- per-test input rows for every variant
    def phrr_row(r, geom, kpc):
        return [r.flux, mat["mcc_peak_hrr_Wg"], mat["mcc_char_yield_pct"], geom["cone_mass_per_area_kgm2"],
                geom["cone_thickness_mm"], geom["density_kgm3"], mat["mcc_total_hr_kJg"], mat["mcc_t_peak_C"],
                mat["mcc_t_onset_C_mean"], kpc]

    def tti_row(r, geom, kpc):
        return [np.log(1e3 * r.flux), np.log(mat["mcc_t_onset_C_median"] - T0_C), kpc,
                np.log(geom["cone_thickness_mm"] / 1000), np.log(geom["density_kgm3"])]

    def cr_row(r, geom, flags):
        return [r.flux, geom["density_kgm3"], mat["mcc_char_yield_pct"], sta["sta_res_3K"], sta["sta_res_10K"],
                sta["sta_res_30K"]] + [flags[c] for c in CR_FEATS[6:]]

    X = {}
    for _, r in W.iterrows():
        g1 = dict(cone_thickness_mm=r.cone_thickness_mm, density_kgm3=r.density_kgm3,
                  cone_mass_per_area_kgm2=r.cone_mass_per_area_kgm2)
        X[r.stem] = dict(
            phrr_primary=phrr_row(r, g1, np.nan), phrr_S1=phrr_row(r, g1, log_kpc_S1), phrr_S2=phrr_row(r, G2, np.nan),
            tti_primary=tti_row(r, g1, np.nan), tti_S1=tti_row(r, g1, log_kpc_S1), tti_S2=tti_row(r, G2, np.nan),
            cr_primary=cr_row(r, g1, flags_primary), cr_S2=cr_row(r, G2, flags_primary), cr_S3=cr_row(r, g1, flags_S3))
    # Edinburgh: FSRI Black_PMMA material features from exp046's training frame, log_L = log(0.006)
    import train_tti_v5_physics as v5
    bp = v5.df[v5.df.material == "Black_PMMA"]
    bp_feats = bp[["log_dT", "log_kpc", "log_rho"]].drop_duplicates()
    if len(bp_feats) != 1:
        block(f"Black_PMMA rows in exp046's frame carry {len(bp_feats)} distinct (log_dT, log_kpc, log_rho)")
    bpv = bp_feats.iloc[0].to_dict()
    log(f"Edinburgh inputs: FSRI Black_PMMA log_dT {bpv['log_dT']:.4f}, log_kpc {bpv['log_kpc']:.4f}, "
        f"log_rho {bpv['log_rho']:.4f}; log_L = log(0.006); log_q = log(1e3 * flux)")
    for _, r in E.iterrows():
        X[r.stem] = dict(tti_primary=[np.log(1e3 * r.flux), bpv["log_dT"], bpv["log_kpc"], np.log(0.006), bpv["log_rho"]])

    def clean(v):
        return None if (isinstance(v, float) and not np.isfinite(v)) else (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)

    rows_out = []
    for _, r in W.iterrows():
        rows_out.append(dict(lab=r.lab, file=r.file, stratum=r.stratum, flux=int(r.flux), orientation=r.orientation,
                             m0_g=clean(r.m0_g), L_mm=clean(r.L_mm), planform_m2=clean(r.planform_m2),
                             density_kgm3=clean(r.density_kgm3), mass_per_area_kgm2=clean(r.cone_mass_per_area_kgm2),
                             A_norm_m2=r.A_norm_m2, sampling_s=clean(r.sampling_s),
                             inputs={k: [clean(x) for x in v] for k, v in X[r.stem].items()}))
    for _, r in E.iterrows():
        rows_out.append(dict(lab="Edinburgh_PMMA", file=f"PMMA/Calibration_Data/Edinburgh/README.md#{r.stem}",
                             stratum="Edinburgh_PMMA", flux=int(r.flux), orientation="n/a",
                             inputs={k: [clean(x) for x in v] for k, v in X[r.stem].items()}))
    features = dict(
        config=CONFIG, feature_order=dict(phrr=PHRR_FEATS, tti=TTI_FEATS, cr=CR_FEATS),
        material_level={k: (v if not isinstance(v, float) else round(v, 6)) for k, v in mat.items()},
        sta_at_750C_report_only=sta750, cr_flags_primary=flags_primary, cr_flags_S3=flags_S3,
        mcc_runs=dict(fzj_raw_30K=fzj_in, tubs_30K=tubs_in, tubs_peak_used=use_tubs_peak),
        tps=declared_tps, edinburgh_black_pmma=bpv, tests=rows_out)
    fpath = os.path.join(OUT, "features.json")
    dump_json(features, fpath)
    feat_sha = sha256(fpath)
    log(f"FROZEN features.json ({len(rows_out)} input rows) sha256 {feat_sha}  [no label column read yet]")

    # ---- label-free reports
    log("\nlabel-free feature table (MaCFP pine vs FSRI training rows of Pine_Siding and Wood_Stud):")
    pdf = pd.DataFrame(Dp["X"], columns=PHRR_FEATS)
    pdf["material"] = Dp["g"]
    cdf = pd.DataFrame(Dc["X"], columns=CR_FEATS)
    cdf["material"] = Dc["g"]
    tdf = v5.df
    ftab = []
    for f, mv in (("mcc_peak_hrr_Wg", mat["mcc_peak_hrr_Wg"]), ("mcc_char_yield_pct", mat["mcc_char_yield_pct"]),
                  ("mcc_total_hr_kJg", mat["mcc_total_hr_kJg"]), ("mcc_t_peak_C", mat["mcc_t_peak_C"]),
                  ("mcc_t_onset_C", mat["mcc_t_onset_C_mean"]), ("log_kpc", float("nan")),
                  ("cone_thickness_mm", G2["cone_thickness_mm"]), ("density_kgm3", G2["density_kgm3"]),
                  ("cone_mass_per_area_kgm2", G2["cone_mass_per_area_kgm2"])):
        row = {"feature (exp074)": f, "MaCFP pine": f"{mv:.3f}" if np.isfinite(mv) else f"NaN (S1 {log_kpc_S1:.3f})"}
        for m in ("Pine_Siding", "Wood_Stud"):
            s = pdf.loc[pdf.material == m, f]
            row[m] = f"{s.iloc[0]:.3f}" if len(s) and np.isfinite(s.iloc[0]) else ("NaN" if len(s) else "not in frame")
        ftab.append(row)
    for f, mv in (("log_dT (exp046)", np.log(mat["mcc_t_onset_C_median"] - T0_C)),
                  ("log_L (exp046)", np.log(G2["cone_thickness_mm"] / 1000)),
                  ("log_rho (exp046)", np.log(G2["density_kgm3"]))):
        col = f.split(" ")[0]
        row = {"feature (exp074)": f, "MaCFP pine": f"{mv:.3f}"}
        for m in ("Pine_Siding", "Wood_Stud"):
            s = tdf.loc[tdf.material == m, col]
            row[m] = f"{s.iloc[0]:.3f}" if len(s) else "not in frame"
        ftab.append(row)
    for f in ("sta_res_3K", "sta_res_10K", "sta_res_30K", "cat_structural", "cat_exterior_finish", "cat_interior_finish"):
        mv = sta.get(f, flags_primary.get(f))
        row = {"feature (exp074)": f + " (exp117)", "MaCFP pine": f"{mv:.3f}"}
        for m in ("Pine_Siding", "Wood_Stud"):
            s = cdf.loc[cdf.material == m, f]
            row[m] = f"{s.iloc[0]:.3f}" if len(s) and np.isfinite(s.iloc[0]) else ("NaN" if len(s) else "not in frame")
        ftab.append(row)
    FT = pd.DataFrame(ftab).rename(columns={"feature (exp074)": "feature"})
    log(FT.to_string(index=False))

    if fzj_fail:
        log("FZJ PS gate failed: skipping steps 4 and the prediction metrics of step 5")

    # =========================================== STEP 4 ===========================================
    log("\n========== STEP 4: predict all variants ==========")
    W = W.reset_index(drop=True)
    preds = pd.DataFrame({"stem": W.stem})
    if not fzj_fail:
        for v in ("primary", "S1", "S2"):
            preds[f"pred_phrr_{v}"] = m_phrr.predict(np.array([X[s][f"phrr_{v}"] for s in W.stem], float))
            preds[f"pred_tti_{v}"] = m_tti.predict(np.array([X[s][f"tti_{v}"] for s in W.stem], float))
        for v in ("primary", "S2", "S3"):
            preds[f"pred_cr_{v}"] = m_cr.predict(np.array([X[s][f"cr_{v}"] for s in W.stem], float))
        E["pred_tti_primary"] = m_tti.predict(np.array([X[s]["tti_primary"] for s in E.stem], float))
        log(preds.describe().T[["count", "mean", "min", "max"]].round(2).to_string())
        log(f"Edinburgh TTI predictions: {E.groupby('flux').pred_tti_primary.first().round(2).to_dict()}")
    # tests.csv, predictions only (labels blank) -- written before any label is read
    base = T[["lab", "file", "stem", "stratum", "flux", "orientation", "m0_g", "L_mm", "planform_m2", "sampling_s",
              "declared_sampling_s"]].copy()
    geo = W.set_index("stem")[["density_kgm3", "cone_mass_per_area_kgm2"]]
    base = base.join(geo, on="stem").merge(preds, on="stem", how="left")
    edb = E[["lab", "stem", "stratum", "flux"]].copy()
    edb["file"] = "PMMA/Calibration_Data/Edinburgh/" + edb.stem + ".csv (README row)"
    edb["orientation"] = "n/a"
    if "pred_tti_primary" in E:
        edb["pred_tti_primary"] = E.pred_tti_primary
    tests_pred = pd.concat([base, edb], ignore_index=True)
    tpath = os.path.join(OUT, "tests.csv")
    tests_pred.to_csv(tpath, index=False)
    log(f"predictions written to tests.csv (labels blank) sha256 {sha256(tpath)}")

    # =========================================== STEP 5 ===========================================
    log("\n========== STEP 5: labels, metrics, tables ==========")
    lab_rows = []
    for _, t in T.iterrows():
        df = data[t.stem]
        tt, mm, hh = (df[c].to_numpy(float) for c in ("Time (s)", "Mass (g)", "HRR (kW/m2)"))
        if t.lab == "FZJ":
            mm = mm + t.m0_g  # tared channel + README m0 (MaCFP read_cone_data)
        hok = np.isfinite(hh) & np.isfinite(tt)
        mok = np.isfinite(mm) & np.isfinite(tt)
        rec = dict(stem=t.stem)
        lr = labels_readme.get(t.stem, {})
        rec["t_ign_reported_s"] = lr.get("t_ign_s", np.nan)
        rec["t_flameout_reported_s"] = lr.get("t_flameout_s", np.nan)
        if hok.any():
            i = int(np.argmax(np.where(hok, hh, -np.inf)))
            rec["obs_phrr_kWm2"] = float(hh[i])
            rec["t_phrr_minus_tign_s"] = float(tt[i] - rec["t_ign_reported_s"]) if np.isfinite(rec["t_ign_reported_s"]) else np.nan
        else:
            rec["obs_phrr_kWm2"] = rec["t_phrr_minus_tign_s"] = np.nan
        ch_m0 = float(np.median(mm[mok & (tt <= 5)])) if (mok & (tt <= 5)).any() else np.nan
        rec["channel_m0_g"] = ch_m0
        m0 = t.m0_g if np.isfinite(t.m0_g) else ch_m0
        if t.in_cr_flameout:
            mf = float(np.interp(rec["t_flameout_reported_s"], tt[mok], mm[mok]))
            rec["cr_flameout_readme_m0_pct"] = 100 * mf / t.m0_g
            rec["cr_flameout_channel_m0_pct"] = 100 * mf / ch_m0
        else:
            rec["cr_flameout_readme_m0_pct"] = rec["cr_flameout_channel_m0_pct"] = np.nan
        if mok.any():
            rec["eot_residue_pct"] = 100 * float(mm[mok][-1]) / m0
        else:
            rec["eot_residue_pct"] = np.nan
        rec["duration_s"] = last_t[t.stem]
        lab_rows.append(rec)
    Lb = pd.DataFrame(lab_rows)
    R = tests_pred.merge(Lb, on="stem", how="left").merge(T[["stem", "in_cr_flameout", "has_finite_mass"]], on="stem",
                                                          how="left")
    ed_idx = R.stratum == "Edinburgh_PMMA"
    R.loc[ed_idx, "t_ign_reported_s"] = R.loc[ed_idx, "stem"].map(E.set_index("stem").t_ign_s)
    R["obs_tti_s"] = R.t_ign_reported_s
    cols = ["lab", "file", "stratum", "flux", "orientation", "m0_g", "L_mm", "planform_m2", "density_kgm3",
            "cone_mass_per_area_kgm2", "sampling_s", "declared_sampling_s", "t_ign_reported_s", "t_flameout_reported_s",
            "obs_phrr_kWm2", "pred_phrr_primary", "pred_phrr_S1", "pred_phrr_S2", "t_phrr_minus_tign_s", "obs_tti_s",
            "pred_tti_primary", "pred_tti_S1", "pred_tti_S2", "cr_flameout_readme_m0_pct", "cr_flameout_channel_m0_pct",
            "pred_cr_primary", "pred_cr_S2", "pred_cr_S3", "eot_residue_pct", "duration_s", "channel_m0_g",
            "in_cr_flameout", "has_finite_mass"]
    for c in cols:
        if c not in R:
            R[c] = np.nan
    R[cols].to_csv(tpath, index=False)
    log(f"tests.csv written with labels: {len(R)} rows, sha256 {sha256(tpath)}")

    S = {"counts": dict(primary_80=80, fsri_lab=18, other_labs=62, tti_25=11, umet=6, tti_86_secondary=86,
                        edinburgh=12, cr_flameout=53, cr_primary_50=50, cr_minus_umet_uq_47=47, end_of_test=89),
         "step1": step1, "ps_gate": {k: dict(pass_=v["pass"], hrc_ratio=round(v["hrc_ratio"], 4),
                                             h_c_ratio=round(v["h_c_ratio"], 4)) for k, v in ps_lab.items()},
         "features_sha256": feat_sha, "material_level": features["material_level"]}
    rep = {}  # markdown sections

    # ---- stop-time table (always written)
    eot = R[R.has_finite_mass == True]  # noqa: E712
    st_rows = []
    for lab in ["Aalto", "BUWfd", "FPL", "FSRI", "FZJ", "IMT", "TUBS", "UDRI", "UMET", "UQ"]:
        g = eot[eot.lab == lab]
        st_rows.append({"lab": lab_label(lab), "n": len(g), "residue mean %": f1(g.eot_residue_pct.mean()),
                        "residue min %": f1(g.eot_residue_pct.min()), "residue max %": f1(g.eot_residue_pct.max()),
                        "duration mean s": f1(g.duration_s.mean()), "duration min s": f1(g.duration_s.min()),
                        "duration max s": f1(g.duration_s.max()), "stop rule (README)": STOP_RULE[lab]})
    rep_stop = pd.DataFrame(st_rows)
    cr_ctx = pd.read_csv(os.path.join(HERE, "cone_replicates_fsri.csv"))
    ctx_rows = []
    for m in ("Pine_Siding", "Wood_Stud"):
        g = cr_ctx[cr_ctx.material == m]
        ctx_rows.append({"FSRI DB material (training data, no model)": m, "n": len(g),
                         "end_of_test_time_s range": f"{g.end_of_test_time_s.min():.1f} - {g.end_of_test_time_s.max():.1f}",
                         "residue_pct range": f"{g.residue_pct.min():.1f} - {g.residue_pct.max():.1f}"})
    rep_stop_ctx = pd.DataFrame(ctx_rows)
    log("\nstop-time table (end-of-test residue, all cone files with a finite-mass record):")
    log(rep_stop.to_string(index=False))
    log(rep_stop_ctx.to_string(index=False))
    S["stop_time"] = {r["lab"]: {k: r[k] for k in r if k != "lab"} for r in st_rows}

    if fzj_fail:
        write_summary(S)
        write_report(rep, rep_stop, rep_stop_ctx, None, S, WM, FT, ps_lab, ref_hrc, ref_hc, ps_runs, gates, mat, sta750,
                     G2, step1)
        return

    PR = R[R.stratum == "primary"].copy()
    UM = R[R.stratum == "UMET"].copy()
    ED = R[ed_idx].copy()
    fsri, other = PR[PR.lab == "FSRI"], PR[PR.lab != "FSRI"]
    assert len(PR) == 80 and len(fsri) == 18 and len(other) == 62 and len(UM) == 6 and len(ED) == 12

    def headline(obs, pred, with25=False):
        out = {}
        for name, d, ci in (("primary_80", PR, True), ("FSRI-lab_18", fsri, False), ("other-labs_62", other, True)):
            y, p = d[obs].to_numpy(float), d[pred].to_numpy(float)
            h = dict(n=len(d), r2=round(r2(y, p), 4), mae=round(float(np.mean(np.abs(p - y))), 1),
                     bias=round(float(np.mean(p - y)), 1))
            if ci:
                lo, hi = lab_ci(y, p, d.lab.to_numpy())
                h.update(ci_lo=round(lo, 4), ci_hi=round(hi, 4), n_labs=int(d.lab.nunique()))
            out[name] = h
        if with25:
            d = PR[PR.flux == 25]
            y, p = d[obs].to_numpy(float), d[pred].to_numpy(float)
            out["tti_25"] = dict(n=len(d), r2=round(r2(y, p), 4), mae=round(float(np.mean(np.abs(p - y))), 1),
                                 bias=round(float(np.mean(p - y)), 1))
        return out

    def umet_line(obs, pred):
        y, p = UM[obs].to_numpy(float), UM[pred].to_numpy(float)
        return dict(n=len(UM), mae=round(float(np.mean(np.abs(p - y))), 1), bias=round(float(np.mean(p - y)), 1))

    res = {}
    # ---- pHRR
    if "pHRR" not in STATE["targets_blocked"]:
        res["pHRR"] = {v: headline("obs_phrr_kWm2", f"pred_phrr_{v}") for v in ("primary", "S1", "S2")}
        res["pHRR_umet"] = {v: umet_line("obs_phrr_kWm2", f"pred_phrr_{v}") for v in ("primary", "S1", "S2")}
        rep["phrr_lab"] = group_table(PR, "lab", "obs_phrr_kWm2", "pred_phrr_primary", order=PRIMARY_LABS)
        rep["phrr_flux"] = group_table(PR, "flux", "obs_phrr_kWm2", "pred_phrr_primary")
        rep["phrr_orient"] = group_table(PR, "orientation", "obs_phrr_kWm2", "pred_phrr_primary",
                                         order=["parallel", "perpendicular", "unknown"])
        rep["phrr_contrast"] = contrast_table(PR, "obs_phrr_kWm2", "pred_phrr_primary")
        rep["phrr_umet"] = UM[["file", "flux", "obs_phrr_kWm2", "pred_phrr_primary"]].assign(
            **{"pred-obs": lambda d: (d.pred_phrr_primary - d.obs_phrr_kWm2)})
        q = PR.groupby("lab").t_phrr_minus_tign_s.agg(["min", "median", "max"])
        rep["phrr_qa"] = q.reset_index().rename(columns={"min": "t(pHRR)-t_ign min s", "median": "median s",
                                                         "max": "max s"})
        cm = PR.groupby(["lab", "flux", "orientation"])[["obs_phrr_kWm2", "pred_phrr_primary"]].mean()
        res["pHRR_extras"] = dict(cell_mean_r2=round(r2(cm.obs_phrr_kWm2, cm.pred_phrr_primary), 4), n_cells=len(cm))
    # ---- TTI
    if "TTI" not in STATE["targets_blocked"]:
        res["TTI"] = {v: headline("obs_tti_s", f"pred_tti_{v}", with25=True) for v in ("primary", "S1", "S2")}
        res["TTI_umet"] = {v: umet_line("obs_tti_s", f"pred_tti_{v}") for v in ("primary", "S1", "S2")}
        res["TTI"]["primary"]["primary_80"]["log_r2"] = round(log_r2(PR.obs_tti_s, PR.pred_tti_primary), 4)
        res["TTI"]["primary"]["FSRI-lab_18"]["log_r2"] = round(log_r2(fsri.obs_tti_s, fsri.pred_tti_primary), 4)
        res["TTI"]["primary"]["other-labs_62"]["log_r2"] = round(log_r2(other.obs_tti_s, other.pred_tti_primary), 4)
        d25 = PR[PR.flux == 25]
        res["TTI"]["primary"]["tti_25"]["log_r2"] = round(log_r2(d25.obs_tti_s, d25.pred_tti_primary), 4)
        rep["tti_lab"] = group_table(PR, "lab", "obs_tti_s", "pred_tti_primary", tti=True, order=PRIMARY_LABS)
        rep["tti_flux"] = group_table(PR, "flux", "obs_tti_s", "pred_tti_primary", tti=True)
        rep["tti_orient"] = group_table(PR, "orientation", "obs_tti_s", "pred_tti_primary", tti=True,
                                        order=["parallel", "perpendicular", "unknown"])
        rep["tti_contrast"] = contrast_table(PR, "obs_tti_s", "pred_tti_primary")
        rep["tti_umet"] = UM[["file", "flux", "obs_tti_s", "pred_tti_primary"]].assign(
            **{"pred-obs": lambda d: (d.pred_tti_primary - d.obs_tti_s)})
        a86 = pd.concat([PR, UM])
        res["TTI_86_secondary"] = dict(n=len(a86), r2=round(r2(a86.obs_tti_s, a86.pred_tti_primary), 4),
                                       mae=round(float(np.mean(np.abs(a86.pred_tti_primary - a86.obs_tti_s))), 1))
        ed_out = dict(n=len(ED), r2=round(r2(ED.obs_tti_s, ED.pred_tti_primary), 4),
                      log_r2=round(log_r2(ED.obs_tti_s, ED.pred_tti_primary), 4),
                      mae=round(float(np.mean(np.abs(ED.pred_tti_primary - ED.obs_tti_s))), 1),
                      bias=round(float(np.mean(ED.pred_tti_primary - ED.obs_tti_s)), 1), per_flux={})
        for qf in (25, 65):
            g = ED[ED.flux == qf]
            ed_out["per_flux"][qf] = dict(n=len(g), mean_obs=round(float(g.obs_tti_s.mean()), 1),
                                          pred=round(float(g.pred_tti_primary.mean()), 1),
                                          mae=round(float(np.mean(np.abs(g.pred_tti_primary - g.obs_tti_s))), 1),
                                          bias=round(float(np.mean(g.pred_tti_primary - g.obs_tti_s)), 1))
        bp25 = v5.df[(v5.df.material == "Black_PMMA") & (v5.df.heat_flux_kWm2 == 25)][v5.TARGET]
        ed_out["context_fsri_black_pmma_25"] = dict(n=int(len(bp25)), mean_tti_s=round(float(bp25.mean()), 1),
                                                    edinburgh_25_mean_s=round(float(ED[ED.flux == 25].obs_tti_s.mean()), 1))
        res["TTI_edinburgh"] = ed_out
        cm = PR.groupby(["lab", "flux", "orientation"])[["obs_tti_s", "pred_tti_primary"]].mean()
        res["TTI_extras"] = dict(cell_mean_r2=round(r2(cm.obs_tti_s, cm.pred_tti_primary), 4), n_cells=len(cm))
    # ---- flux response (pHRR, TTI)
    fr = []
    for tgt, obs, pred in (("pHRR", "obs_phrr_kWm2", "pred_phrr_primary"), ("TTI", "obs_tti_s", "pred_tti_primary")):
        if tgt in STATE["targets_blocked"]:
            continue
        for name, d in [("primary_80", PR)] + [(lab, PR[PR.lab == lab]) for lab in ("Aalto", "FSRI", "FZJ", "IMT")]:
            if d.flux.nunique() < 3:
                continue
            lq = np.log(d.flux.to_numpy(float))
            fr.append({"target": tgt, "pool": name, "n": len(d), "fluxes": "/".join(str(int(v)) for v in sorted(d.flux.unique())),
                       "slope log(obs)": f"{np.polyfit(lq, np.log(d[obs].to_numpy(float)), 1)[0]:.3f}",
                       "slope log(pred)": f"{np.polyfit(lq, np.log(d[pred].to_numpy(float)), 1)[0]:.3f}"})
    rep["flux_response"] = pd.DataFrame(fr)
    # ---- exp074 step response by flux
    if "pHRR" not in STATE["targets_blocked"]:
        sr = []
        for qf in sorted(PR.flux.unique()):
            g = PR[PR.flux == qf]
            sr.append({"flux": qf, "n": len(g), "pred mean": f1(g.pred_phrr_primary.mean()),
                       "pred min": f1(g.pred_phrr_primary.min()), "pred max": f1(g.pred_phrr_primary.max())})
        rep["phrr_by_flux_pred"] = pd.DataFrame(sr)
        sweep_fl = [15, 25, 30, 35, 45, 50, 60, 75, 100]
        sw = m_phrr.predict(np.array([[qf, mat["mcc_peak_hrr_Wg"], mat["mcc_char_yield_pct"],
                                       G2["cone_mass_per_area_kgm2"], G2["cone_thickness_mm"], G2["density_kgm3"],
                                       mat["mcc_total_hr_kJg"], mat["mcc_t_peak_C"], mat["mcc_t_onset_C_mean"], np.nan]
                                      for qf in sweep_fl], float))
        rep["phrr_sweep"] = pd.DataFrame({"flux": sweep_fl, "exp074 pred at G2 geometry (kW/m2)": [f1(v) for v in sw]})
    # ---- CR diagnostic
    if "CR" not in STATE["targets_blocked"]:
        CR = R[R.in_cr_flameout == True].copy()  # noqa: E712
        assert len(CR) == 53
        pools = {"50 (declared primary: cr_flameout minus UMET)": CR[CR.lab != "UMET"], "53 (all)": CR,
                 "47 (minus UMET and UQ)": CR[~CR.lab.isin(["UMET", "UQ"])]}
        res["CR"] = {}
        for v in ("primary", "S2", "S3"):
            res["CR"][v] = {}
            for name, d in pools.items():
                e = d[f"pred_cr_{v}"] - d.cr_flameout_readme_m0_pct
                res["CR"][v][name] = dict(n=len(d), mae=round(float(e.abs().mean()), 1), bias=round(float(e.mean()), 1))
        crl = []
        for lab in ["Aalto", "FPL", "FSRI", "IMT", "UDRI", "UMET", "UQ"]:
            g = CR[CR.lab == lab]
            o, p = g.cr_flameout_readme_m0_pct, g.pred_cr_primary
            crl.append({"lab": lab_label(lab) + (" (separate stratum)" if lab == "UMET" else ""), "n": len(g),
                        "mean obs (flameout, README m0) %": f1(o.mean()),
                        "mean obs (channel m0) % [report]": f1(g.cr_flameout_channel_m0_pct.mean()),
                        "mean pred %": f1(p.mean()), "MAE": f1((p - o).abs().mean()), "bias": f1((p - o).mean()),
                        "mean end-of-test residue % [extra]": f1(g.eot_residue_pct.mean()),
                        "MAE vs end-of-test [extra]": f1((p - g.eot_residue_pct).abs().mean())})
        rep["cr_lab"] = pd.DataFrame(crl)
        fl = []
        for qf in (25, 50, 75):
            g = CR[(CR.lab == "FSRI") & (CR.flux == qf)]
            fl.append({"FSRI-lab flux": qf, "n": len(g), "mean obs %": f1(g.cr_flameout_readme_m0_pct.mean()),
                       "mean pred %": f1(g.pred_cr_primary.mean()),
                       "bias": f1((g.pred_cr_primary - g.cr_flameout_readme_m0_pct).mean())})
        rep["cr_fsri_flux"] = pd.DataFrame(fl)
        rep["cr_tests"] = CR[["lab", "file", "flux", "orientation", "t_flameout_reported_s", "cr_flameout_readme_m0_pct",
                              "cr_flameout_channel_m0_pct", "pred_cr_primary", "eot_residue_pct"]]

    S["results"] = res
    s = write_summary(S)
    log("\nHEADLINE (report-only; external cross-lab check, MaCFP-4 pine, one material):")
    log(json.dumps(res, indent=1, default=str))
    write_report(rep, rep_stop, rep_stop_ctx, res, S, WM, FT, ps_lab, ref_hrc, ref_hc, ps_runs, gates, mat, sta750, G2,
                 step1)
    log(f"\nstatus: {s['status']}; runtime {time.time() - T_START:.0f}s")


# ===============================================================================================================
def write_report(rep, rep_stop, rep_stop_ctx, res, S, WM, FT, ps_lab, ref_hrc, ref_hc, ps_runs, gates, mat, sta750,
                 G2, step1):
    keep = ""
    if os.path.exists(REPORT):
        txt = open(REPORT, encoding="utf-8").read()
        m = re.search(r"<!-- summary:start -->.*?<!-- summary:end -->", txt, re.S)
        keep = m.group(0) if m else ""
    if not keep:
        keep = "<!-- summary:start -->\n(summary to be written after the run)\n<!-- summary:end -->"
    L = ["# A2: MaCFP-4 pine and Edinburgh PMMA, external cross-lab check (report-only)", "", keep, "",
         "Quote as: external cross-lab check, MaCFP-4 pine (one material, 8 oxygen-consumption labs), report-only. "
         "Nothing here is a candidate, nothing is compared with any bar, and no number of record changes. "
         "Config: `experiments/configs/macfp_wood_crosslab.json` (committed dd9c9c6 before this run). "
         "Script: `data/v2/validate_macfp_wood_crosslab.py`. Log: `experiments/runs/macfp_wood_crosslab.log`. "
         "Data files: `experiments/runs/macfp_wood_crosslab/` (ps_check.json, features.json, tests.csv, summary.json); "
         "the config names experiments/results/macfp_wood_crosslab/ and experiments/runs/validate_macfp_wood_crosslab.log, "
         "but this run's orchestrator forbade writing into experiments/results/, so the same files went to "
         "experiments/runs/ (contents as declared; not ingested by the tracker either way). "
         "MaCFP matl-db pinned at 0e4ad8b (MIT; DOI 10.18434/mds2-2586; per-lab READMEs; TUBS doi "
         "10.24355/dbbs.084-202604271326-0).", "",
         "Status: **" + ("BLOCKED" if S.get("status") == "blocked" else ("FLAG" if STATE["flags"] else "ok")) + "**"
         + (f"; blocked targets: {STATE['targets_blocked']}" if STATE["targets_blocked"] else "")
         + (f"; flags: {STATE['flags']}" if STATE["flags"] else ""), ""]
    L += ["## How to read this", "",
          "- One material. R2 mostly measures the flux response plus lab bias; the per-lab and per-flux tables are the "
          "real result. Quote R2 only with them.",
          "- Lab TTI definitions and time zeros are not harmonised (Aalto: from start of DAQ; FSRI: operator button). "
          "Sampling is 5 s at Aalto and IMT and 1 s elsewhere, which can lower the raw pHRR max there. Specimens were "
          "conditioned at about 23 C / 50% RH (UQ measured 10.2% moisture); some FZJ 60 kW/m2 specimens were desiccated.",
          "- UQ is kept everywhere and flagged (README: 60 kW/m2 mass readings fluctuated).",
          "- The models are the records exactly as fitted on FSRI: exp074 (`models/v2/phrr_hgb_v4.joblib`), exp046 "
          "(Poisson GLM on 5 physics features, refit on all 681 FSRI rows) and exp117 (Avg of HGB, ExtraTrees and Ridge, "
          "refit on all 567 FSRI rows). No MaCFP row was used to fit anything.",
          "- The PMMA cross-lab 0.588 (exp054) used exp043's model, not exp074; it is a different material and model "
          "and is never pooled with anything here.", ""]
    L += ["## Exclusions applied (all a priori, from the config)", "",
          "- BUWfd_Wood_Cone_50kW_hor_R1: non-physical mass (MaCFP's own script excludes it). The other 5 BUWfd tests "
          "have no cone metadata and appear only in the end-of-test residue table.",
          "- UCantabria: 4 README-only rows with no CSV and no specimen mass; excluded everywhere.",
          "- UMET (ISO 13927 mass-loss cone, thermopile HRR): a separate stratum, never pooled with the oxygen-consumption labs.",
          "- Aalto `*_Gasification_*` and TUBS `*_Gasification_*` files are N2 tests and are never used.",
          "- MaCFP's outcome-based averaging exclusion (IMT 25 kW/m2 R2/R5, chosen by ignition time) is NOT applied.",
          "- Char residue at flameout: a test enters only if it has a README t_flameout inside its finite mass record "
          "(53 tests). FZJ and TUBS report no flameout; FPL R2 lost its DAQ at 1400 s; 4 Aalto and 3 UMET tests fall outside.",
          "- FZJ 60 kW/m2 perpendicular R3 and 75 kW/m2 parallel R1 have all-NaN mass: they are in the pHRR/TTI pools but "
          "not in the residue tables.",
          "- Edinburgh PMMA (12 README rows) is a TTI-only stratum. It probably overlaps FSRI Black_PMMA, which is in "
          "exp046's training rows: known material (probable), new lab.", ""]
    L += ["## Step 1: records reproduced on FSRI before any MaCFP prediction", "",
          md(pd.DataFrame([{"record": k, "rows/groups": f"{v['n_rows']}/{v['n_groups']}", "GKF 5x10 mean": f4(v["mean"]),
                            "CI": f"[{f4(v['ci_lo'])}, {f4(v['ci_hi'])}]",
                            "record value": f"{v['record'][0]} [{v['record'][1]}, {v['record'][2]}]",
                            "reproduced (within 0.002)": v["reproduced"]} for k, v in step1.items()])), "",
          f"exp074 refit vs saved joblib on the 191 FSRI rows: max |diff| "
          f"{step1['exp074'].get('joblib_refit_max_abs_diff_kWm2', float('nan')):.3g} kW/m2 (limit 1e-6).", ""]
    L += ["## Step 2: PS MCC gate (label-free)", "",
          f"Reference = median of the 5 lab medians: HRC {ref_hrc:.1f} J/gK, h_c {ref_hc:.2f} kJ/g. Pass: 0.90 <= "
          "lab/reference <= 1.10 for both. No PS-derived factor is applied to any input. HRC and h_c are treated as "
          "rate-independent instrument checks (FZJ's 60 K/min PS runs vouch for its 30 K/min wood runs).", "",
          md(pd.DataFrame([{"lab": k, "runs": v["n"], "HRC J/gK": f1(v["hrc_JgK"]), "HRC ratio": f"{v['hrc_ratio']:.3f}",
                            "h_c kJ/g": f2(v["h_c_kJg"]), "h_c ratio": f"{v['h_c_ratio']:.3f}",
                            "gate": "PASS" if v["pass"] else "FAIL"} for k, v in ps_lab.items()])), "",
          "Context only, never part of the gate: TUBS self-report 1099 J/gK and 38.8 kJ/g (long-run 1108 +- 15 J/gK, "
          "38.7 +- 0.2 kJ/g, 30 K/min); UDRI self-report 1099-1166 W/g and 41.2-41.8 kJ/g (60 K/min).", ""]
    L += ["## Step 3: frozen inputs (label-free)", "",
          f"Material-level: MCC peak {mat['mcc_peak_hrr_Wg']:.2f} W/g (FZJ raw 30K R1-R2 + TUBS 30K R1-R3), T_peak "
          f"{mat['mcc_t_peak_C']:.2f} C, total HR (bc) {mat['mcc_total_hr_kJg']:.3f} kJ/g, onset (bc) "
          f"{mat['mcc_t_onset_C_mean']:.2f} C (mean = median with 2 runs: {mat['mcc_t_onset_C_median']:.2f}), MCC char "
          f"yield {mat['mcc_char_yield_pct']:.2f}% (FZJ README masses), STA residues 3/10/30 K/min "
          f"{mat['sta_res_3K']:.2f} / {mat['sta_res_10K']:.2f} / {mat['sta_res_30K']:.2f}% (FSRI MaCFP STA; at 750 C, "
          f"report only: {' / '.join(f'{v:.2f}' for v in sta750.values())}%), log_kpc primary NaN, S1 "
          f"{mat['log_kpc_S1']:.4f} (TPS). G2 medians: L {G2['cone_thickness_mm']:.2f} mm, density "
          f"{G2['density_kgm3']:.1f} kg/m3, mass/area {G2['cone_mass_per_area_kgm2']:.3f} kg/m2. CR flags primary "
          f"{{structural, exterior finish}}; S3 all 0. features.json sha256 `{S.get('features_sha256')}`.", "",
          "Sanity gates (all must pass before any prediction):", "",
          md(pd.DataFrame([{"gate": g["name"], "value": f"{g['value']:.3f}", "range": f"[{g['lo']}, {g['hi']}]",
                            "ok": g["ok"]} for g in gates])), "",
          "MaCFP pine inputs next to the FSRI training rows of the two solid-softwood products (label-free; MaCFP "
          "geometry shown as G2 medians):", "", md(FT), "",
          "Every Wood MCC file in the pin (label-free; only FZJ raw 30K R1-R2 and TUBS 30K R1-R3 peak are inputs):", "",
          md(WM.fillna("")), ""]
    if res is None:
        L += ["## Predictions not run", "", "FZJ failed the PS gate (FLAG). No pHRR, TTI or CR prediction was made.", ""]
    if res is not None and "pHRR" in res:
        h = res["pHRR"]["primary"]
        L += ["## pHRR (exp074), primary", "",
              md(pd.DataFrame([{"pool": k, "n": v["n"], "R2 (kW/m2)": f4(v["r2"]),
                                "lab-bootstrap 95% CI": f"[{f4(v['ci_lo'])}, {f4(v['ci_hi'])}]" if "ci_lo" in v else "n/a (single lab)",
                                "MAE": f1(v["mae"]), "bias (pred-obs)": f1(v["bias"])} for k, v in h.items()])), "",
              "Per lab:", "", md(rep["phrr_lab"]), "", "Per nominal flux:", "", md(rep["phrr_flux"]), "",
              "Per orientation:", "", md(rep["phrr_orient"]), "",
              "Same-flux perpendicular minus parallel (kW/m2):", "", md(rep["phrr_contrast"]), "",
              "QA: time of the pHRR max minus reported t_ign (s), per lab:", "",
              md(rep["phrr_qa"].round(1)), "",
              "UMET stratum (mass-loss cone, never pooled): n {n}, MAE {mae:.1f}, bias {bias:+.1f} kW/m2.".format(
                  **res["pHRR_umet"]["primary"]), "",
              md(rep["phrr_umet"].round(1)), "",
              "Secondary variants (each changes one thing; never primary):", "",
              md(pd.DataFrame([{"variant": v, "pool": k, "n": x["n"], "R2": f4(x["r2"]),
                                "CI": f"[{f4(x['ci_lo'])}, {f4(x['ci_hi'])}]" if "ci_lo" in x else "n/a",
                                "MAE": f1(x["mae"]), "bias": f1(x["bias"])}
                               for v in ("S1", "S2") for k, x in res["pHRR"][v].items()]
                              + [{"variant": v, "pool": "UMET 6", "n": 6, "R2": "", "CI": "",
                                  "MAE": f1(res["pHRR_umet"][v]["mae"]), "bias": f1(res["pHRR_umet"][v]["bias"])}
                                 for v in ("S1", "S2")])), "",
              "S1: log_kpc = TPS value; S2: G2 (median) geometry for every row.", ""]
    if res is not None and "TTI" in res:
        h = res["TTI"]["primary"]
        L += ["## TTI (exp046), primary", "",
              md(pd.DataFrame([{"pool": k, "n": v["n"], "R2 (s)": f4(v["r2"]),
                                "lab-bootstrap 95% CI": f"[{f4(v['ci_lo'])}, {f4(v['ci_hi'])}]" if "ci_lo" in v else "n/a",
                                "log R2": f4(v.get("log_r2")), "MAE s": f1(v["mae"]), "bias s": f1(v["bias"])}
                               for k, v in h.items()])), "",
              "Per lab:", "", md(rep["tti_lab"]), "", "Per nominal flux:", "", md(rep["tti_flux"]), "",
              "Per orientation:", "", md(rep["tti_orient"]), "",
              "log R2 is natural log of seconds (as experiments/configs/tti_secondary_metrics.json). Within a lab "
              "tested at one flux (FPL, TUBS, UQ) or within one flux, log R2 is computed on a narrow spread of "
              "observed times and is not informative (e.g. TUBS).", "",
              "Same-flux perpendicular minus parallel (s):", "", md(rep["tti_contrast"]), "",
              "UMET stratum: n {n}, MAE {mae:.1f} s, bias {bias:+.1f} s.".format(**res["TTI_umet"]["primary"]), "",
              md(rep["tti_umet"].round(1)), "",
              "tti_86_secondary (primary_80 + UMET 6; aggregate only, to match the plan's '86 rows' wording): "
              f"R2 {f4(res['TTI_86_secondary']['r2'])}, MAE {f1(res['TTI_86_secondary']['mae'])} s.", "",
              "Secondary variants:", "",
              md(pd.DataFrame([{"variant": v, "pool": k, "n": x["n"], "R2 (s)": f4(x["r2"]),
                                "CI": f"[{f4(x['ci_lo'])}, {f4(x['ci_hi'])}]" if "ci_lo" in x else "n/a",
                                "MAE s": f1(x["mae"]), "bias s": f1(x["bias"])}
                               for v in ("S1", "S2") for k, x in res["TTI"][v].items()]
                              + [{"variant": v, "pool": "UMET 6", "n": 6, "R2 (s)": "", "CI": "",
                                  "MAE s": f1(res["TTI_umet"][v]["mae"]), "bias s": f1(res["TTI_umet"][v]["bias"])}
                                 for v in ("S1", "S2")])), ""]
        e = res["TTI_edinburgh"]
        L += ["## Edinburgh black cast PMMA TTI (separate stratum; known material (probable), new lab)", "",
              f"n {e['n']}: R2 (s) {f4(e['r2'])}, log R2 {f4(e['log_r2'])}, MAE {f1(e['mae'])} s, bias {e['bias']:+.1f} s. "
              "Inputs: FSRI Black_PMMA material features from exp046's training frame, 6 mm thickness. Only two "
              "flux levels (25 and 65 kW/m2), so this R2 mostly reflects the 25-vs-65 difference; the per-flux "
              "rows are the result.", "",
              md(pd.DataFrame([{"flux": k, "n": v["n"], "mean obs s": f1(v["mean_obs"]), "pred s": f1(v["pred"]),
                                "MAE s": f1(v["mae"]), "bias s": f1(v["bias"])} for k, v in e["per_flux"].items()])), "",
              f"Context row, no model: FSRI Black_PMMA replicate-mean TTI at 25 kW/m2 (exp046 training rows, n "
              f"{e['context_fsri_black_pmma_25']['n']}) {f1(e['context_fsri_black_pmma_25']['mean_tti_s'])} s vs "
              f"Edinburgh 25 kW/m2 mean {f1(e['context_fsri_black_pmma_25']['edinburgh_25_mean_s'])} s.", ""]
    if res is not None and "CR" in res:
        L += ["## Char Residue (exp117), diagnostic only (absolute error; no R2)", "",
              "Observed = 100 * m(t_flameout) / m0 (README m0), the mass channel interpolated at the README flameout. "
              "This is NOT the FSRI end-of-test CR definition, which is unchanged; residue at flameout is a diagnostic "
              "comparison, never a CR target.", "",
              md(pd.DataFrame([{"variant": v, "pool": k, "n": x["n"], "MAE %": f1(x["mae"]), "bias %": f1(x["bias"])}
                               for v in ("primary", "S2", "S3") for k, x in res["CR"][v].items()])), "",
              "S2: G2 density; S3: all five category flags 0.", "",
              "Per lab (primary):", "", md(rep["cr_lab"]), "", "FSRI-lab per flux:", "", md(rep["cr_fsri_flux"]), "",
              "Per test:", "", md(rep["cr_tests"].round(1)), ""]
    L += ["## Stop-time table (end-of-test residue; a warning about the FSRI end-of-test CR definition for charring "
          "materials, never grounds to change it)", "", md(rep_stop), "", md(rep_stop_ctx), ""]
    if res is not None:
        L += ["## Reported-only extras", ""]
        if "pHRR_extras" in res:
            L += [f"- pHRR R2 on lab x flux x orientation cell means: {f4(res['pHRR_extras']['cell_mean_r2'])} "
                  f"({res['pHRR_extras']['n_cells']} cells)."]
        if "TTI_extras" in res:
            L += [f"- TTI R2 (s) on lab x flux x orientation cell means: {f4(res['TTI_extras']['cell_mean_r2'])} "
                  f"({res['TTI_extras']['n_cells']} cells)."]
        L += ["", "Flux response, OLS slope on log(q):", "", md(rep["flux_response"]), ""]
        if "phrr_by_flux_pred" in rep:
            L += ["exp074 predictions by flux on primary_80 (FSRI trains on 15/25/35/50/75/100 kW/m2):", "",
                  md(rep["phrr_by_flux_pred"]), "", "exp074 at G2 geometry and the MaCFP material inputs, swept over flux:",
                  "", md(rep["phrr_sweep"]), ""]
    L += ["## Follow-up", "", "- Step 6 of the config: a fresh-context critic pass before any number here is quoted in "
          "STATUS or a model card. Any leaderboard row is the orchestrator's decision after the critic.", ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    log(f"report written: {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
