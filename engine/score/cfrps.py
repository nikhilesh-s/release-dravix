"""CFRPS: composite fire risk proxy score over the Dravix engine's predicted properties.

The patent (US provisional 64/054,493) calls this the Composite Fire-Risk Screening Score
("CFRS Score"): normalised ML fire-risk signals, weighted aggregation of fire-behaviour indices,
lower = lower relative fire susceptibility, comparative not absolute, optionally with confidence
and descriptor-coverage indicators. It gives no formula, weights, ranges or output scale.

Score is 0-100, higher = more fire risk. Each present input is min-max normalised over a fixed
reference range, oriented so 1 = worst, clipped to [0, 1], weighted, and summed. Weights are
renormalised over the inputs that are present; `coverage` reports how much weight was present.
`indices` groups the subscores into the patent's FIG. 18 index names (no smoke index: the engine
predicts no smoke property).

Provenance:
- Weights and ranges are Dravix choices, not the patent's. Weights are versioned JSON configs in
  weights/ (cfrs_weights_<version>.json, with provenance). Default v1: calibrated to within-paper UL-94
  rankings (weights/DECLARATIONS.md). v0 = the old placeholders (FIGRA slot given to avg HRR),
  kept for reproducibility: cfrps(props, weights=load_weights("v0")).
- Ranges: rounded 5th-95th percentiles of the engine's training snapshot
  data/v2/cone_lit_benchmark_v1.csv (all heat fluxes pooled; values > 0).
- `dfrs_v1` reproduces the old engine's DFRS exactly (nikhilesh-s/dravix-engine
  app/services/scoring_engine.py:43-108 and api/main.py:174-179), for comparison only.

Compare scores at the same heat flux: TTI and pHRR depend on it and the ranges pool all fluxes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# name: (unit, low, high, direction). direction +1: higher value raises risk; -1: lowers it.
PROPERTIES = {
    "TTI": ("s", 5.0, 130.0, -1),
    "pHRR": ("kW/m2", 75.0, 1300.0, +1),
    "THR": ("MJ/m2", 8.0, 160.0, +1),
    "avg_HRR": ("kW/m2", 20.0, 400.0, +1),
    "char_residue": ("wt%", 0.0, 55.0, -1),
    "LOI": ("vol% O2", 18.0, 38.0, -1),
}
UNITS = {k: v[0] for k, v in PROPERTIES.items()}
RANGES = {k: (v[1], v[2]) for k, v in PROPERTIES.items()}
DIRECTION = {k: v[3] for k, v in PROPERTIES.items()}

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS_VERSION = "v1"


def load_weights(version: str = DEFAULT_WEIGHTS_VERSION) -> dict:
    """Weights dict from weights/cfrs_weights_<version>.json; checked >= 0, known keys, sum 1."""
    w = json.loads((WEIGHTS_DIR / f"cfrs_weights_{version}.json").read_text())["weights"]
    w = {k: float(v) for k, v in w.items()}
    if set(w) != set(PROPERTIES) or any(v < 0 for v in w.values()) or not math.isclose(sum(w.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"bad weights config {version}: {w}")
    return w


WEIGHTS = load_weights()
WEIGHTS_VERSION = DEFAULT_WEIGHTS_VERSION
ENGINE_OUTPUTS = ("TTI", "pHRR", "THR", "avg_HRR")  # keys of dravix_engine.predict() rows
# patent FIG. 18 index names (spec [00276]-[00281], pp. 96-97); LOI placed with ignition as old DFRS did
INDICES = {"ignition_resistance": ("TTI", "LOI"), "heat_release": ("pHRR", "THR", "avg_HRR"),
           "thermal_stability": ("char_residue",)}


def subscore(name: str, value) -> float:
    """Risk-oriented normalised value in [0, 1] (1 = worst) for one property in UNITS[name]."""
    lo, hi = RANGES[name]
    x = (np.asarray(value, float) - lo) / (hi - lo)
    if DIRECTION[name] < 0:
        x = 1.0 - x
    return np.clip(x, 0.0, 1.0)


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _sigma(s) -> float:
    """Accept a 1-sigma value or a (lo, hi) 95% interval; return 1-sigma."""
    if isinstance(s, (tuple, list)):
        lo, hi = map(float, s)
        s = (hi - lo) / 3.92
    s = float(s)
    if s < 0 or math.isnan(s):
        raise ValueError(f"sigma must be >= 0, got {s}")
    return s


def cfrps(props: dict, sigma: dict | None = None, weights: dict = WEIGHTS,
          n_samples: int = 4000, seed: int = 0) -> dict:
    """props: {name: value in UNITS[name]}; None/NaN or absent = missing; unknown keys ignored.
    sigma: optional {name: 1-sigma (same units) or (lo, hi) 95% interval}; propagated by Monte Carlo
    (normal noise, clipping applied per sample).

    Returns {"score", "sigma", "interval95": (lo, hi), "coverage", "used", "missing",
             "subscores", "indices", "weights_used"}; score/sigma/interval on the 0-100 risk scale,
    subscores/indices in [0, 1] (1 = worst), indices None when none of their inputs are present."""
    if any(w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("weights must be >= 0 and sum to 1")
    if set(weights) - set(PROPERTIES):
        raise ValueError(f"unknown weight keys: {sorted(set(weights) - set(PROPERTIES))}")
    used = [k for k in weights if weights[k] > 0 and _present(props.get(k))]
    if not used:
        raise ValueError("no CFRPS inputs present")
    missing = [k for k in weights if weights[k] > 0 and k not in used]
    coverage = sum(weights[k] for k in used)
    w = {k: weights[k] / coverage for k in used}
    x = {k: float(props[k]) for k in used}
    sub = {k: float(subscore(k, x[k])) for k in used}
    score = 100.0 * sum(w[k] * sub[k] for k in used)
    indices = {}
    for name, members in INDICES.items():
        m = [k for k in members if k in w]
        indices[name] = sum(w[k] * sub[k] for k in m) / sum(w[k] for k in m) if m else None

    sd = {k: _sigma(sigma[k]) for k in used if sigma and sigma.get(k) is not None}
    if any(sd.values()):
        rng = np.random.default_rng(seed)
        draws = 100.0 * sum(w[k] * subscore(k, x[k] + rng.normal(0.0, sd.get(k, 0.0), n_samples)) for k in used)
        spread, interval = float(draws.std()), tuple(float(q) for q in np.percentile(draws, [2.5, 97.5]))
    else:
        spread, interval = 0.0, (score, score)
    return {"score": score, "sigma": spread, "interval95": interval, "coverage": coverage,
            "used": used, "missing": missing, "subscores": sub, "indices": indices, "weights_used": w}


# ---------------------------------------------------------------- old engine DFRS (verbatim logic)
def _get_numeric(payload, *keys):
    for key in keys:
        value = payload.get(key)
        try:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        return numeric
    return None


def _normalize(value, low, high, invert=False):
    if value is None:
        return 0.5
    normalized = 0.5 if np.isclose(high, low) else (float(value) - low) / (high - low)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return float(1.0 - normalized) if invert else normalized


def dfrs_v1(payload: dict, model_score: float) -> dict:
    """Old DFRS (fire RESISTANCE, 0-1, higher = safer) exactly as dravix-engine computed it.
    payload: material descriptors (snake_case or the old display names); model_score: the old ML
    model's Base_Resistance_Target prediction. Adds the old risk_score = (1 - DFRS) * 100 and
    resistance_index = DFRS * 100 (api/main.py:174-179)."""
    g = lambda *k: _get_numeric(payload, *k)  # noqa: E731
    mp, ait = g("melting_point", "Melting Point (°C)"), g("autoignition_temp", "Autoignition Temp (°C)")
    fp, td = g("flash_point", "Flash Point (°C)"), g("decomposition_temp", "Decomp. Temp (°C)")
    tg, cp = g("glass_transition_temp", "Glass Transition Temp (°C)"), g("specific_heat", "Specific Heat (J/g-°C)")
    rho, char = g("density", "Density (g/cc)"), g("char_yield", "Char Yield (%)")
    k, hoc = g("thermal_conductivity", "Thermal Cond. (W/m-K)"), g("heat_of_combustion", "Heat of Combustion (MJ/kg)")
    fsi, ds = g("flame_spread_index", "Flame Spread Index"), g("smoke_density", "Smoke Density (Ds)")
    loi = g("limiting_oxygen_index", "Limiting Oxygen Index (%)")

    ign = float(np.mean([_normalize(mp, 100.0, 2200.0), _normalize(ait, 200.0, 900.0),
                         _normalize(fp, 100.0, 800.0), _normalize(loi, 10.0, 100.0)]))
    therm = float(np.mean([_normalize(td, 150.0, 2200.0), _normalize(cp, 0.2, 2.0),
                           _normalize(rho, 0.2, 5.0), _normalize(char, 0.0, 80.0)]))
    margin = _normalize(None if td is None or tg is None else td - tg, 20.0, 1500.0)
    heat = float(np.mean([_normalize(k, 0.01, 35.0, True), _normalize(hoc, 0.0, 40.0, True),
                          _normalize(fsi, 0.0, 100.0, True), _normalize(ds, 0.0, 500.0, True)]))
    dfrs = float(np.clip(0.45 * float(np.clip(model_score, 0.0, 1.0)) + 0.2 * ign + 0.2 * therm
                         + 0.1 * margin + 0.05 * heat, 0.0, 1.0))
    c = float(np.clip(dfrs, 0.0, 1.0))
    return {"DFRS": dfrs, "ignition_resistance": ign, "thermal_persistence": therm,
            "decomposition_margin": margin, "heat_propagation_risk": heat,
            "risk_score": (1.0 - c) * 100.0, "resistance_index": c * 100.0}
