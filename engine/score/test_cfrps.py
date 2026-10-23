"""Tests for engine/score/cfrps.py. Run: python -m pytest engine/score"""
import math

import pytest

from cfrps import (DIRECTION, ENGINE_OUTPUTS, INDICES, PROPERTIES, RANGES, UNITS, WEIGHTS, cfrps,
                   dfrs_v1, load_weights, subscore)

# illustrative values shaped like dravix_engine.predict() rows for engine/example.json
# (PP + 20 wt% APP vs neat PP); hand-picked, not model output. In UNITS.
FR = {"TTI": 45.0, "pHRR": 350.0, "THR": 70.0, "avg_HRR": 150.0}
NEAT = {"TTI": 35.0, "pHRR": 1100.0, "THR": 110.0, "avg_HRR": 380.0}


def test_units_and_ranges():
    assert UNITS == {"TTI": "s", "pHRR": "kW/m2", "THR": "MJ/m2", "avg_HRR": "kW/m2",
                     "char_residue": "wt%", "LOI": "vol% O2"}
    for k, (lo, hi) in RANGES.items():
        assert 0 <= lo < hi, k
    assert RANGES["char_residue"][1] <= 100 and RANGES["LOI"][1] <= 100
    assert set(ENGINE_OUTPUTS) <= set(PROPERTIES)


def test_normalization_endpoints_and_direction():
    for k, (lo, hi) in RANGES.items():
        worst, best = (hi, lo) if DIRECTION[k] > 0 else (lo, hi)
        assert subscore(k, worst) == 1.0 and subscore(k, best) == 0.0
        assert subscore(k, (lo + hi) / 2) == pytest.approx(0.5)
        # clipping beyond the reference range
        assert subscore(k, hi + 10 * (hi - lo)) in (0.0, 1.0)
        assert subscore(k, lo - 10 * (hi - lo)) in (0.0, 1.0)
    assert DIRECTION == {"TTI": -1, "char_residue": -1, "LOI": -1, "pHRR": 1, "THR": 1, "avg_HRR": 1}


def test_weights_sum_to_one_and_cover_all_properties():
    assert math.isclose(sum(WEIGHTS.values()), 1.0)
    assert set(WEIGHTS) == set(PROPERTIES) and all(w > 0 for w in WEIGHTS.values())
    with pytest.raises(ValueError):
        cfrps(FR, weights={"pHRR": 0.5, "TTI": 0.4})


@pytest.mark.parametrize("name", list(PROPERTIES))
def test_monotonic_worse_property_never_lowers_risk(name):
    lo, hi = RANGES[name]
    base = dict(FR, char_residue=15.0, LOI=26.0)
    scores = []
    for t in [i / 20 for i in range(-5, 26)]:  # sweep past both ends
        v = lo + t * (hi - lo)
        scores.append(cfrps(dict(base, **{name: v}))["score"])
    if DIRECTION[name] < 0:
        scores = scores[::-1]  # order from best to worst value
    assert all(b >= a - 1e-12 for a, b in zip(scores, scores[1:])), name
    assert scores[-1] > scores[0]


def test_fr_formulation_scores_lower_than_neat():
    # patent [0082]: lower CFRS = lower relative fire susceptibility
    assert cfrps(FR)["score"] < cfrps(NEAT)["score"]


def test_patent_indices_partition_inputs():
    members = [k for m in INDICES.values() for k in m]
    assert sorted(members) == sorted(PROPERTIES)
    r = cfrps(FR)
    assert r["indices"]["thermal_stability"] is None  # char not given
    assert r["indices"]["ignition_resistance"] == pytest.approx(r["subscores"]["TTI"])
    assert all(0 <= v <= 1 for v in r["indices"].values() if v is not None)


def test_missing_inputs_renormalize_and_report_coverage():
    r = cfrps(FR)  # engine outputs only: no char/LOI
    assert r["coverage"] == pytest.approx(sum(WEIGHTS[k] for k in ENGINE_OUTPUTS))
    assert cfrps(FR, weights=load_weights("v0"))["coverage"] == pytest.approx(0.80)
    assert sorted(r["missing"]) == ["LOI", "char_residue"]
    assert math.isclose(sum(r["weights_used"].values()), 1.0)
    r2 = cfrps({"pHRR": 1300.0, "TTI": None, "THR": float("nan")})
    assert r2["used"] == ["pHRR"] and r2["score"] == pytest.approx(100.0)
    assert r2["coverage"] == pytest.approx(WEIGHTS["pHRR"])
    with pytest.raises(ValueError):
        cfrps({"TTI": None})
    assert cfrps(dict(FR, unrelated=5))["score"] == cfrps(FR)["score"]


def test_bounded_output():
    extremes = [{k: v * s for k, v in FR.items()} for s in (0.0, 1e-6, 1e6)]
    extremes.append({k: -1e9 for k in PROPERTIES})
    for p in extremes:
        r = cfrps(p, sigma={k: 1e6 for k in p})
        assert 0.0 <= r["score"] <= 100.0
        assert 0.0 <= r["interval95"][0] <= r["interval95"][1] <= 100.0
    best = {k: (RANGES[k][0] if DIRECTION[k] > 0 else RANGES[k][1]) for k in PROPERTIES}
    worst = {k: (RANGES[k][1] if DIRECTION[k] > 0 else RANGES[k][0]) for k in PROPERTIES}
    assert cfrps(best)["score"] == 0.0 and cfrps(worst)["score"] == 100.0


def test_uncertainty_zero_sigma_zero_spread():
    r = cfrps(FR, sigma={k: 0.0 for k in FR})
    assert r["sigma"] == 0.0 and r["interval95"] == (r["score"], r["score"])
    assert cfrps(FR)["sigma"] == 0.0


def test_uncertainty_grows_with_sigma():
    spreads = [cfrps(FR, sigma={k: f * v for k, v in FR.items()})["sigma"] for f in (0.02, 0.05, 0.10, 0.20)]
    assert all(b > a for a, b in zip(spreads, spreads[1:]))
    r = cfrps(FR, sigma={"pHRR": 50.0})
    lo, hi = r["interval95"]
    assert lo < r["score"] < hi
    # linear check inside the unclipped region: d score / d pHRR = 100 * w / (hi - lo)
    w = r["weights_used"]["pHRR"]
    assert r["sigma"] == pytest.approx(100 * w * 50.0 / (RANGES["pHRR"][1] - RANGES["pHRR"][0]), rel=0.05)


def test_interval_input_equals_sigma_input():
    a = cfrps(FR, sigma={"pHRR": 50.0})
    b = cfrps(FR, sigma={"pHRR": (350.0 - 1.96 * 50.0, 350.0 + 1.96 * 50.0)})
    assert a["sigma"] == pytest.approx(b["sigma"])
    with pytest.raises(ValueError):
        cfrps(FR, sigma={"pHRR": -1.0})


# ---------------------------------------------------------------- old DFRS
# golden values computed by running nikhilesh-s/dravix-engine app/services/scoring_engine.py
# (commit 68928945) on these payloads; API example from docs/api-specification.md:31-49,
# Kevlar 49 from data/benchmark_materials.csv:2 (target_dfrs used as model_score).
API_EXAMPLE = {"density": 1.25, "specific_heat": 1.50, "thermal_conductivity": 0.21, "flash_point": 320.0,
               "autoignition_temperature": 450.0, "limiting_oxygen_index": 19.0, "char_yield": 12.0,
               "heat_of_combustion": 28.0, "flame_spread_index": 40.0, "smoke_density": 120.0}
KEVLAR = {"density": 1.44, "thermal_conductivity": 0.04, "specific_heat": 1.42, "decomposition_temp": 500.0}
FULL = {"Density (g/cc)": 1.2, "Melting Point (°C)": 260, "Thermal Cond. (W/m-K)": 0.3,
        "Specific Heat (J/g-°C)": 1.3, "Decomp. Temp (°C)": 400, "Autoignition Temp (°C)": 480,
        "Flash Point (°C)": 350, "Heat of Combustion (MJ/kg)": 25, "Limiting Oxygen Index (%)": 28,
        "Smoke Density (Ds)": 200, "Char Yield (%)": 30, "Flame Spread Index": 25,
        "Glass Transition Temp (°C)": 150}


@pytest.mark.parametrize("payload,model_score,golden", [
    (API_EXAMPLE, 0.6, {"DFRS": 0.5034414478399725, "ignition_resistance": 0.3535714285714286,
                        "thermal_persistence": 0.39774305555555556, "decomposition_margin": 0.5,
                        "heat_propagation_risk": 0.6635710202915119}),
    (KEVLAR, 0.6092742450063383, {"DFRS": 0.5357548338264478, "ignition_resistance": 0.5,
                                  "thermal_persistence": 0.40171070460704605, "decomposition_margin": 0.5,
                                  "heat_propagation_risk": 0.6247856530437268}),
    (FULL, 0.55, {"DFRS": 0.41448588937617387, "ignition_resistance": 0.25833333333333336,
                  "thermal_persistence": 0.3290989159891599, "decomposition_margin": 0.1554054054054054,
                  "heat_propagation_risk": 0.6791779794226922}),
    ({}, 0.5, {"DFRS": 0.5, "ignition_resistance": 0.5, "thermal_persistence": 0.5,
               "decomposition_margin": 0.5, "heat_propagation_risk": 0.5}),
])
def test_dfrs_v1_reproduces_old_engine(payload, model_score, golden):
    r = dfrs_v1(payload, model_score)
    for k, v in golden.items():
        assert r[k] == pytest.approx(v, abs=1e-12), k
    assert r["risk_score"] == pytest.approx((1 - golden["DFRS"]) * 100)
    assert r["resistance_index"] == pytest.approx(golden["DFRS"] * 100)


@pytest.mark.parametrize("key,lo,hi", [("LOI", 19.0, 35.0), ("char_residue", 5.0, 40.0)])
def test_direction_agrees_with_dfrs_on_shared_inputs(key, lo, hi):
    """LOI and char are the only inputs both scores share: raising either must lower risk in both."""
    old_key = {"LOI": "limiting_oxygen_index", "char_residue": "char_yield"}[key]
    old = [dfrs_v1(dict(API_EXAMPLE, **{old_key: v}), 0.6)["risk_score"] for v in (lo, hi)]
    new = [cfrps(dict(FR, **{key: v}))["score"] for v in (lo, hi)]
    assert old[1] < old[0] and new[1] < new[0]


@pytest.mark.xfail(strict=True, reason="CFRPS (cone-calorimeter composite) and old DFRS (0.45 x ML resistance "
                   "score + descriptor sub-scores, missing = 0.5) are different definitions; their "
                   "risk values do not agree numerically. See context/decisions/CONFLICTS.md.")
def test_numeric_agreement_with_dfrs_on_shared_inputs():
    old = dfrs_v1({"limiting_oxygen_index": 19.0, "char_yield": 12.0}, 0.5)["risk_score"]
    new = cfrps({"LOI": 19.0, "char_residue": 12.0})["score"]
    assert new == pytest.approx(old, abs=5.0)
