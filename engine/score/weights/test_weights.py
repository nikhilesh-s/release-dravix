"""Tests for the versioned CFRS weights. Run: python -m pytest engine/score (from the repo root)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROPS, load_lit, score_matrix, subscores, ul94_class, weights_vec  # noqa: E402
from cfrps import DIRECTION, RANGES, WEIGHTS, WEIGHTS_DIR, WEIGHTS_VERSION, cfrps, load_weights  # noqa: E402

VERSIONS = ("v0", "v1")


@pytest.mark.parametrize("v", VERSIONS)
def test_config_loads_sums_to_one_with_provenance(v):
    w = load_weights(v)
    assert set(w) == set(PROPS) and math.isclose(sum(w.values()), 1.0) and min(w.values()) >= 0
    cfg = json.loads((WEIGHTS_DIR / f"cfrs_weights_{v}.json").read_text())
    assert cfg["version"] == v and cfg["provenance"]


def test_default_is_v1_and_v0_is_the_old_placeholder_set():
    assert WEIGHTS_VERSION == "v1" and WEIGHTS == load_weights("v1")
    assert load_weights("v0") == {"pHRR": 0.30, "TTI": 0.20, "avg_HRR": 0.20, "THR": 0.10, "LOI": 0.10,
                                  "char_residue": 0.10}
    assert min(WEIGHTS.values()) >= 0.05  # amendment A1-1 floor: every input moves the score


def test_v1_band_contains_weights():
    cfg = json.loads((WEIGHTS_DIR / "cfrs_weights_v1.json").read_text())
    for k, (lo, hi) in cfg["uncertainty_90pct"].items():
        assert lo - 0.005 <= cfg["weights"][k] <= hi + 0.005, k


def test_v1_matches_recorded_calibration():
    r = json.loads((WEIGHTS_DIR / "results/calibration_floor05.json").read_text())["D1"]
    assert r["adoption"]["adopt_fitted"]
    for k, v in r["fitted_weights"].items():
        assert abs(WEIGHTS[k] - v) <= 0.006, k


def test_bad_config_rejected(tmp_path, monkeypatch):
    import cfrps as c
    (tmp_path / "cfrs_weights_bad.json").write_text(json.dumps({"weights": {"pHRR": 0.6, "TTI": 0.6}}))
    monkeypatch.setattr(c, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(ValueError):
        c.load_weights("bad")


@pytest.mark.parametrize("v", VERSIONS)
@pytest.mark.parametrize("name", PROPS)
def test_monotonic_under_each_version(v, name):
    w = load_weights(v)
    base = {"TTI": 45.0, "pHRR": 350.0, "THR": 70.0, "avg_HRR": 150.0, "char_residue": 15.0, "LOI": 26.0}
    lo, hi = RANGES[name]
    s = [cfrps(dict(base, **{name: lo + t / 20 * (hi - lo)}), weights=w)["score"] for t in range(-5, 26)]
    s = s[::-1] if DIRECTION[name] < 0 else s
    assert all(b >= a - 1e-12 for a, b in zip(s, s[1:])) and s[-1] > s[0]


def test_vectorised_scorer_matches_cfrps():
    df = load_lit().iloc[:300]
    s = score_matrix(subscores(df), weights_vec(WEIGHTS))
    for i in np.flatnonzero(~np.isnan(s))[:100]:
        props = {k: (None if np.isnan(df.at[i, k]) else df.at[i, k]) for k in PROPS}
        assert s[i] == pytest.approx(cfrps(props)["score"], abs=1e-9)


@pytest.mark.parametrize("raw,cls", [("V-0", 3), ("V0 (3.2 mm); V2 (1.6 mm)", 3), ("F/F/V-2", 0),
                                     ("V-2/V-2/F", 0), ("V-1", 2), ("HB", 0), ("NR", 0), ("5VA", 3),
                                     ("No rating", 0), ("VTM-0", None), ("", None)])
def test_ul94_parser(raw, cls):
    got = ul94_class(raw)
    assert (np.isnan(got) if cls is None else got == cls)


# ---------------------------------------------------------------- declared stability threshold (A2)
@pytest.fixture(scope="module")
def stab():
    if not (Path(__file__).resolve().parents[3] / "data/v2/cone_replicates_fsri.csv").exists():
        pytest.skip("FSRI-derived tables are not redistributed in the public release")
    from robustness import stability
    return stability(WEIGHTS)


def test_ranking_stability_lit50(stab):
    med, p5 = stab["lit50"]
    assert med >= 0.90 and p5 >= 0.80


@pytest.mark.xfail(strict=True, reason="Declared A2 threshold not met on FSRI 50 kW/m2 for v1 "
                   "(median tau 0.888, p5 0.767; v0 also misses, 0.898/0.805). FSRI has no LOI or "
                   "avg HRR, so the 0.05-0.10 weights on TTI/THR/char carry the ranking and Dir(50 w) "
                   "perturbs them strongly. See context/research/CFRS_WEIGHTS.md.")
def test_ranking_stability_fsri50(stab):
    med, p5 = stab["fsri50"]
    assert med >= 0.90 and p5 >= 0.80
