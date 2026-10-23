"""Tests for api/uncertainty.py (needs api/uncertainty.joblib). Run: python -m pytest api/test_uncertainty.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine/score"))
from api import model, uncertainty as UQ  # noqa: E402
from cfrps import cfrps  # noqa: E402

model.load()
B = model.E._B
U = UQ.Uncertainty()
EXAMPLES = json.loads((ROOT / "engine/example.json").read_text())
EP_APP = {"matrix_canonical": "EP", "material_form": "bulk", "components": "EP|matrix|80 wt%; APP|FR|20 wt%",
          "heat_flux_kWm2": 50, "thickness_mm": 3, "fr_total_wtpct": 20}
# ENGINE_QA: an 8-additive recipe returned pHRR 28 kW/m2 with no warning
EIGHT = dict(EP_APP, components="EP|matrix|60 wt%; APP|FR|5 wt%; MPP|FR|5 wt%; ZB|synergist|5 wt%; "
             "LDH|filler|5 wt%; MMT|nanofiller|5 wt%; ATH|filler|5 wt%; EG|FR|5 wt%; TiO2|filler|5 wt%")


def test_features_match_engine_prediction():
    rows = EXAMPLES + [EP_APP, EIGHT]
    X = UQ.features(model.E._frame(rows), B)
    served = model.E.predict(rows)
    for t, b in B["models"].items():
        mu = UQ.members(b["model"], X).mean(0)
        mine = np.round(np.clip(np.expm1(mu), b["ymin"], b["ymax"]), 2)
        assert np.allclose(mine, [r[t] for r in served], atol=0.011), t


def test_intervals_nest_and_contain_point():
    rows = EXAMPLES + [EP_APP, EIGHT]
    served = model.E.predict(rows)
    unc, _, _ = U.annotate(rows, B)
    for p, u in zip(served, unc):
        for t, v in u.items():
            (l8, h8), (l9, h9) = v["interval_80"], v["interval_90"]
            assert l9 <= l8 <= p[t] <= h8 <= h9, (t, v, p[t])


def test_ood_flags():
    unc, _, _ = U.annotate([EXAMPLES[0], EIGHT, dict(EP_APP, matrix_canonical="XYZ"), dict(EP_APP, heat_flux_kWm2=150)], B)
    assert not any(v["ood"] for v in unc[0].values()), unc[0]
    assert all(v["ood"] for v in unc[1].values()), unc[1]
    assert all(any("matrix" in r for r in v["ood_reasons"]) for v in unc[2].values())
    assert all(any("heat flux" in r for r in v["ood_reasons"]) for v in unc[3].values())


def test_cfrs_interval_contains_point_and_is_monotone():
    _, mu, h90 = U.annotate([EXAMPLES[0]], B)
    c = cfrps(model.E.predict([EXAMPLES[0]])[0])
    iv = U.cfrs(c["score"], mu[0], h90[0], B, c["weights_used"])
    assert iv["interval_90"][0] <= iv["interval_80"][0] <= c["score"] <= iv["interval_80"][1] <= iv["interval_90"][1]
    # raising pHRR (more risk) or TTI (less risk) with sigma fixed moves every draw, so the bounds move the same way
    j = {t: U.U["targets"].index(t) for t in ("pHRR", "TTI")}
    prev = None
    for shift in np.linspace(-1, 1, 9):
        m = mu[0].copy()
        m[j["pHRR"]] += shift
        b = U.cfrs(c["score"], m, h90[0], B, c["weights_used"])
        cur = (b["interval_90"][0], b["interval_90"][1])
        assert prev is None or (cur[0] >= prev[0] - 1e-9 and cur[1] >= prev[1] - 1e-9)
        prev = cur
    up = mu[0].copy()
    up[j["TTI"]] += 0.5
    b = U.cfrs(c["score"], up, h90[0], B, c["weights_used"])
    a = U.cfrs(c["score"], mu[0], h90[0], B, c["weights_used"])
    assert b["interval_90"][1] <= a["interval_90"][1] + 1e-9
