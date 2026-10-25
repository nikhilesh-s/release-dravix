import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import ahp

HERE = Path(__file__).resolve().parent
W1 = np.array([.20, .25, .10, .15, .10, .12, .08])
W2 = np.array([.10, .15, .20, .20, .15, .10, .10])


def consistent(w):  # a_ij = w_i / w_j, flattened in PAIRS order (non-Saaty ratios are fine here)
    return [w[i] / w[j] for i, j in ahp.PAIRS]


def test_consistent_matrix_recovers_weights():
    A = ahp.matrix(consistent(W1))
    assert np.allclose(ahp.weights(A), W1)
    assert abs(ahp.consistency(A)[2]) < 1e-9


def test_reciprocal_and_sign_conventions():
    A = ahp.matrix([3] + [1] * 20)
    assert np.allclose(A * A.T, 1)
    assert A[0, 1] == 3
    for right in (-3, 1 / 3):
        assert np.isclose(ahp.matrix([right] + [1] * 20)[0, 1], 1 / 3)


def test_ri_n7():
    assert ahp.RI[7] == 1.32


def test_geometric_mean():
    A = ahp.matrix(consistent(W1))
    assert np.allclose(ahp.aggregate([A, A]), A)
    B = ahp.matrix(consistent(W2))
    w = ahp.weights(ahp.aggregate([A, B]))
    lo, hi = np.minimum(W1, W2), np.maximum(W1, W2)
    assert np.all(w >= lo - 1e-9) and np.all(w <= hi + 1e-9) and np.isclose(w.sum(), 1)


def test_inconsistent_matrix_flagged():
    # TTI > pHRR > THR by 9 each, but THR > TTI by 9: a cycle
    j = [1] * 21
    j[ahp.PAIRS.index((0, 1))] = 9
    j[ahp.PAIRS.index((1, 2))] = 9
    j[ahp.PAIRS.index((0, 2))] = -9
    A = ahp.matrix(j)
    assert ahp.consistency(A)[2] > 0.10
    assert {"TTI vs pHRR", "pHRR vs THR", "TTI vs THR"} & {m["pair"] for m in ahp.most_inconsistent(A)}


def test_cfrs_weights_drop_smoke():
    d, smoke = ahp.cfrs_weights(W1)
    assert set(d) == set(ahp.CFRS_KEYS) and np.isclose(sum(d.values()), 1) and np.isclose(smoke, .08)


def test_cli_round_trip(tmp_path):
    out = tmp_path / "w.json"
    subprocess.run([sys.executable, str(HERE / "ahp.py"), str(HERE / "survey/example_responses.csv"),
                    "--out", str(out)], check=True)
    r = json.load(open(out))
    assert len(r["weights"]) == 6 and np.isclose(sum(r["weights"].values()), 1)
    assert all(e["ok"] for e in r["experts"].values()) and len(r["experts"]) == 2
