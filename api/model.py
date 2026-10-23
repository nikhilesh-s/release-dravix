"""Load Arnav's engine with a compact model artifact so it fits Render's free plan (512 MB).

The engine bundle written by `engine/dravix_engine.py train` is ~106 MB on disk and ~700 MB in RAM,
almost all of it the four 500-tree ExtraTrees forests. `compact()` swaps each fitted ExtraTrees for
`CompactForest`, which holds the same trees as three flat arrays (~13 bytes/node instead of ~72).
Predictions are unchanged: X is cast to float32 as sklearn does, thresholds are rounded DOWN to the
nearest float32 (for float32 x, x <= t64 iff x <= floor32(t64)), leaf values stay float64.

Rebuild the artifact: python3 api/build_model.py   (trains with the engine, then compacts)
"""
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import dravix_engine as E  # noqa: E402  (also puts data/v2 on sys.path for the pickled Avg class)

ARTIFACT = ROOT / "api/dravix_engine_compact.joblib"
LEAF = 255  # feature code for a leaf; the engine's ExtraTrees see 183 features, so uint8 fits


class CompactForest:
    """Read-only replacement for a fitted sklearn ExtraTreesRegressor (single output)."""

    def __init__(self, et):
        trees = [e.tree_ for e in et.estimators_]
        sizes = np.array([t.node_count for t in trees])
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        assert et.n_features_in_ < LEAF and et.n_outputs_ == 1
        feat, right, val = [], [], []
        for t, o in zip(trees, offs):
            leaf = t.children_left < 0
            internal = np.nonzero(~leaf)[0]
            # depth-first builder: left child is always the next node; we rely on it below
            assert (t.children_left[internal] == internal + 1).all()
            thr = t.threshold.astype(np.float32)
            up = thr.astype(np.float64) > t.threshold
            thr[up] = np.nextafter(thr[up], np.float32(-np.inf))
            feat.append(np.where(leaf, LEAF, t.feature).astype(np.uint8))
            right.append(np.where(leaf, 0, t.children_right + o).astype(np.int32))
            val.append(np.where(leaf, t.value[:, 0, 0], thr.astype(np.float64)))
        self.roots = offs.astype(np.int32)
        self.feat, self.right, self.val = np.concatenate(feat), np.concatenate(right), np.concatenate(val)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        node = np.tile(self.roots, (len(X), 1))
        rows = np.arange(len(X))[:, None]
        while True:
            f = self.feat[node]
            live = f != LEAF
            if not live.any():
                return self.val[node].mean(axis=1)
            left = X[rows, np.where(live, f, 0)] <= self.val[node]
            node = np.where(live, np.where(left, node + 1, self.right[node]), node)


class CompactPipe:
    """Imputer + CompactForest; stands in for make_pipeline(imp(), ExtraTreesRegressor)."""

    def __init__(self, pipe):
        self.imputer, self.forest = pipe[0], CompactForest(pipe[-1])

    def predict(self, X):
        return self.forest.predict(self.imputer.transform(X))


def compact(bundle):
    for m in bundle["models"].values():
        m["model"].fitted_[1] = CompactPipe(m["model"].fitted_[1])  # members: HGB, ExtraTrees, Ridge
    return bundle


def data_sha256():
    return hashlib.sha256(Path(E.T.SNAP).read_bytes()).hexdigest()


def load(path=ARTIFACT):
    """Load the compact bundle and hand it to the engine; returns its metadata."""
    b = joblib.load(path)
    E._B = b
    return {**b["meta"], "matrices": b["mat_cats"], "forms": b["form_cats"]}
