"""Train the engine (engine/dravix_engine.py), compact it, write api/dravix_engine_compact.joblib.

    python3 api/build_model.py [--out PATH]

Then checks that the compact model predicts the same as the full one on 200 training rows.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from api import model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(model.ARTIFACT))
a = ap.parse_args()

E = model.E
with tempfile.TemporaryDirectory() as d:
    E.BUNDLE = Path(d) / "full.joblib"  # keep the 106 MB full bundle out of the repo
    E.train()
    full = joblib.load(E.BUNDLE)

# engine predicts row by row (~0.5 s/row with the full forests), so check a fixed sample
rows = pd.read_csv(E.T.SNAP, low_memory=False).sample(200, random_state=0).to_dict("records")
E._B = full
ref = pd.DataFrame(E.predict(rows))
small = model.compact(full)
small["meta"] = {"data_sha256": model.data_sha256(), "sklearn": sklearn.__version__,
                 "rows_per_target": {t: m["n"] for t, m in small["models"].items()}}
E._B = small
diff = float(np.abs(pd.DataFrame(E.predict(rows)).to_numpy() - ref.to_numpy()).max())
print(f"max |compact - full| over {len(rows)} training rows: {diff}")
assert diff <= 0.011, diff  # outputs are rounded to 0.01; allow a last-digit rounding flip
joblib.dump(small, a.out, compress=("xz", 6))
print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
