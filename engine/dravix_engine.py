"""Dravix engine: formulation -> cone-calorimeter predictions (TTI, pHRR, THR, avg HRR).

Model = the declared litcone primary blend (HGB + ExtraTrees + Ridge on log1p, v3 features,
predictions clipped to the training range), fit on the full scored snapshot
data/v2/cone_lit_benchmark_v1.csv (4,529 rows / 806 papers).

Held-out accuracy (from experiments/runs/litcone_gt_round1_FINAL.log, primary, v3):
quote P2 (random 5x10) / P3 (paper-grouped 5x10), never the in-sample fit.

    python3 engine/dravix_engine.py train
    python3 engine/dravix_engine.py predict --json engine/example.json
"""
import argparse, json, sys
from pathlib import Path

import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/v2"))
import train_litcone_gt_round1 as T  # noqa: E402

T.TOPN, T.PHYS, T.CLIP = 100, True, True
BUNDLE = ROOT / "engine/dravix_engine.joblib"


def _cmap():
    cm = pd.read_csv(T.CLASS_MAP, dtype=str)
    return dict(zip(cm["component_name"].str.strip().str.lower(), cm["class"]))


def _frame(rows):
    df = pd.DataFrame(rows)
    for c in T.NUM:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["is_control"] = pd.to_numeric(df.get("is_control", 0), errors="coerce").fillna(0)
    df["group"] = df.get("group", "query")
    return df


def train():
    df = pd.read_csv(T.SNAP, low_memory=False)
    for c in T.NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cmap = _cmap()
    ppm = df.groupby("matrix_canonical")["group"].nunique()
    mat_cats = sorted(ppm[ppm >= 3].index.difference(["other"]))
    form_cats = sorted(df["material_form"].dropna().unique())
    # freeze the top-100 additive vocabulary so query rows get identical columns
    X = T.build_X(df, cmap, mat_cats, form_cats)
    papers = {}
    for g, comp in zip(df["group"], df["components"]):
        for name, role, v, u in T.parse_components(comp):
            if role not in ("matrix", "curing_agent") and cmap.get(name.strip().lower()) != "not_additive":
                papers.setdefault(T._norm(name), set()).add(g)
    vocab = sorted(papers, key=lambda k: -len(papers[k]))[:T.TOPN]
    models = {}
    for tname, tcol in T.TARGETS.items():
        m = ((df[tcol] > 0) & df["heat_flux_kWm2"].notna()).to_numpy()
        y = df.loc[m, tcol].to_numpy(float)
        mdl = T.primary().fit(X[m].to_numpy(float), np.log1p(y))
        models[tname] = dict(model=mdl, ymin=float(y.min()), ymax=float(y.max()), n=int(m.sum()))
        print(f"trained {tname}: {m.sum()} rows")
    joblib.dump(dict(models=models, columns=list(X.columns), mat_cats=mat_cats, form_cats=form_cats,
                     vocab=vocab, cmap=cmap), BUNDLE, compress=3)
    print(f"wrote {BUNDLE}")


_B = None


def predict(rows):
    """rows: list of dicts with matrix_canonical, material_form, components ('name|role|amount; ...'),
    heat_flux_kWm2, thickness_mm, optional density_kgm3, fr_total_wtpct, filler_total_wtpct."""
    global _B
    _B = _B or joblib.load(BUNDLE)
    df = _frame(rows)
    T.TOPN = 0
    X = T.build_X(df, _B["cmap"], _B["mat_cats"], _B["form_cats"])
    idx = {k: i for i, k in enumerate(_B["vocab"])}
    A = np.zeros((len(df), len(idx)))
    for i, comp in enumerate(df["components"]):
        for name, role, v, u in T.parse_components(comp):
            j = idx.get(T._norm(name))
            if j is not None and role != "matrix" and not np.isnan(v):
                A[i, j] = v if u == "wt%" else 1.0
    X = pd.concat([X, pd.DataFrame(A, columns=[f"a_{i}" for i in range(len(idx))], index=df.index)], axis=1)
    X = X.reindex(columns=_B["columns"], fill_value=0.0).to_numpy(float)
    out = []
    for i in range(len(df)):
        r = {}
        for t, b in _B["models"].items():
            p = float(np.expm1(b["model"].predict(X[i:i + 1]))[0])
            r[t] = round(float(np.clip(p, b["ymin"], b["ymax"])), 2)
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--json", help="file with one formulation dict or a list of them")
    a = ap.parse_args()
    if a.cmd == "train":
        train()
    else:
        d = json.loads(Path(a.json).read_text())
        print(json.dumps(predict(d if isinstance(d, list) else [d]), indent=2))
