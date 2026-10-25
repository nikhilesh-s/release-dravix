#!/usr/bin/env python3
"""Step-0 shim for experiments/configs/samedata_huang_dopo_ep_ml_cone.json (PHRR, THR). Runs ONLY in the authors'
pinned env (~/dravix/.venv_dopo) with `python -I`, and imports only their common/ package from the clone
at e550e4a (data/v2/raw/dopo_ep_ml/code). Adapted from data/v2/export_dopo_folds.py (LOI/UL-94 round).

  meta   TASK              row order, y, Reference + molecule groups, loading, ctrl (EP_matrix_<TASK>) -> <TASK>_meta.npz
  gate   TASK              r0_full = their evaluate_task_nested(molecule, fixed, baseline_inclusive, no BDE, 5x5, rs 42)
  rerun  TASK NAME [NAME]  r0_full with only its outer folds replaced by split NAME (P1_mol_rX: molecule; P2_rX: reference)
  export TASK              per split: their front end + r0 (their GBDT) fitted on the training rows -> post-front-end matrices
"""
import json, sys, tempfile, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path("~/dravix-dopo-cone")
CLONE = WT / "data/v2/raw/dopo_ep_ml/code"
sys.path.insert(0, str(CLONE))
from common import scientific_evaluation as se  # noqa: E402
from common.literature_feature_views import get_groups  # noqa: E402
from common.split_strategies import Fold, build_group_labels  # noqa: E402

warnings.filterwarnings("ignore")
DATA = CLONE / "data/DOPO_EP_new.csv"  # sha256 6970ac8d...
OUT = WT / "experiments/runs/samedata_dopo_cone_folds"
SPLITS = WT / "experiments/runs/samedata_huang_dopo_ep_ml_cone_splits.json"
PUB = CLONE / "results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5/{t}/molecule/formulation/without_BDE/{t}_outer_predictions.csv"
GATE = {"PHRR": (0.594, 0.614), "THR": (0.473, 0.493)}
SEED = 42


def task_frame(bundle, task):
    """evaluate_task_nested's row filter, matrix (current view), neat masking and group labels, no fit."""
    cfg = se.TASK_CONFIGS[task]
    cm = bundle.base.colmap
    y = se.get_target(bundle.base, task)
    valid = se.core.build_task_valid_mask(bundle.df, cm, task, target=y)
    loading = pd.to_numeric(bundle.df[cm.get("Loading_total_FR wt%", "Loading_total_FR wt%")], errors="coerce")
    valid = valid & loading.ge(0)  # baseline_inclusive
    df = bundle.df.loc[valid].reset_index(drop=True)
    ref = build_group_labels(df, strategy="reference", reference_col=cm.get("Reference", "Reference"))
    mol = get_groups(bundle.base, task).loc[valid].reset_index(drop=True)
    X = se._filter_task_matrix(bundle, task, cfg.current_view, use_bde=False, screening_mode="formulation",
                               feature_scope="all").loc[valid].reset_index(drop=True)
    X = se._apply_baseline_inclusive_mask(X, df, cm, task, "baseline_inclusive")
    b = se.MATCHING_EP_BASELINE[task]
    ctrl = pd.to_numeric(df[cm.get(b, b)], errors="coerce")
    return cfg, X, y.loc[valid].reset_index(drop=True), df, ref, mol, loading.loc[valid].reset_index(drop=True), ctrl


def meta(task):
    bundle = se.prepare_scientific_bundle(DATA)
    cfg, X, y, df, ref, mol, load, ctrl = task_frame(bundle, task)
    np.savez(OUT / f"{task}_meta.npz", record_id=df["Record_ID"].to_numpy(dtype=str), y=y.to_numpy(float),
             ref=ref.to_numpy(dtype=str), mol=mol.to_numpy(dtype=str), loading=load.to_numpy(float),
             ctrl=ctrl.to_numpy(float))
    neat = load == 0
    print(f"{task}: rows {len(y)}, raw features {X.shape[1]} (view {cfg.current_view}, K {cfg.current_k}, "
          f"{cfg.descriptor_mode}, V7 {cfg.use_v7_features}), refs {ref.nunique()}, molecule groups {mol.nunique()}, "
          f"neat {int(neat.sum())} (molecule groups among neat: {mol[neat].nunique()}), "
          f"modified with ctrl {int(((load > 0) & (ctrl > 0)).sum())}; "
          f"ctrl columns in raw X: {[c for c in X.columns if se.MATCHING_EP_BASELINE[task].lower() in str(c).lower()]}")


def _patch_outer(task, sets, n):
    orig = se.iter_cv_folds

    def patched(X, y, *, groups, n_splits, random_state, classification):
        if len(y) == n and random_state == SEED:  # the one outer call on all rows
            for k, te in enumerate(sets, start=1):
                te = np.asarray(te)
                yield Fold(np.setdiff1d(np.arange(n), te), te, k)
        else:
            yield from orig(X, y, groups=groups, n_splits=n_splits, random_state=random_state,
                            classification=classification)
    se.iter_cv_folds = patched
    return orig


def nested(bundle, task, strategy):
    opts = se.EvaluationOptions(split_strategy=strategy, selection_scope="fixed", row_policy="baseline_inclusive",
                                use_bde=False, outer_splits=5, inner_splits=5, random_state=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        folds, preds, _ = se.evaluate_task_nested(bundle, task, tmp, opts)
    return folds, preds


def gate(task):
    bundle = se.prepare_scientific_bundle(DATA)
    t = time.time()
    folds, preds = nested(bundle, task, "molecule")
    v = folds["outer_R2"].to_numpy(float)
    lo, hi = GATE[task]
    pub = pd.read_csv(str(PUB).format(t=task), encoding="utf-8-sig")
    m = preds[["Record_ID", "outer_fold", "y_pred"]].merge(pub[["Record_ID", "outer_fold", "y_pred"]], on="Record_ID",
                                                           suffixes=("", "_pub"))
    out = dict(task=task, per_fold=v.tolist(), mean=float(v.mean()), sd=float(v.std(ddof=1)),
               MAE_per_fold=folds["outer_MAE"].astype(float).tolist(),
               modified_R2_per_fold=folds["outer_modified_R2"].astype(float).tolist(), gate=[lo, hi],
               passed=bool(lo <= v.mean() <= hi),
               folds_equal_published=bool(len(m) == len(pub) == len(preds) and (m.outer_fold == m.outer_fold_pub).all()),
               max_abs_pred_diff_vs_published=float(np.abs(m.y_pred - m.y_pred_pub).max()),
               selected_models=folds["selected_model"].tolist(), minutes=round((time.time() - t) / 60, 2))
    (OUT / f"{task}_gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def rerun(task, names):
    bundle = se.prepare_scientific_bundle(DATA)
    rid = np.load(OUT / f"{task}_meta.npz")["record_id"]
    S = json.loads(SPLITS.read_text())[task]
    assert S["record_id"] == rid.tolist()
    d = OUT / task / "r0_full"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (d / f"{name}.json").exists():
            continue
        t = time.time()
        orig = _patch_outer(task, S["splits"][name], len(rid))
        try:
            folds, preds = nested(bundle, task, "reference" if name.startswith("P2") else "molecule")
        finally:
            se.iter_cv_folds = orig
        p = preds.set_index("Record_ID").loc[rid]
        assert len(folds) == 5 and p["outer_fold"].notna().all()
        out = dict(name=name, per_fold_R2=folds["outer_R2"].astype(float).tolist(),
                   selected_models=folds["selected_model"].tolist(), oof=p["y_pred"].astype(float).tolist(),
                   outer_fold=p["outer_fold"].astype(int).tolist(), minutes=round((time.time() - t) / 60, 2))
        (d / f"{name}.json").write_text(json.dumps(out))
        print(f"{task} {name}: mean R2 {np.mean(out['per_fold_R2']):.4f} {out['selected_models']} {out['minutes']} min",
              flush=True)


def export(task):
    bundle = se.prepare_scientific_bundle(DATA)
    cfg, X, y, df, ref, mol, load, ctrl = task_frame(bundle, task)
    rid = df["Record_ID"].astype(str).to_numpy()
    S = json.loads(SPLITS.read_text())[task]
    assert S["record_id"] == rid.tolist(), "row order differs from the splits file"
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    kept = {}
    for name, sets in S["splits"].items():
        for f, te in enumerate(sets):
            te = np.asarray(te)
            tr = np.setdiff1d(np.arange(len(y)), te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            pipe = se.build_pipeline(task_type="regression", model_name="GBDT", requested_k=cfg.current_k, seed=SEED)
            pipe.fit(Xtr, y.iloc[tr].reset_index(drop=True))
            names = se._selected_feature_names(pipe)
            kept[f"{name}__f{f}"] = [c for c in names if se.MATCHING_EP_BASELINE[task].lower() in str(c).lower()]
            fr = lambda A: np.asarray(pipe[:5].transform(A), dtype=np.float64)
            np.savez_compressed(d / f"{name}__f{f}.npz", Xtr=fr(Xtr), Xte=fr(Xte), tr=tr, te=te,
                                ytr=y.iloc[tr].to_numpy(float), yte=y.iloc[te].to_numpy(float),
                                r0=np.asarray(pipe.predict(Xte), float))
        print(f"{task} {name}: {len(sets)} folds, n_features {fr(Xtr).shape[1]}, ctrl features kept {kept[f'{name}__f0']}",
              flush=True)
    (OUT / f"{task}_ctrl_features_kept.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mode, task = sys.argv[1], sys.argv[2]
    {"meta": lambda: meta(task), "gate": lambda: gate(task), "rerun": lambda: rerun(task, sys.argv[3:]),
     "export": lambda: export(task)}[mode]()
