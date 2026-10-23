#!/usr/bin/env python3
"""Step-0 shim for experiments/configs/samedata_huang_dopo_ep_ml(.json, _ul94.json). Runs ONLY in the authors' pinned
env (~/dravix/.venv_dopo, their requirements.txt) and imports their code from the clone pinned at
e550e4a247d01c837041afd1aa6702f62a8f2ccf (data/v2/raw/dopo_ep_ml_code). Run with `python -I`.

  meta   TASK   row order, targets, Reference and molecule groups, loading (no fit) -> runs/samedata_dopo_folds/<TASK>_meta.npz
  gate   TASK   r0_full = their evaluate_task_nested(..., split_strategy='reference', fixed, baseline_inclusive, no BDE,
                5x5, random_state=42) on DOPO_EP_new.csv; prints the gate and checks its folds vs the published per-row folds
  export TASK   for every split in runs/samedata_huang_dopo_ep_ml_splits.json: their front end (build_pipeline steps
                nonempty -> imputer -> deduplicate -> variance -> selector) + r0 model fitted on the training rows only;
                writes post-front-end train/test matrices + r0 test predictions; LOI also P3 (external, fit on all 599)
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

DRAVIX = Path("~/dravix")
CLONE = DRAVIX / "data/v2/raw/dopo_ep_ml_code"
sys.path.insert(0, str(CLONE))
from common import scientific_evaluation as se  # noqa: E402
from common.literature_feature_views import get_groups  # noqa: E402
from common.split_strategies import build_group_labels  # noqa: E402

warnings.filterwarnings("ignore")
DATA = CLONE / "data/DOPO_EP_new.csv"  # sha256 6970ac8d... == data/sources/incoming/huang_dopo_ep_ml/DOPO_EP_new.csv
OUT = DRAVIX / "experiments/runs/samedata_dopo_folds"
SPLITS = DRAVIX / "experiments/runs/samedata_huang_dopo_ep_ml_splits.json"
PUB = {"LOI": DRAVIX / "data/sources/incoming/huang_dopo_ep_ml/LOI_outer_predictions_reference.csv",
       "UL94_V0": DRAVIX / "data/sources/incoming/huang_dopo_ep_ml/UL94_V0_outer_predictions_reference.csv"}
R0_MODEL = {"LOI": "GBDT", "UL94_V0": "ExtraTrees"}
SEED = 42  # config: GBDT random_state=42 / _classification_model('ExtraTrees', 42)


def task_frame(bundle, task):
    """Exactly evaluate_task_nested's row filter, matrix, masking and group labels (no fit)."""
    cfg = se.TASK_CONFIGS[task]
    y = se.get_target(bundle.base, task)
    valid = se.core.build_task_valid_mask(bundle.df, bundle.base.colmap, task, target=y)
    loading = pd.to_numeric(bundle.df[bundle.base.colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")], errors="coerce")
    valid = valid & loading.ge(0)  # baseline_inclusive
    y_task = y.loc[valid].reset_index(drop=True)
    df_task = bundle.df.loc[valid].reset_index(drop=True)
    ref = build_group_labels(df_task, strategy="reference", reference_col=bundle.base.colmap.get("Reference", "Reference"))
    mol = get_groups(bundle.base, task).loc[valid].reset_index(drop=True)
    X = se._filter_task_matrix(bundle, task, cfg.current_view, use_bde=False, screening_mode="formulation",
                               feature_scope="all").loc[valid].reset_index(drop=True)
    X = se._apply_baseline_inclusive_mask(X, df_task, bundle.base.colmap, task, "baseline_inclusive")
    return cfg, X, y_task, df_task, ref, mol, loading.loc[valid].reset_index(drop=True)


def meta(task):
    bundle = se.prepare_scientific_bundle(DATA)
    cfg, X, y, df, ref, mol, load = task_frame(bundle, task)
    pair = df["Canonical_main_co_pair"].fillna("").astype(str) if "Canonical_main_co_pair" in df else None
    same = None
    if pair is not None:  # do get_groups() labels and Canonical_main_co_pair induce the same partition?
        same = bool((pd.factorize(mol)[0] == pd.factorize(pair)[0]).all())
    np.savez(OUT / f"{task}_meta.npz", record_id=df["Record_ID"].to_numpy(dtype=str), y=y.to_numpy(float),
             ref=ref.to_numpy(dtype=str), mol=mol.to_numpy(dtype=str), loading=load.to_numpy(float))
    print(f"{task}: rows {len(y)}, raw features {X.shape[1]} (view {cfg.current_view}, K {cfg.current_k}), "
          f"refs {ref.nunique()}, molecule groups {mol.nunique()}, neat {int((load == 0).sum())}; "
          f"get_groups == Canonical_main_co_pair partition: {same}")


def gate(task, tmp):
    bundle = se.prepare_scientific_bundle(DATA)
    t = time.time()
    opts = se.EvaluationOptions(split_strategy="reference", selection_scope="fixed", row_policy="baseline_inclusive",
                                use_bde=False, outer_splits=5, inner_splits=5, random_state=42)
    folds, preds, summ = se.evaluate_task_nested(bundle, task, tmp, opts)
    col = "outer_R2" if task == "LOI" else "outer_Macro_F1"
    v = folds[col].to_numpy(float)
    lo, hi = (0.699, 0.719) if task == "LOI" else (0.807, 0.827)
    pub = pd.read_csv(PUB[task], encoding="utf-8-sig")
    m = preds[["Record_ID", "outer_fold"]].merge(pub[["Record_ID", "outer_fold"]], on="Record_ID", suffixes=("", "_pub"))
    same_folds = bool(len(m) == len(pub) and (m.outer_fold == m.outer_fold_pub).all())
    mp = preds.merge(pub[["Record_ID", "y_pred"]], on="Record_ID", suffixes=("", "_pub"))
    max_pred_diff = float(np.abs(mp.y_pred.astype(float) - mp.y_pred_pub.astype(float)).max())
    out = dict(task=task, metric=col, per_fold=v.tolist(), mean=float(v.mean()), sd=float(v.std(ddof=1)),
               gate=[lo, hi], passed=bool(lo <= v.mean() <= hi), folds_equal_published=same_folds,
               max_abs_pred_diff_vs_published=max_pred_diff,
               selected_models=folds["selected_model"].tolist(), minutes=round((time.time() - t) / 60, 2))
    if task == "LOI":
        out["modified_R2_mean"] = float(folds["outer_modified_R2"].mean())
    else:
        out["modified_Macro_F1_mean"] = float(folds["outer_modified_Macro_F1"].mean())
    (OUT / f"{task}_gate{PUB_SUFFIX}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def gate_pub(task, tmp):
    """Diagnostic gate: same evaluate_task_nested, but its OUTER folds (the one iter_cv_folds call with the full
    599 rows and random_state=42) are replaced by the published per-row folds. Inner CV, conformal, final refit untouched."""
    from common.split_strategies import Fold
    pub = pd.read_csv(PUB[task], encoding="utf-8-sig")
    fold_of = dict(zip(pub.Record_ID.astype(str), pub.outer_fold.astype(int)))
    rid = np.load(OUT / f"{task}_meta.npz")["record_id"]  # evaluate_task_nested row order (all 599 valid)
    f = np.array([fold_of[r] for r in rid])
    orig = se.iter_cv_folds

    def patched(X, y, *, groups, n_splits, random_state, classification):
        if len(y) == len(rid) and random_state == 42:
            for k in sorted(set(f)):
                yield Fold(np.flatnonzero(f != k), np.flatnonzero(f == k), int(k))
        else:
            yield from orig(X, y, groups=groups, n_splits=n_splits, random_state=random_state, classification=classification)
    se.iter_cv_folds = patched
    global PUB_SUFFIX
    PUB_SUFFIX = "_pubfolds"
    gate(task, tmp)


PUB_SUFFIX = ""


def r0_fit_predict(task, Xtr, ytr, gtr, Xte):
    """Their pipeline with the r0 model; returns (fitted pipeline, test prediction, threshold or None)."""
    cfg = se.TASK_CONFIGS[task]
    pipe = se.build_pipeline(task_type=cfg.task_type, model_name=R0_MODEL[task], requested_k=cfg.current_k, seed=SEED)
    pipe.fit(Xtr, ytr)
    if cfg.task_type == "regression":
        return pipe, np.asarray(pipe.predict(Xte), float), None
    # their UL-94 rule (evaluate_task_nested): inner grouped OOF probabilities on the training part -> Platt -> threshold
    raw = np.asarray(pipe.predict_proba(Xte)[:, 1], float)
    inner = se._inner_splits(Xtr, ytr, gtr, n_splits=5, seed=SEED, classification=True)
    oof = se._probability_oof(se.build_pipeline(task_type="classification", model_name=R0_MODEL[task],
                                                requested_k=cfg.current_k, seed=SEED), Xtr, ytr, inner)
    cal = se._fit_platt_calibrator(ytr, oof)
    thr, _, _ = se.optimise_threshold(ytr, se._apply_platt(cal, oof))
    prob = se._apply_platt(cal, raw)
    return pipe, prob, float(thr)


def front(pipe, X):
    return np.asarray(pipe[:5].transform(X), dtype=np.float64)  # nonempty, imputer, deduplicate, variance, selector


def export(task):
    bundle = se.prepare_scientific_bundle(DATA)
    cfg, X, y, df, ref, mol, load = task_frame(bundle, task)
    rid = df["Record_ID"].astype(str).to_numpy()
    S = json.loads(SPLITS.read_text())[task]
    assert S["record_id"] == rid.tolist(), "row order differs from the splits file"
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    for name, sets in S["splits"].items():
        groups = mol if name.startswith("P1_mol") or name.startswith("P1_pub_mol") else ref
        for f, te in enumerate(sets):
            te = np.asarray(te)
            tr = np.setdiff1d(np.arange(len(y)), te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr = y.iloc[tr].reset_index(drop=True)
            gtr = groups.iloc[tr].reset_index(drop=True)
            pipe, pred, thr = r0_fit_predict(task, Xtr, ytr, gtr, Xte)
            np.savez_compressed(d / f"{name}__f{f}.npz", Xtr=front(pipe, Xtr), Xte=front(pipe, Xte), tr=tr, te=te,
                                ytr=ytr.to_numpy(float), yte=y.iloc[te].to_numpy(float), r0=pred,
                                r0_thr=np.nan if thr is None else thr)
        print(f"{task} {name}: {len(sets)} folds, {time.time() - t:.0f}s", flush=True)
    if task == "LOI":
        p3(bundle, X, y)


def p3(train_bundle, X, y):
    """P3: their 10_ExternalValidation input path (prepared CSV fallback), fit on all 599, positive-loading rows scored."""
    sys.path.insert(0, str(CLONE / "10_ExternalValidation"))
    import run_external_validation as ev
    train = ev.read_csv_auto(DATA)
    external, kind, path = ev.resolve_external_source(ev.DEFAULT_EXTERNAL, train)
    ext_v4 = ev._coerce_prepared_external_to_train_schema(train, external)
    meta_ = ev._normalise_prepared_external_metadata(external)
    merged = pd.concat([train.reset_index(drop=True), ext_v4.reset_index(drop=True)], ignore_index=True, sort=False)
    tmp = OUT / "_p3_merged_train_external.csv"
    merged.to_csv(tmp, index=False, encoding="utf-8-sig")
    mb = se.prepare_scientific_bundle(tmp)
    cfg = se.TASK_CONFIGS["LOI"]
    M = se._filter_task_matrix(mb, "LOI", cfg.current_view, use_bde=False, screening_mode="formulation", feature_scope="all")
    M = se._apply_baseline_inclusive_mask(M, mb.df, mb.base.colmap, "LOI", "baseline_inclusive")
    X_ext = M.iloc[len(train):].reset_index(drop=True)
    y_ext = ev.external_truth(meta_, "LOI")
    elig = (ev.numeric(meta_["Standardized_total_FR_wt_pct"]).fillna(0).gt(0) & y_ext.notna()).to_numpy()
    pipe, _, _ = r0_fit_predict("LOI", X, y, None, X.iloc[:1])
    np.savez_compressed(OUT / "LOI" / "P3__f0.npz", Xtr=front(pipe, X), Xte=front(pipe, X_ext), ytr=y.to_numpy(float),
                        yte=y_ext.to_numpy(float), elig=elig, r0=np.asarray(pipe.predict(X_ext), float))
    print(f"P3: external source {kind} {path.name}, rows {len(X_ext)}, eligible n = {int(elig.sum())}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mode, task = sys.argv[1], sys.argv[2]
    if mode == "meta":
        meta(task)
    elif mode == "gate":
        gate(task, sys.argv[3])
    elif mode == "gate_pub":
        gate_pub(task, sys.argv[3])
    elif mode == "export":
        export(task)
