# samedata_huang_dopo_ep_ml (+ _ul94): FLAGs and implementation notes

Worker: Opus 5.5 (Claude), 2026-10-08. Configs followed literally; nothing tuned. Items marked FLAG need a human look.

## FLAG 1: LOI published folds are not what the authors' code produces on this machine
- r0_full (their `evaluate_task_nested`, reference grouping, unmodified, authors' pinned env) builds its outer folds with
  `GroupKFold(5)` unshuffled. Here those folds hold the same sizes (120/120/120/120/119, 26 papers each) but only 24.7%
  of rows share a fold id with the published per-row `outer_fold` (LOI_outer_predictions_reference.csv); the partitions
  differ. The config assumed they are identical ("equals GroupKFold(5) unshuffled by Reference"). Most likely cause:
  unshuffled GroupKFold orders papers by size with an unstable argsort, so ties (many papers have 2-9 rows) break
  differently across platforms (their Windows run vs this macOS arm64 machine). The same holds in sklearn 1.7.1.
  Inner CV folds are built the same way, so their inner model choice also runs on slightly different inner folds.
- UL-94 is unaffected: its StratifiedGroupKFold(shuffle=True, random_state=42) folds equal the published ones exactly.
- Literal r0_full (their folds as generated here) gives 0.7012 ± 0.0393 -> inside [0.699, 0.719] -> PASS.
- Diagnostic, to honour "same folds": the same unmodified routine with only its outer split replaced by the published
  per-row folds (data/v2/export_dopo_folds.py gate_pub): 0.7050 ± 0.0816 (folds 0.708, 0.835, 0.704, 0.664, 0.615;
  inner choice XGB, GBDT x4; published 0.709 ± 0.084, XGB x3 / GBDT x2) -> also PASS
  (experiments/runs/samedata_dopo_folds/LOI_gate_pubfolds.json). Both readings pass, so the gate outcome does not
  depend on which one the owner adopts.
- All Dravix-side LOI P1_pub numbers (r0, c1-c3) use the published folds from the CSV, as the config says.

## FLAG 2: UL-94 c2u and c3u BLOCKED (TabPFN 500-feature limit)
- The authors' UL-94 input (morgan_r3, K = all) leaves 665-694 features per training fold after their front end
  (1749 raw). TabPFN V2 raised on the first fold: `TabPFNValidationError: Number of features 667 reaching the model is
  greater than the maximum number of features 500 ...` (experiments/runs/samedata_dopo_preds/UL94_V0_c2u_BLOCKED.json).
  Per samedata_common.json c2.on_error: c2u and c3u BLOCKED for this dataset, never retried with
  ignore_pretraining_limits or anything else. The UL-94 family is {c1u}, so c* = c1u by default.
- The V2 classifier checkpoint itself was fetched fine from HF Prior-Labs/TabPFN-v2-clf (sha256 cf8c519c...,
  29,009,539 bytes, recorded in samedata_env.txt); it simply cannot take this many features.

## Wording note
- The P1 sentence template says "vs their published Y (their single split)". The authors' Y is the mean over one
  5-fold assignment, not a single train/test split; the template is printed verbatim as the config requires.

## Implementation notes (no deviation intended; listed so a reviewer can check them)
- c1u logistic member: sklearn 1.7.1 / numpy 2.2.6 on macOS arm64 prints "divide by zero / overflow / invalid value
  encountered in matmul" RuntimeWarnings (known spurious Accelerate floating-point flags); every c1u probability is
  finite and rows sum to 1.
- LOI c2 (TabPFN, OMP/torch threads = 1 as bound by the config, 5 worker processes) runs on a machine shared with
  other sessions (load average ~65 on 10 cores); per-fold predictions are cached under
  experiments/runs/samedata_dopo_preds/LOI_c2/ so an interruption resumes without refitting.
- libomp: xgboost 3.2.0 and lightgbm 4.6.0 macOS wheels need libomp.dylib, which is not installed system-wide. Without it
  their code cannot import (pipeline_core imports xgboost at top level). Every step-0 run sets
  DYLD_FALLBACK_LIBRARY_PATH to the libomp that ships inside the pinned scikit-learn 1.6.1 wheel. Nothing was installed
  beyond the approved venv. Recorded in experiments/runs/samedata_env.txt.
- Input file: config says prepare_scientific_bundle('data/DOPO_EP_new.csv'); used as written. Their published FINAL run
  used DOPO_EP_new_with_BDE.csv (same 599 rows and 66 shared columns byte-identical, plus 31 BDE-prediction columns, all
  dropped by use_bde=False except names without 'bde'; not tried, since the gate passed with the configured file).
- r0 seeds: config gives random_state=42 for r0 (LOI GBDT) and _classification_model('ExtraTrees', 42) for UL-94. Their
  nested code uses 42 + outer fold id; r0 here uses 42 on every fold of every protocol, and the UL-94 threshold rule's
  inner StratifiedGroupKFold also uses seed 42 (groups = the split's groups: Reference, or molecule for P1_mol).
- c1-c3 get the post-front-end matrices (their imputer runs inside the front end, so no NaN reaches c1-c3).
- Splits JSON was written before any fit; the step-0 env freeze was appended to samedata_env.txt about one minute after
  the r0_full gate runs started (same packages, nothing installed in between). The Dravix-env freeze preceded every
  Dravix-env fit.
- UL-94 P1 uses StratifiedGroupKFold(shuffle=True, random_state=r) and P2 uses GroupKFold(shuffle=True, random_state=r),
  so they are different splits; UL-94 P2 CI uses the eval_protocol.bootstrap_ci resampling with macro-F1 as statistic.
- SD conventions: P1_pub over the 5 folds with ddof=1 (as the authors); over 10 repeats with ddof=0 (as eval_protocol).

## Runtime (wall, shared machine; load average 20-77 on 10 cores)
- Clone + venv 2 min; r0_full gates 3.3 min (LOI) / 1.3 min (UL-94), published-fold diagnostic 5.5 min; step-0 exports
  6.1 min (LOI, 111 fold sets incl. P3) / 9.6 min (UL-94, 105); c1 8.9 min; c1u 2.7 min; c2 383.6 min (TabPFN, 5 single-
  thread workers), plus ~15 min of c2 work discarded when it was restarted with per-fold caching; c2u blocked in 10 s.
  Total about 7.5 h (01:44-09:10 PDT).
- Process note: when restarting c2, `pkill -f spawn_main` was used to stop this run's 5 worker processes. That pattern
  could also have matched another session's multiprocessing workers; the other sessions' visible jobs (loky workers,
  train_samedata.py, train_loi_rescore_r8.py) were still running afterwards.
