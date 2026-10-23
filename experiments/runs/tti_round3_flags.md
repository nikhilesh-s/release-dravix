# TTI round 3 + round-9 tti_arm: executor flags and readings

Executor: worker subagent (Opus 5.5), 2026-10-07. Configs: `experiments/configs/tti_round3.json`, `experiments/configs/round9_tabpfn_smearing.json` (tti_arm), `experiments/configs/adoption_gate_v2.json`. Script: `data/v2/train_tti_round3.py`. Nothing here was decided by looking at a score; every item below was settled before step 1 unless marked otherwise.

## Readings that needed no choice (informational)

1. **Interpreter.** round9 `software_pins` names Python 3.12.4 at `/Library/Frameworks/...`; its D3 amendment names the repo `.venv` (Python 3.13.13). The orchestrator instructed `.venv312` (Python 3.12.12). Used `.venv312`. Every package in `already_installed_must_stay_unchanged` matches its pin exactly (importlib.metadata; `experiments/runs/tti_round3_tabpfn_env.json`), tabpfn 9.1.0, and step 1 reproduced exp046 (0.3133 [0.0715, 0.4958]), which is the drift guard the config names.
2. **`pip check` was vacuous.** `.venv312` has no pip (`No module named pip`), so post_install_checks item 2 printed no line naming a pinned package but did not actually check anything. Replacement check (label-free, recorded here): every requirement declared by tabpfn, einops, lightgbm, narwhals, skrub, pydot, cloudpickle, mlx and mlx-metal (importlib.metadata.requires, markers evaluated, packaging specifiers) is installed and satisfied: 0 unsatisfied.
3. **TabPFN cache dir.** `TABPFN_MODEL_CACHE_DIR=~/Library/Caches/tabpfn`, the darwin default for this user. The config writes `~/Library/Caches/tabpfn` and calls it "the darwin default, made explicit"; that home directory does not exist on this machine.
4. **Weights.** `tabpfn-v2-regressor.ckpt` was already present when this worker's setup ran (another worker fetched it at 22:24-22:25 through the package's own locked downloader). Size 44,390,977 bytes and sha256 `2ab5a07d…0736` equal the config; no 2.5/2.6/3.x file in the cache. Determinism: two in-process fits, a clone fit and a fresh-process fit give bitwise-equal synthetic predictions. Fitted attributes n_estimators_ 8, devices_ (cpu), use_autocast_ False, forced_inference_dtype_ None (as expected).
5. **round9 step 0 vs this script.** The tti_arm "runs only after this config's step 0 has passed". `data/v2/train_round9_tabpfn.py --target SETUP` did not exist when this round ran, so S0.1-S0.5 were done for the TTI arm inside this script (`--tabpfn-setup`, then the fresh-process leg in step 0). The S0.2 synthetic draw "predict 50 synthetic rows" is unspecified; this script draws `rng.normal(size=(50, 5))` after y. These are TTI-arm-local files (`experiments/runs/tti_round3_tabpfn_env.json`, `tti_round3_tabpfn_setup_preds.npz`); `round9_env.txt` / `round9_setup_preds.npz` were not written.
6. **D4d output-names clause.** tti_round3's amendment fixes the family order but does not literally say "this config's TTI output names govern"; round9's amendment says "tti_round3's output names govern", and tti_round3 `outputs.oof_and_gate` already names the one shared npz. Single-valued, so treated as recorded.
7. **adoption_gate.py** was written by another worker. Used unchanged after (a) a whitespace-insensitive check that every `pseudo_code` line occurs in it in order (its line wrapping differs from the config) and (b) its own label-free self-test. Its `run_npz` does the 1e-9 consistency checks and adds `order`, `selectable`, `npz` keys to the gate JSON.
8. **Spurious numpy FP warnings.** numpy 2.2.6 with macOS Accelerate raises "divide by zero / overflow / invalid value encountered in matmul" on finite inputs. These are ignored by a message filter; T1-T3 pass to 1e-7, and every OOF array is checked for non-finite values (R2).
9. **train_tti_v5_physics.py diagnostics bug** (repeats=1 without log_target): not inherited. This script never calls `v5.run()`; all models are scored on raw seconds through `eval_protocol.repeated_group_kfold` or `rgkf()`, which has no log_target path.
10. **p1/p2 pooled CI.** round9's result_schema asks for pooled_ci_lo/hi but names constructions only for pHRR/CR/FIGRA and LOI. Used the pHRR one (bootstrap_ci of the repeat-mean OOF, seed 0). Report-only.

## Ambiguities: not chosen, piece skipped

11. **Gate on the non-canonical rows.** tti_round3 `adoption.non_canonical_rows` says "Everything above is also computed on the non-canonical row set". adoption_gate_v2 `anti_gaming` allows one gate run per target per round. Not run on S681. Standard numbers (mean, std, per-repeat, CI, paired delta/wins vs exp046 on S681, family-GKF, legacy, secondary metrics, top-k, SSE share, fit diagnostics) are in `experiments/runs/tti_round3_noncanonical.json`. If the orchestrator wants the gate_v2 quantities on S681 too, that needs a ruling (it never selects anything either way).

## Round outcome flags

See the "Run outcome" section appended below after the run.

## Run outcome (appended by the orchestrator after the critic, 2026-10-08)
- All seven family members scored on canonical S675 (675 rows / 76 materials). Gate v2 (K = 7): best p2 (Avg(exp046, TabPFN)) +0.027, 7/10 paired wins, G1 fails; no adoption. a1/a2 (censored AFT variants) diverge badly on the seconds scale (R2 far below 0).
- exp046 re-scored on S675 = exp154 0.3179 [0.0917, 0.5067] becomes the TTI number of record (critic SOUND WITH CAVEATS). TTI parked again.
- Follow-up owed: re-run the TTI part of experiments/configs/conformal_intervals.json on the 675 rows (that config says to when 675 becomes canonical).
