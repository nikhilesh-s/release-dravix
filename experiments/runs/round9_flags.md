# Round 9 (round9_tabpfn_smearing + cr_round2) -- worker flags

Written by the round-9 worker (Oct 7 2026). None of these changed a model, a feature, a row, a fold or a gate constant.
Items marked FLAG need an orchestrator amendment or decision; items marked NOTE are environment adaptations recorded for the critic.

1. FLAG (interpreter record). The round9 amendment names the repo's `.venv` (Python 3.13.13). The orchestrator's task
   instruction says to use `.venv312` (Python 3.12.12) and calls that the stand-in "as recorded in round9's amendment", but
   the committed amendment does not mention `.venv312`. I followed the orchestrator and used `.venv312`. Its library versions
   match every pin exactly (torch 2.8.0, scikit-learn 1.7.1, numpy 2.2.6, scipy 1.16.1, pandas 2.3.3, the 15 must-stay
   packages, and tabpfn 9.1.0 + the 8 pinned no-deps packages; see round9_env.txt). cr_round2.json `environment` names
   Python 3.12.4 and says "if any version differs, stop BLOCKED". train_cr_round2.py checks the four library versions
   exactly and only prints the Python version (3.12.12), on the strength of the orchestrator's stand-in instruction. Please
   record the `.venv312` stand-in in both configs' amendments. exp074, exp117, exp119 and exp115 all reproduced within 0.002
   in this interpreter.
   UPDATE: commit 284ed43 added the `.venv312` erratum to round9_tabpfn_smearing.json. cr_round2.json `environment` still
   names 3.12.4 with no erratum.
2. NOTE (TABPFN_MODEL_CACHE_DIR). The config's value is `~/Library/Caches/tabpfn`, described as "the darwin
   default, made explicit". On this machine the darwin default is `~/Library/Caches/tabpfn`, and that is what I
   used. The checkpoint is the only `.ckpt` there: tabpfn-v2-regressor.ckpt, 44,390,977 bytes, sha256 2ab5a07d...0736 (match),
   fetched by the package's HuggingFace downloader. The download log has no direct-URL or S3 fallback line.
3. NOTE (S0.2 synthetic test rows). The config says to "predict 50 synthetic rows" but does not say which rows. I used the
   next draws of the same generator (`Xte = rng.normal(size=(50, 5))` after X and the noise). This affects only the
   label-free determinism check, which passed bitwise both in-process and in a fresh process.
4. NOTE (pip -> uv). `pip freeze` / `pip check` were run as `~/.local/bin/uv pip freeze|check --python .venv312/bin/python`,
   as instructed. Output: "All installed packages are compatible".
5. FLAG (LOI driver). Run exactly as written, `rescore_loi_snapshot.py` fails before any scoring: its `copy_tree` tars the whole repo
   root, which now holds `.venv/` and `.venv312/`, and Python 3.12's tarfile `filter="data"` refuses the absolute symlink
   `./.venv/bin/python` (`tarfile.AbsoluteLinkError`). I did not modify `rescore_loi_snapshot.py`. I ran the unmodified driver,
   with the config's arguments, from a copy of the repo tree without `.venv*`, `.git`, `data/v2/raw`, `experiments/runs`,
   `experiments/results` and `__pycache__` (session scratchpad `loi_tree/`). That gives the same files the driver would
   otherwise copy, minus the virtualenvs. The driver's outputs (result JSONs, its log, and the kept copy's
   `round9_loi_oof.npz`, sha256-checked) were copied into the repo, and the scratch copies were deleted. Side note: the
   background numbers-of-record repro (`run_numbers_of_record.sh`) copies `.venv*` into its scratch tree, so its LOI step
   (`rescore_loi_snapshot.py`) will most likely hit the same error.
   UPDATE: commit 284ed43 (22:45) patched `copy_tree` to skip `.venv*`. My LOI run had already started at about 22:30,
   from the venv-free copy with the pre-patch driver. Apart from the `.venv*` skip, both driver versions are the same code,
   so the result is unaffected.
6. NOTE (threads). Following the worker limit of 4 cores, every process ran with OMP_NUM_THREADS=1, torch.set_num_threads(1) and
   LOKY_MAX_CPU_COUNT=1. The last one makes `n_jobs=-1` forests run on one thread. The code still says `n_jobs=-1`, and
   scikit-learn forest output does not depend on n_jobs, because per-tree seeds are drawn before the parallel fit. The LOI
   script sets LOKY_MAX_CPU_COUNT itself because the driver strips the environment. At most 4 of my processes ran at once.
7. NOTE (TTI arm). Skipped here as instructed; the TTI worker runs p1/p2 inside tti_round3. Step 0 (the weights, round9_env.txt,
   determinism) passed at 22:25 PDT, before any Dravix label was touched.
