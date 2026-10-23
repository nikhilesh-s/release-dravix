#!/bin/sh
# Re-run every train/validate script behind experiments/leaderboard.csv (exp001-exp009) and save
# each one's stdout+stderr to experiments/runs/<script_stem>.log. Paths resolve from this file's
# location, so it works from any cwd. None of these scripts writes CSVs or models; they only print.
#
# PY overrides the interpreter. It is word-split on purpose, so a multi-word command works:
#   PY="uv run -q --no-project --with-requirements requirements.txt python" sh data/v2/run_all_experiments.sh
# New experiments: append the script stem to the list below.
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS="$HERE/../../experiments/runs"
mkdir -p "$RUNS"

fail=0
for s in train_phrr_baseline train_phrr_v2_mcc_only train_tti_v1 train_tti_v2_logtarget \
         train_char_residue_v1 promote_check_phrr train_tti_v3_onset validate_phrr_crosslab \
         train_tti_v4_thermal; do
    echo "=== $s"
    # POSIX sh has no pipefail: carry the python exit code past tee through a file.
    { $PY "$HERE/$s.py" 2>&1; echo $? > "$RUNS/.$s.rc"; } | tee "$RUNS/$s.log"
    [ "$(cat "$RUNS/.$s.rc")" = 0 ] || { echo "FAILED: $s" >&2; fail=1; }
    rm -f "$RUNS/.$s.rc"
done
exit $fail
