#!/bin/sh
# Reproduce every number of record in experiments/numbers_of_record.json from the committed data and
# check them: runs the 8 script calls behind the targets, logs to experiments/runs/repro_<stem>[_<target>].log, then
# calls scripts/check_numbers_of_record.py (tolerance 0.001 on means and CI bounds). Exit 0 = all match.
#
#   target        exp          script (args)
#   pHRR          exp158       rescore_fsri_composite_snapshot.py --target pHRR   (C1 snapshot S2; S0 first reproduces exp074 within 0.002, else exit 2)
#   TTI           exp154       train_tti_round3.py --record-only                  (exp046 pipeline on S675: data/QA asserts incl. the 6 flagged ids,
#                                                                                   STEP 1 reproduces exp046 on S681 within 0.002, STEP 2 = exp154; no TabPFN)
#   THR           exp045       train_thr_v1_replicates.py                         (needs data/v2/raw)
#   Char Residue  exp157       rescore_fsri_composite_snapshot.py --target CR     (C1 snapshot S2; S0 first reproduces exp117 within 0.002, else exit 2)
#   LOI supp.     exp100       train_loi_round7.py (given-control d1, snapshot 1); re-checks exp092 first
#   LOI           exp163/164   rescore_loi_snapshot.py --stage wave4_staging --stage wave5_staging --stage chemfix_staging --stage unitfix_staging
#                              --script data/v2/train_loi_rescore_r8.py (snapshot 3 + chemistry-table correction, 606 rows / 107 papers:
#                              exp113 config = record exp163, proxy-free = exp164; result slugs loi_snap606_*; the longest step)
#   FIGRA         exp119       train_blend_round1.py --target FIGRA (blend round 1 b1; reproduces exp073 first)
#   Pomazi recon  exp076-079   train_pomazi_round1b.py
#   (history, no longer run: exp074 train_phrr_promote_c2, exp046 train_tti_v5_physics, exp117 train_blend_round1 --target CR
#    and exp081, exp092 train_loi_round5, exp109 train_loi_round6, exp115/116 (snapshot 3 without chemfix), exp073
#    train_figra_round1b. exp074/exp117/exp046 are still re-checked inside the pHRR/CR/TTI calls above.)
#
# NOTHING IN THE REPO IS OVERWRITTEN. Several of these scripts write result JSONs into experiments/results/
# (the tracker would ingest them as new experiments), pHRR/CR rewrite data/v2/fsri_composite_layers.csv and
# experiments/runs/fsri_composite_snapshot_*, and LOI/Pomazi write experiments/runs/*_all.csv. So the
# scripts run unmodified inside a throwaway copy of the tree (they resolve every path from their own location);
# only the logs and the produced result JSONs (experiments/runs/repro_results/) are brought back.
#
# THR, pHRR and CR read the raw FSRI clone, so data/v2/raw/ (gitignored) must exist. In a fresh clone:
#   git clone https://github.com/ulfsri/fsri_materials_database data/v2/raw/fsri_materials_database
#   git -C data/v2/raw/fsri_materials_database checkout a432697
#
# PY overrides the interpreter (word-split on purpose), e.g.
#   PY="uv run -q --no-project --with-requirements requirements.txt python" sh data/v2/run_numbers_of_record.sh
# Needs scikit-learn 1.7.1 (requirements.txt): RandomForest output changes across versions. pHRR/CR also need Python 3.12
# with numpy 2.2.6, pandas 2.3.3, scipy 1.16.1 and the clone at a432697 (else they write BLOCKED and exit 2), e.g. .venv312.
# KEEP=1 keeps the scratch copy (path printed). Runtime 2026-10-08: 62 min wall with LOKY_MAX_CPU_COUNT=4 on a shared
# 10-core Mac (LOI rescore ~21 min, Pomazi ~28 min, the rest < 10 min); 48/48 matched.
# The older LOI round scripts (train_loi_rpuf_round1/2/3.py) assert 333 rows and no longer run on the current loi_rpuf_literature.csv; exp085 (333 rows / 56 papers), exp087 (460 rows) and exp089 (train_loi_round4.py, 460 rows) are history.
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUNS="$ROOT/experiments/runs"
OUT="$RUNS/repro_results"
FSRI="$ROOT/data/v2/raw/fsri_materials_database"
export OMP_NUM_THREADS=1

if [ ! -d "$FSRI/01_Data" ]; then
    echo "missing $FSRI/01_Data (THR needs the raw FSRI clone). Re-clone with:" >&2
    echo "  git clone https://github.com/ulfsri/fsri_materials_database $FSRI" >&2
    echo "  git -C $FSRI checkout a432697" >&2
    exit 2
fi

mkdir -p "$RUNS"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/dravix_repro.XXXXXX") || exit 2
if [ -n "$KEEP" ]; then echo "scratch copy kept at $WORK"; else trap 'rm -rf "$WORK"' EXIT; fi

# Throwaway copy of the tree: no .git, no raw (symlinked), empty results/ and runs/ so nothing stale can be mistaken for output.
(cd "$ROOT" && tar -cf - --exclude=./.git --exclude='./.venv*' --exclude=./data/v2/raw --exclude=./experiments/results \
    --exclude=./experiments/runs --exclude=__pycache__ .) | (cd "$WORK" && tar -xf -) || exit 2
mkdir -p "$WORK/experiments/results" "$WORK/experiments/runs"
ln -s "$ROOT/data/v2/raw" "$WORK/data/v2/raw"

fail=0
for spec in "rescore_fsri_composite_snapshot --target pHRR" \
            "train_tti_round3 --record-only" \
            "train_thr_v1_replicates" \
            "rescore_fsri_composite_snapshot --target CR" \
            "train_loi_round7" \
            "rescore_loi_snapshot --stage data/v2/loi_rpuf_parts/wave4_staging --stage data/v2/loi_rpuf_parts/wave5_staging --stage data/v2/loi_rpuf_parts/chemfix_staging --stage data/v2/loi_rpuf_parts/unitfix_staging --script data/v2/train_loi_rescore_r8.py" \
            "train_blend_round1 --target FIGRA" \
            "train_pomazi_round1b"; do
    set -- $spec
    s=$1; shift
    [ "$1" = --target ] && n="${s}_$2" || n=$s   # one log per --target call
    echo "=== $s $*"
    # POSIX sh has no pipefail: carry the python exit code past tee through a file.
    { (cd "$WORK" && $PY "data/v2/$s.py" "$@") 2>&1; echo $? > "$RUNS/.repro_$n.rc"; } | tee "$RUNS/repro_$n.log"
    [ "$(cat "$RUNS/.repro_$n.rc")" = 0 ] || { echo "FAILED: $s $*" >&2; fail=1; }
    rm -f "$RUNS/.repro_$n.rc"
done

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$WORK"/experiments/results/*.json "$OUT"/ 2>/dev/null

$PY "$ROOT/scripts/check_numbers_of_record.py" || fail=1
exit $fail
