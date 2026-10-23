#!/usr/bin/env python3
"""LOI snapshot re-score tool -- experiments/configs/loi_snapshot_update2.json (not a modelling round).

Builds an LOI snapshot in a throwaway copy of the tree and runs train_loi_round6.py there UNCHANGED except for
(a) the hard 460-row / 83-paper asserts, generalised to the snapshot's counts, and (b) the exp089/exp092
reproduction stops, which become reports (they cannot hold once rows are added). The round-6 summary then gives
the exp092 config (round-5 c2) and the exp093 config (c3, proxy-free) on that snapshot. Nothing in the repo is
written except the log under experiments/runs/.

  python3 data/v2/rescore_loi_snapshot.py            # self-test: current snapshot, must reproduce exp092 + exp093
  python3 data/v2/rescore_loi_snapshot.py --stage DIR [--stage DIR2 ...] [--build-only]
        # add each staging DIR's rows_w*.csv, fr_element_table_additions.csv, fr_structures_additions.csv and
        # EXCLUDE_REFS additions (config R2) -- only when the config's "when" condition holds.
        # A stage may also REPLACE entries (loi_snapshot_update3_chemfix): fr_element_table_replace.csv,
        # fr_element_table_unset.csv, fr_structures_replace.csv -- see replace_entries() -- and correct single
        # row cells via unit_corrections.csv (config bug-fix amendment) -- see correct_units().
        # LOKY_MAX_CPU_COUNT, if set, is passed to the scoring subprocess (core cap; results do not depend on it).

Self-test tolerance 0.002 (as the round scripts). Exit 0 ok, 2 self-test failed, 3 run failed.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRA_EXCLUDES = {"W032": "open-cell semi-rigid spray foam (loi_snapshot_update2 R2, N019 rule)"}
REC = {"exp092": (0.3738, 0.2007, 0.4700), "c3": (0.3594, 0.1829, 0.4737)}  # exp093 = round-6 c3


def copy_tree(dst: Path):
    skip = {".git", "raw", "runs", "results", "__pycache__", "pdfs"}
    with tempfile.TemporaryFile() as tf:
        with tarfile.open(fileobj=tf, mode="w") as t:
            t.add(ROOT, arcname=".", filter=lambda ti: None if Path(ti.name).name in skip or Path(ti.name).name.startswith(".venv") else ti)
        tf.seek(0)
        with tarfile.open(fileobj=tf) as t:
            t.extractall(dst, filter="data")
    (dst / "experiments/results").mkdir(parents=True, exist_ok=True)
    (dst / "experiments/runs").mkdir(parents=True, exist_ok=True)


def apply_stage(w: Path, stage: Path):
    parts = w / "data/v2/loi_rpuf_parts"
    rows = sorted(stage.glob("rows_w*.csv"))
    for f in rows:
        if (parts / f.name).exists():
            sys.exit(f"stage file {f.name} would overwrite an existing part")
        shutil.copy(f, parts / f.name)
    el = stage / "fr_element_table_additions.csv"
    if el.exists():
        with open(el) as a, open(w / "data/v2/fr_element_table.csv", "a", newline="") as b:
            next(a)
            b.write(a.read())
    st = stage / "fr_structures_additions.csv"
    if st.exists():
        target = w / "data/v2/fr_structures.csv"
        have = {r["fr_name"] for r in csv.DictReader(open(target))}
        add = [r for r in csv.DictReader(open(st)) if r["fr_name"] not in have]  # existing names kept verbatim
        with open(target, "a", newline="") as b:
            cw = csv.DictWriter(b, fieldnames=list(csv.DictReader(open(target)).fieldnames))
            cw.writerows(add)
    replace_entries(w, stage)
    correct_units(w, stage)
    print(f"stage {stage}: {len(rows)} row files, element additions {el.exists()}, structure additions {st.exists()}")


def correct_units(w: Path, stage: Path):
    """Bug-fix amendment of loi_snapshot_update3_chemfix (a61c218): the stage's unit_corrections.csv
    (row_id, column, old_value, new_value, ...) rewrites single cells of the working copy's row files.
    Each old_value must match exactly; every other cell must read back unchanged."""
    f = stage / "unit_corrections.csv"
    if not f.exists():
        return
    import pandas as pd
    fixes = list(csv.DictReader(open(f)))
    done = 0
    for p in sorted((w / "data/v2/loi_rpuf_parts").glob("rows_w*.csv")):
        before = pd.read_csv(p, dtype=str)
        hit = [x for x in fixes if x["row_id"] in set(before.row_id)]
        if not hit:
            continue
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            fields, rows = rd.fieldnames, list(rd)
        exp = before.copy()
        for x in hit:
            [r] = [r for r in rows if r["row_id"] == x["row_id"]]
            assert r[x["column"]] == x["old_value"], (x, r[x["column"]])
            r[x["column"]] = x["new_value"]
            exp.loc[exp.row_id == x["row_id"], x["column"]] = x["new_value"]
        _rewrite(p, rows, fields)
        pd.testing.assert_frame_equal(pd.read_csv(p, dtype=str), exp)
        done += len(hit)
    assert done == len(fixes), f"{len(fixes) - done} unit corrections did not find their row"
    print(f"stage {stage.name}: {done} unit cells corrected")


def _rewrite(path: Path, rows: list, fields: list):
    with open(path, "w", newline="") as b:
        cw = csv.DictWriter(b, fieldnames=fields)
        cw.writeheader()
        cw.writerows(rows)


def replace_entries(w: Path, stage: Path):
    """REPLACE (loi_snapshot_update3_chemfix): lookups are first-regex-match, so
    fr_element_table_replace.csv rows go in front of the table and win for the names they match;
    fr_element_table_unset.csv names (column fr_name) are guarded out of every existing entry (lookup -> None);
    fr_structures_replace.csv rows replace a non-accepted row of the same fr_name (accepted rows are never replaced)."""
    rep, uns, srep = (stage / f for f in ("fr_element_table_replace.csv", "fr_element_table_unset.csv",
                                         "fr_structures_replace.csv"))
    if rep.exists() or uns.exists():
        target = w / "data/v2/fr_element_table.csv"
        cur = list(csv.DictReader(open(target)))
        fields = list(cur[0])
        if uns.exists():
            names = [r["fr_name"] for r in csv.DictReader(open(uns))]
            guard = r"^(?!(?:" + "|".join(map(re.escape, names)) + r")$)[\s\S]*?(?:"
            for r in cur:
                r["match"] = guard + r["match"] + ")"
        new = list(csv.DictReader(open(rep))) if rep.exists() else []
        assert all(list(r) == fields for r in new), "replace file must use the element-table schema"
        _rewrite(target, new + cur, fields)
        print(f"stage {stage.name}: {len(new)} element entries placed first, {len(names) if uns.exists() else 0} names unset")
    if srep.exists():
        target = w / "data/v2/fr_structures.csv"
        cur = list(csv.DictReader(open(target)))
        fields = list(cur[0])
        idx = {r["fr_name"]: i for i, r in enumerate(cur)}
        new = list(csv.DictReader(open(srep)))
        assert all(list(r) == fields for r in new), "replace file must use the fr_structures schema"
        n_rep = 0
        for r in new:
            i = idx.get(r["fr_name"])
            if i is None:
                cur.append(r)
            elif cur[i]["status"] == "accepted":
                sys.exit(f"stage {stage.name} would replace the accepted structure of {r['fr_name']!r}")
            else:
                cur[i], n_rep = r, n_rep + 1
        _rewrite(target, cur, fields)
        print(f"stage {stage.name}: structures replaced {n_rep}, appended {len(new) - n_rep}")


def patch_build(w: Path):
    p = w / "data/v2/build_loi_rpuf.py"
    s = p.read_text()
    old = 'EXCLUDE_REFS = {"N019": "open-cell spray foam, outside the rigid-foam population"}'
    assert old in s
    s = s.replace(old, old[:-1] + "".join(f', "{k}": "{v}"' for k, v in EXTRA_EXCLUDES.items()) + "}")
    # basis() bug fix (loi_snapshot_update2 amendment, Oct 5): 'polyol+isocyanate' is a foam basis; equivalent ratios are unparsed
    b_old = '    if "wt% p" in u or u.startswith("p wt"):\n        return "P"\n'
    assert s.count(b_old) == 1
    s = s.replace(b_old, b_old + '    if re.search(r"polyol\\s*\\+\\s*iso", u):\n        return "foam"\n'
                  '    if "equivalent" in u:\n        return None\n')
    p.write_text(s)


def patch_round6(w: Path, n_rows: int, n_papers: int, script: str = "data/v2/train_loi_round6.py"):
    p = w / script
    s = p.read_text()
    a1 = "assert bm.sum() == 460 and df[bm].ref_id.nunique() == 83"
    a2 = "assert len(y) == 460 and len(np.unique(g)) == 83"
    assert s.count(a1) == 1 and s.count(a2) == 1
    s = s.replace(a1, f"assert bm.sum() == {n_rows} and df[bm].ref_id.nunique() == {n_papers}")
    s = s.replace(a2, f"assert len(y) == {n_rows} and len(np.unique(g)) == {n_papers}")
    stop = 'not reproduced within 0.002; stopping.")\n    sys.exit(2)'
    assert s.count(stop) == 2
    s = s.replace(stop, 'not reproduced within 0.002 (expected on an expanded snapshot; reported, not stopped).")')
    p.write_text(s)


def parse_summary(log: str) -> dict:
    out = {}
    for key, label in (("exp092", "exp092 (record)"), ("c3", "c3 exp092 - std (proxy-free)")):
        m = re.search(re.escape(label) + r"\s+([\d.-]+) \[[^\]]*\]\s+([\d.-]+) ± ([\d.]+)\s+\[([\d.-]+), ([\d.-]+)\]", log)
        if m:
            out[key] = dict(A=float(m.group(1)), B=float(m.group(2)), std=float(m.group(3)), lo=float(m.group(4)), hi=float(m.group(5)))
    return out


def write_results(w: Path, log: str, res: dict, n_rows: int, n_papers: int, stages: list):
    """Result JSONs (tracker schema) for the exp092 config and the exp093 config on the staged snapshot.
    4-decimal values: exp092 config from round 6's REPRO line, c3 from round 6's own c3 JSON."""
    import json
    m = re.search(r"REPRO exp092: B ([\d.-]+) CI \[([\d.-]+), ([\d.-]+)\]", log)
    c3 = json.loads((w / "experiments/results/loi_r6_c3_nostd_proxyfree.json").read_text())
    ds = (f"LOI snapshot = data/v2/loi_rpuf_literature.csv rebuilt with staged {', '.join(stages)} "
          f"(loi_pct notna, extraction != figure; {n_rows} rows / {n_papers} papers); built by rescore_loi_snapshot.py")
    common = dict(target="LOI (%) RPUF literature", round_config="experiments/configs/loi_snapshot_update2.json",
                  dataset=ds, model="RandomForestRegressor", code="data/v2/rescore_loi_snapshot.py (runs train_loi_round6.py)",
                  n_rows=n_rows, n_groups=n_papers, protocol="gkf_paper5x10", paired_vs="", paired_delta=0.0,
                  paired_wins="n/a (different snapshot)", sota=0.84, sota_source="sun_fu_2026", status="ok",
                  blocked_reason="")
    r2 = res["exp092"]
    out2 = dict(common, slug="loi_snap2b_exp092cfg", features=c3["features"] + ["std_* one-hot (round-5 c2)"],
                hyperparameters="n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native; exp092 config unchanged",
                rkf_mean=float(m.group(1)), rkf_std=r2["std"], ci_lo=float(m.group(2)), ci_hi=float(m.group(3)),
                gss_mean=r2["A"], note=f"exp092 config re-scored on the expanded snapshot; A {r2['A']:.3f}; rkf_std 3 d.p. from the "
                                         f"round-6 summary (log rescore_loi_expanded.log)")
    out3 = dict(common, slug="loi_snap2b_exp093cfg_proxyfree", features=c3["features"],
                hyperparameters=c3["hyperparameters"], rkf_mean=c3["rkf_mean"], rkf_std=c3["rkf_std"],
                ci_lo=c3["ci_lo"], ci_hi=c3["ci_hi"], pooled_ci_lo=c3["pooled_ci_lo"], pooled_ci_hi=c3["pooled_ci_hi"],
                gss_mean=c3["gss_mean"], c_mean=c3["c_mean"],
                note=f"exp093 config (round-6 c3, proxy-free) re-scored on the expanded snapshot; {c3['note']}")
    for o in (out2, out3):
        (ROOT / f"experiments/results/{o['slug']}.json").write_text(json.dumps(o, indent=2))
    print("wrote experiments/results/loi_snap2b_exp092cfg.json, loi_snap2b_exp093cfg_proxyfree.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", default=[], type=Path)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="build and count the snapshot; no scoring")
    ap.add_argument("--script", default=None, help="run this round script (built on the round-6 prefix) instead of round 6; its result JSONs are copied back")
    a = ap.parse_args()
    t0 = time.time()
    tag = "selftest" if not a.stage else "expanded"
    w = Path(tempfile.mkdtemp(prefix=f"dravix_rescore_{tag}_"))
    try:
        copy_tree(w)
        if a.stage:
            for st in a.stage:
                apply_stage(w, st.resolve())
            patch_build(w)
            subprocess.run([sys.executable, "data/v2/build_loi_rpuf.py"], cwd=w, check=True)
        import pandas as pd
        d = pd.read_csv(w / "data/v2/loi_rpuf_literature.csv")
        bm = d.loi_pct.notna() & (d.extraction != "figure")
        n_rows, n_papers = int(bm.sum()), int(d[bm].ref_id.nunique())
        print(f"snapshot ({tag}): {n_rows} benchmark rows / {n_papers} papers")
        if a.build_only:
            return 0
        script = a.script or "data/v2/train_loi_round6.py"
        if a.script:
            tag = Path(a.script).stem + (f"_snap{n_rows}" if a.stage else "_snap1")
        patch_round6(w, n_rows, n_papers, script)
        cap = {k: os.environ[k] for k in ("LOKY_MAX_CPU_COUNT",) if k in os.environ}  # core cap only; same numbers
        r = subprocess.run([sys.executable, script], cwd=w, capture_output=True, text=True,
                           env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", **cap})
        log = r.stdout + r.stderr
        (ROOT / f"experiments/runs/rescore_loi_{tag}.log").write_text(log)
        if r.returncode != 0:
            print(log[-3000:])
            print(f"FAILED: {script} exit {r.returncode}")
            return 3
        if a.script:
            n = 0
            for f in (w / "experiments/results").glob("*.json"):
                if not f.name.startswith("loi_r6_"):
                    shutil.copy(f, ROOT / "experiments/results" / f.name)
                    n += 1
            print("\n".join(l for l in log.splitlines() if l[:3] in ("exp", "r8a", "r8b", "r8c", "r8d", "s8 ", "REP", "SEL", "r8d", "Tri", "  r", "wro")))
            print(f"copied {n} result JSONs; log experiments/runs/rescore_loi_{tag}.log; runtime {time.time() - t0:.0f}s")
            return 0
        res = parse_summary(log)
        for k, v in res.items():
            print(f"{k:7s} A {v['A']:.3f}  B {v['B']:.4f}  CI [{v['lo']:.4f}, {v['hi']:.4f}]")
        print(f"runtime {time.time() - t0:.0f}s; log experiments/runs/rescore_loi_{tag}.log")
        if not a.stage:
            ok = all(k in res and all(abs(x - y) <= 0.002 for x, y in zip((res[k]["B"], res[k]["lo"], res[k]["hi"]), REC[k]))
                     for k in REC)
            print("SELF-TEST " + ("PASSED: exp092 and exp093 reproduced within 0.002" if ok else "FAILED"))
            return 0 if ok else 2
        trig = [k for k, v in res.items() if v["lo"] > 0.84 or v["A"] > 0.84]
        print("Triggers:", trig or "none")
        write_results(w, log, res, n_rows, n_papers, [st.name for st in a.stage])
        return 0
    finally:
        if a.keep:
            print(f"scratch copy kept at {w}")
        else:
            shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
