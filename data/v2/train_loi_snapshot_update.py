#!/usr/bin/env python3
"""LOI snapshot update -- follows experiments/configs/loi_snapshot_update.json exactly (no model/feature changes).

Re-scores exp085 (round-3 c2: RF + round-1 features + PubChem-coverage el_* + RDKit FR-structure features) and
exp064 (round-1 RF, round-1 features) on the expanded benchmark (460 rows / 83 papers). Data prep, features,
rf, protocols A/B/C, seeds and CI are copied from train_loi_rpuf_round3.py / train_loi_rpuf_round1.py; only the
row-count asserts (333/56 -> 460/83) and the raw-row alignment (EXCLUDE_REFS as in build_loi_rpuf.main) differ.
Sanity first: the old 333-row subset (ref_id starting with "M") must reproduce exp085 (B 0.3627, CI [0.1667, 0.5175]).

Run: OMP_NUM_THREADS=1 python3 data/v2/train_loi_snapshot_update.py | tee experiments/runs/train_loi_snapshot_update.log
"""
import glob, json, re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "v2"))
import eval_protocol as ep  # noqa: E402
from build_loi_rpuf import ELEMS, EXCLUDE_REFS, basis, iso_parts  # noqa: E402

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
T0 = time.time()
SOTA = 0.84
DATA = ROOT / "data/v2/loi_rpuf_literature.csv"
CLS_PREFIX = "cls_"
EL_COLS = ["el_P", "el_N", "el_Si", "el_B", "el_halogen", "el_metal"]
BOOLS = ["has_EG", "has_APP", "reactive"]
STRUCT_COLS = ["fr_arom_frac", "fr_fsp3", "fr_PO_per_kDa", "fr_PN_per_kDa", "fr_mollogp", "fr_tpsa_per_kDa"]

df = pd.read_csv(DATA)
cls_cols = [c for c in df.columns if c.startswith(CLS_PREFIX)]
FEAT1 = ["total_fr_wtpct"] + cls_cols + EL_COLS + ["density_kgm3", "n_frs"] + BOOLS
for b in BOOLS:
    df[b] = df[b].astype(str).str.lower().isin(["true", "1"]).astype(float)

# ---- structures ------------------------------------------------------------
st = pd.read_csv(ROOT / "data/v2/fr_structures.csv", dtype=str).fillna("")
acc = st[st.status == "accepted"]
print(f"fr_structures.csv: {len(st)} distinct FR names -> accepted {len(acc)}, "
      f"rejected {(st.status == 'rejected').sum()}, not_found {(st.status == 'not_found').sum()}")
print("\n=== ACCEPTED structures (human audit list): fr_name | query | CID | formula | SMILES ===")
for _, r in acc.iterrows():
    print(f"  {r.fr_name} | {r.query_used} | {r.cid} | {r.formula} | {r.smiles}")
print()

NONMET = set("H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe Ge".split())
ptable = Chem.GetPeriodicTable()


def frac_from_formula(formula):
    cnt = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        cnt[el] = cnt.get(el, 0) + (int(n) if n else 1)
    mass = {e: n * ptable.GetAtomicWeight(e) for e, n in cnt.items()}
    M = sum(mass.values())
    f = dict(P=mass.get("P", 0), N=mass.get("N", 0), Si=mass.get("Si", 0), B=mass.get("B", 0),
             halogen=sum(mass.get(x, 0) for x in ("F", "Cl", "Br", "I")),
             metal=sum(m for e, m in mass.items() if e not in NONMET))
    return M, {k: 100 * v / M for k, v in f.items()}


def struct_feats(smi):
    m = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(m)
    heavy = m.GetNumHeavyAtoms()
    return dict(fr_arom_frac=sum(a.GetIsAromatic() for a in m.GetAtoms()) / heavy,
                fr_fsp3=rdMolDescriptors.CalcFractionCSP3(m),
                fr_PO_per_kDa=len(m.GetSubstructMatches(Chem.MolFromSmarts("[#15]=[#8]"))) / mw * 1000,
                fr_PN_per_kDa=len(m.GetSubstructMatches(Chem.MolFromSmarts("[#15]~[#7]"))) / mw * 1000,
                fr_mollogp=Crippen.MolLogP(m), fr_tpsa_per_kDa=rdMolDescriptors.CalcTPSA(m) / mw * 1000)


SMAP = {r.fr_name: dict(frac=frac_from_formula(r.formula)[1], sf=struct_feats(r.smiles)) for _, r in acc.iterrows()}

# ---- element table lookup exactly as build_loi_rpuf.py -----------------------
el_tab = pd.read_csv(ROOT / "data/v2/fr_element_table.csv")


def lookup_table(name):
    for _, e in el_tab.iterrows():
        if re.search(e["match"], name, flags=re.I):
            return e
    return None


def row_features(r, use_pubchem):
    """Same arithmetic as build_loi_rpuf.main(); optionally falls back to PubChem formulas."""
    frs = []
    for k in ("fr1", "fr2", "fr3"):
        name = str(r.get(f"{k}_name", "") or "").strip()
        if name.lower() in ("", "nan", "none"):
            continue
        frs.append((name, str(r.get(f"{k}_class", "") or "other").strip(),
                    pd.to_numeric(r.get(f"{k}_loading"), errors="coerce"), r.get(f"{k}_unit")))
    iso, _ = iso_parts(r.get("isocyanate"))
    polyol_fr_sum = sum(L for _, _, L, u in frs if basis(u) == "polyol" and pd.notna(L))
    elem = dict.fromkeys(ELEMS, 0.0)
    ok_load, ok_elem = True, True
    wts = []  # (name, wt% of foam)
    for name, c, L, u in frs:
        b, e = basis(u), lookup_table(name)
        fr = None
        if e is None and use_pubchem and name in SMAP:
            fr = SMAP[name]["frac"]
        if pd.isna(L) or b is None:
            ok_load = False
            continue
        if b == "foam":
            w = L
        elif b == "polyol":
            w = L / (100 + iso + 5 + polyol_fr_sum) * 100
        else:
            pf = float(e["P"]) if e is not None else (fr["P"] if fr else 0.0)
            if not pf:
                ok_load = False
                continue
            w = L / pf * 100
        wts.append((name, w))
        if e is not None:
            for x in ELEMS:
                elem[x] += w * float(e[x]) / 100
        elif fr is not None:
            for x in ELEMS:
                elem[x] += w * fr[x] / 100
        else:
            ok_elem = False
    out = {f"el_{x}": (v if ok_load and ok_elem else np.nan) for x, v in elem.items()}
    out["_ok"] = ok_load and ok_elem
    out["_wts"] = wts if ok_load else None
    return out


parts = sorted(glob.glob(str(ROOT / "data/v2/loi_rpuf_parts/rows_w*.csv")))
raw = pd.concat([pd.read_csv(p, dtype=str) for p in parts], ignore_index=True)
raw = raw[~raw["ref_id"].isin(EXCLUDE_REFS)].reset_index(drop=True)  # as build_loi_rpuf.main()
assert list(raw.row_id) == list(df.row_id)
base = [row_features(r, False) for _, r in raw.iterrows()]
newf = [row_features(r, True) for _, r in raw.iterrows()]
# sanity: table-only recomputation reproduces the committed CSV el_* columns
for c in EL_COLS:
    a = np.array([b[c] for b in base], float)
    assert np.allclose(a, df[c].to_numpy(float), equal_nan=True), c
print(f"sanity: table-only recompute == committed el_* columns on all {len(df)} rows")
df1 = df.copy()
for c in EL_COLS:
    df1[c] = [n[c] for n in newf]

# struct features (wt%-weighted over resolved FRs)
rows2 = []
for n in newf:
    rec = dict.fromkeys(STRUCT_COLS, np.nan)
    rec["resolved_fr_wt_share"] = np.nan
    if n["_wts"] is not None:
        tot = sum(w for _, w in n["_wts"])
        res = [(nm, w) for nm, w in n["_wts"] if nm in SMAP and w > 0]
        rw = sum(w for _, w in res)
        rec["resolved_fr_wt_share"] = rw / tot if tot > 0 else 0.0
        if rw > 0:
            for c in STRUCT_COLS:
                rec[c] = sum(w * SMAP[nm]["sf"][c] for nm, w in res) / rw
    rows2.append(rec)
S = pd.DataFrame(rows2)
df2 = pd.concat([df1.reset_index(drop=True), S], axis=1)
FEAT2 = FEAT1 + STRUCT_COLS + ["resolved_fr_wt_share"]

bm = (df.loi_pct.notna() & (df.extraction != "figure")).to_numpy()
assert bm.sum() == 460 and df[bm].ref_id.nunique() == 83, (bm.sum(), df[bm].ref_id.nunique())


def cov(frame_el):
    b = frame_el[bm]
    return int(b[EL_COLS].notna().all(axis=1).sum())


print(f"element-feature coverage (benchmark rows with all el_* non-NaN): before {cov(df)}/460 "
      f"-> after {cov(df1)}/460")
b2 = df2[bm]
print(f"rows with >=1 resolved FR (struct features non-NaN): {int(b2['fr_fsp3'].notna().sum())}/460; "
      f"mean resolved_fr_wt_share over rows with FRs and loading ok: "
      f"{b2.loc[(b2.n_frs > 0) & b2.resolved_fr_wt_share.notna(), 'resolved_fr_wt_share'].mean():.3f}")


def prep(d, feats):
    d = d.reset_index(drop=True)
    X = d[feats].astype(float).to_numpy()
    y = d["loi_pct"].to_numpy(float)
    g = d["ref_id"].to_numpy()
    cl = d[cls_cols].fillna(0.0)
    dom = np.where(cl.max(axis=1) > 0, cl.idxmax(axis=1).str[len(CLS_PREFIX):], "none")
    return X, y, g, dom



sets = {"rf": (df, FEAT1), "c2": (df2, FEAT2)}
CFG = {"rf": "exp064", "c2": "exp085"}


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def pooled_ci(y, oof, groups, n_boot=2000, seed=0):
    """Same paper-level bootstrap as ep.bootstrap_ci, statistic = mean R2 over all repeats' OOF predictions."""
    rng = np.random.default_rng(seed)
    mats = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in mats}
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[m] for m in rng.choice(mats, len(mats))])
        stats.append(np.mean([r2_score(y[rows], p[rows]) for p in oof]))
    return np.percentile(stats, [2.5, 97.5])


# ---- sanity: old 333-row subset (waves 1-2, ref_id starts with "M") -----------------------------
old = df.ref_id.str.startswith("M").to_numpy() & bm
print(f"old subset (ref_id 'M*'): {old.sum()} rows / {df[old].ref_id.nunique()} papers")
assert old.sum() == 333 and df[old].ref_id.nunique() == 56
Xo, yo, go, _ = prep(df2[old], FEAT2)
t = time.time()
Bo, oofo = ep.repeated_group_kfold(rf(), Xo, yo, go)
lo_o, hi_o = ep.bootstrap_ci(yo, oofo[0], go)
print(f"SANITY exp085 on old subset: B {Bo.mean():.4f} CI [{lo_o:.4f}, {hi_o:.4f}] "
      f"(exp085: B 0.3627, CI [0.1667, 0.5175]) in {time.time()-t:.0f}s")
sane = abs(Bo.mean() - 0.3627) <= 0.002 and abs(lo_o - 0.1667) <= 0.002 and abs(hi_o - 0.5175) <= 0.002
print("Sanity within 0.002:", sane)
if not sane:
    print("BLOCKED: exp085 not reproduced on the old 333-row subset (rebuilt features changed something); stopping.")
    sys.exit(2)

# ---- structure coverage ---------------------------------------------------------------------------
bs = df2[bm]
has = bs["fr_fsp3"].notna()
print(f"\nstructure features non-NaN: {int(has.sum())}/{len(bs)} benchmark rows, {bs[has].ref_id.nunique()}/{bs.ref_id.nunique()} papers "
      f"(old M subset: {int(df2[old]['fr_fsp3'].notna().sum())}/333 rows)")
print(f"el_* all non-NaN (c2 frame): {int(df2[bm][EL_COLS].notna().all(axis=1).sum())}/{len(bs)} rows")
n_struct_rows, n_struct_papers = int(has.sum()), int(bs[has].ref_id.nunique())

# ---- scoring on identical splits ------------------------------------------------------------------
data = {k: prep(v[0][bm], v[1]) for k, v in sets.items()}
y, g, dom = data["rf"][1], data["rf"][2], data["rf"][3]
for k in data:
    assert (data[k][1] == y).all() and (data[k][2] == g).all()
print(f"benchmark: {len(y)} rows / {len(np.unique(g))} papers; features: rf {len(FEAT1)}, c2 {len(FEAT2)}")

res, runs = {}, []
for cid in ("c2", "rf"):
    X = data[cid][0]
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, rf().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    Bk, oof = ep.repeated_group_kfold(rf(), X, y, g)
    for i, v in enumerate(Bk):
        runs.append(dict(protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    plo, phi = pooled_ci(y, oof, g)
    Ck, _ = ep.repeated_group_kfold(rf(), X, y, dom)
    res[cid] = dict(A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5), B_mean=Bk.mean(),
                    B_std=Bk.std(), ci_lo=lo, ci_hi=hi, p_lo=plo, p_hi=phi, C_mean=Ck.mean())
    print(f"  {cid} ({CFG[cid]} config) done in {time.time()-t:.0f}s")

print("\n=== Summary (460 rows / 83 papers) ===")
print(f"{'cfg':7} {'A mean [p2.5, p97.5]':22} {'B mean ± std':14} {'B CI rep0':18} {'B CI pooled':18} {'C mean'}")
for cid in ("c2", "rf"):
    r = res[cid]
    print(f"{CFG[cid]:7} {r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(30)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(15) + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(19)
          + f"[{r['p_lo']:.3f}, {r['p_hi']:.3f}]".ljust(19) + f"{r['C_mean']:.3f}")
trig = []
for cid, r in res.items():
    if r["ci_lo"] > SOTA:
        trig.append(f"TRIGGER: B ci_lo ({CFG[cid]}) {r['ci_lo']:.3f} vs {SOTA}")
    if r["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({CFG[cid]}) {r['A_mean']:.3f} vs {SOTA}")
print("Triggers:", trig if trig else "none")

pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_snapshot460_all.csv", index=False)
DESC = {"c2": "RF, round-1 features + PubChem-coverage el_* + 6 RDKit FR-structure features + resolved_fr_wt_share (exp085 config)",
        "rf": "RF, round-1 features (exp064 config)"}
OUT = {"c2": "loi_snapshot460_exp085cfg", "rf": "loi_snapshot460_exp064cfg"}
for cid, r in res.items():
    note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}]; pooled-over-repeats CI [{r['p_lo']:.3f},{r['p_hi']:.3f}]; "
            f"C {r['C_mean']:.3f}; {len(y)} rows/{len(np.unique(g))} papers; snapshot update per loi_snapshot_update.json")
    out = dict(slug=OUT[cid], target="LOI (%) RPUF literature", round_config="experiments/configs/loi_snapshot_update.json",
               dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (460 rows / 83 papers)",
               features=FEAT1 if cid == "rf" else FEAT2, model="RandomForestRegressor",
               hyperparameters="n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native; " + DESC[cid],
               code="data/v2/train_loi_snapshot_update.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
               protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
               ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), pooled_ci_lo=round(float(r["p_lo"]), 4),
               pooled_ci_hi=round(float(r["p_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
               c_mean=round(float(r["C_mean"]), 4), struct_rows=n_struct_rows if cid == "c2" else 0,
               struct_papers=n_struct_papers if cid == "c2" else 0,
               paired_vs="", paired_delta="", paired_wins="", sota=SOTA, sota_source="sun_fu_2026",
               status="ok", blocked_reason="", note=note)
    (ROOT / f"experiments/results/{OUT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 2 result JSONs; runtime {time.time()-T0:.0f}s")
