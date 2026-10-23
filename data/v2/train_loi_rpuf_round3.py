#!/usr/bin/env python3
"""LOI (RPUF literature) round 3 -- follows experiments/configs/loi_round3.json exactly.

Data, features, rf, protocols A/B/C, seeds and CI are copied from train_loi_rpuf_round1.py.
Structures: data/v2/fr_structures.csv (built by resolve_fr_structures.py). Loading/element logic: imports
helpers from build_loi_rpuf.py and re-runs its per-row loop with an extended FR lookup (element table first,
then PubChem formula of accepted structures).

Run: OMP_NUM_THREADS=1 python3 data/v2/train_loi_rpuf_round3.py | tee experiments/runs/train_loi_rpuf_round3.log
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
from build_loi_rpuf import ELEMS, basis, iso_parts  # noqa: E402

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
assert list(raw.row_id) == list(df.row_id)
base = [row_features(r, False) for _, r in raw.iterrows()]
newf = [row_features(r, True) for _, r in raw.iterrows()]
# sanity: table-only recomputation reproduces the committed CSV el_* columns
for c in EL_COLS:
    a = np.array([b[c] for b in base], float)
    assert np.allclose(a, df[c].to_numpy(float), equal_nan=True), c
print("sanity: table-only recompute == committed el_* columns on all 363 rows")
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
assert bm.sum() == 333


def cov(frame_el):
    b = frame_el[bm]
    return int(b[EL_COLS].notna().all(axis=1).sum())


print(f"element-feature coverage (benchmark rows with all el_* non-NaN): before {cov(df)}/333 "
      f"-> after {cov(df1)}/333")
b2 = df2[bm]
print(f"rows with >=1 resolved FR (struct features non-NaN): {int(b2['fr_fsp3'].notna().sum())}/333; "
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


sets = {"rf": (df, FEAT1), "c1": (df1, FEAT1), "c2": (df2, FEAT2)}
data = {k: prep(v[0][bm], v[1]) for k, v in sets.items()}
y, g, dom = data["rf"][1], data["rf"][2], data["rf"][3]
for k in data:
    assert (data[k][1] == y).all() and (data[k][2] == g).all()
print(f"benchmark: {len(y)} rows / {len(np.unique(g))} papers; features: rf/c1 {len(FEAT1)}, c2 {len(FEAT2)}")


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


runs = []


def run_all(cid):
    X = data[cid][0]
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, rf().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(set="bench", protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    Bk, oof = ep.repeated_group_kfold(rf(), X, y, g)
    for i, v in enumerate(Bk):
        runs.append(dict(set="bench", protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    Ck, _ = ep.repeated_group_kfold(rf(), X, y, dom)
    for i, v in enumerate(Ck):
        runs.append(dict(set="bench", protocol="C", cand=cid, rep=i, r2=v))
    print(f"  {cid} done in {time.time()-t:.0f}s")
    return dict(A=A, A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5),
                Bk=Bk, B_mean=Bk.mean(), B_std=Bk.std(), ci_lo=lo, ci_hi=hi, C_mean=Ck.mean())


res = {"rf": run_all("rf")}
r0 = res["rf"]
print(f"\nReproduction of exp064: B {r0['B_mean']:.4f} CI [{r0['ci_lo']:.4f}, {r0['ci_hi']:.4f}] A {r0['A_mean']:.4f} "
      f"(exp064: B 0.3512, CI [0.1296, 0.5196], A 0.6296)")
ok = (abs(r0["B_mean"] - 0.3512) <= 0.002 and abs(r0["ci_lo"] - 0.1296) <= 0.002 and abs(r0["ci_hi"] - 0.5196) <= 0.002)
print("Reproduction within 0.002:", ok)
if not ok:
    print("BLOCKED: exp064 not reproduced; stopping.")
    sys.exit(2)

NEW = ["c1", "c2"]
SHORT = {"c1": "rf_elcov", "c2": "rf_struct"}
for cid in NEW:
    res[cid] = run_all(cid)
    d = res[cid]["Bk"] - r0["Bk"]
    res[cid]["wins"] = int((d > 0).sum())
    res[cid]["delta"] = float(d.mean())

print("\n=== Summary (333 rows / 56 papers) ===")
print(f"{'cand':5} {'A mean [p2.5, p97.5]':24} {'B mean ± std':16} {'B CI (paper bootstrap)':22} {'C mean':8} {'wins/10':8} {'dB'}")
for cid, r in res.items():
    pw = f"{r['wins']}/10" if cid in NEW else "-"
    dl = f"{r['delta']:+.4f}" if cid in NEW else "-"
    print(f"{cid:5} {r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(30)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(17)
          + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(23) + f"{r['C_mean']:.3f}".ljust(9) + pw.ljust(8) + dl)

best = max(NEW, key=lambda c: res[c]["B_mean"])
b = res[best]
adopt_wins, adopt_ci = b["wins"] >= 8, b["ci_lo"] >= 0.130
adopt = adopt_wins and adopt_ci
print(f"\nSelected (highest protocol-B mean among c1,c2): {best} (B {b['B_mean']:.4f})")
print(f"Rule: wins>=8/10: {adopt_wins} ({b['wins']}/10); B ci_lo>=0.130: {adopt_ci} ({b['ci_lo']:.4f})")
print(f"Verdict: {'ADOPT ' + best + ' over exp064' if adopt else 'NOT ADOPTED; exp064 remains incumbent'}")
trig = []
for c in NEW:
    if res[c]["ci_lo"] > SOTA:
        trig.append(f"TRIGGER: B ci_lo ({c}) {res[c]['ci_lo']:.3f} vs {SOTA}")
    if res[c]["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({c}) {res[c]['A_mean']:.3f} vs {SOTA}")
print("Triggers:", trig if trig else "none")

pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round3_all.csv", index=False)
DESC = {"c1": "rf, round-1 features, el_* recomputed with PubChem formulas for FRs missing from fr_element_table.csv",
        "c2": "c1 + 6 RDKit FR-structure features (wt%-weighted over resolved FRs) + resolved_fr_wt_share"}
for cid in NEW:
    r = res[cid]
    note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}] (random 80/20 x50); C(dominant-class GKF) {r['C_mean']:.3f}; "
            f"B primary. paired vs exp064: {r['wins']}/10 repeats, mean dB {r['delta']:+.4f}. "
            f"{'selected' if cid == best else 'not selected'}; {'adopted' if (cid == best and adopt) else 'not adopted'}.")
    assert len(note) <= 300, len(note)
    out = dict(slug=f"loi_rpuf_r3_{cid}_{SHORT[cid]}", target="LOI (%) RPUF literature",
               round_config="experiments/configs/loi_round3.json",
               dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (333 rows / 56 papers)",
               features=FEAT1 if cid == "c1" else FEAT2, model="RandomForestRegressor",
               hyperparameters="n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native; " + DESC[cid],
               code="data/v2/train_loi_rpuf_round3.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
               protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
               ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
               paired_vs="exp064", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10",
               sota=SOTA, sota_source="sun_fu_2026", status="ok", blocked_reason="", note=note)
    (ROOT / f"experiments/results/loi_rpuf_r3_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 2 result JSONs; runtime {time.time()-T0:.0f}s")
