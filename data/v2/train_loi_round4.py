#!/usr/bin/env python3
"""LOI round 4 -- follows experiments/configs/loi_round4.json exactly (c1, c2, c3; no extra candidates, no tuning).

Data loading / features / RF config / protocols A, B, C / seeds / CIs are copied verbatim from
train_loi_snapshot_update.py (record exp087 = "exp085cfg" on 460 rows / 83 papers). Step 1 reproduces exp087
(B 0.3109, CI [0.1459, 0.4666], tolerance 0.002) or exits blocked.
New: F6 (base foam), F7 (class-prior el_* fill), F8 (synergy). Parsing notes for F6 iso_index_num (declared rule:
ratio, values > 10 divided by 100, NaN if missing): the first number in the text is taken; "NCO:OH a:b" gives a/b;
weight-ratio strings ("by weight", "g/g") are not an index and give NaN.
F7 details: priors = per FR class (the row's frK_class), median element mass-fraction (%) over all FR mentions in
the dataset whose names resolve in fr_element_table.csv or accepted fr_structures.csv (no LOI used); a class with no
resolved mention falls back to the global median over all resolved mentions. Only el_* that are NaN because of an
unresolved FR with a parsable loading are filled; rows whose loading is unparsed stay NaN. el_imputed_share = share
of the row's FR wt% whose elements came from priors.
Reported-not-selectable "baseline-aware" variant: c2 + the paper's neat-control LOI (mean LOI of benchmark rows with
n_frs == 0 in the same ref_id; NaN if none). A different task; log only.

Run: OMP_NUM_THREADS=1 python3 data/v2/train_loi_round4.py | tee experiments/runs/train_loi_round4.log
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



# =====================================================================================================
# Round 4 additions
# =====================================================================================================
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PAPER = "ref_id"
data0 = {"rf": prep(df2[bm], FEAT2)}
y, g, dom = data0["rf"][1], data0["rf"][2], data0["rf"][3]
assert len(y) == 460 and len(np.unique(g)) == 83


def hgb():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=5,
                                         random_state=0)


class Blend(BaseEstimator, RegressorMixin):
    """Equal-weight average of the RF and HGB (fit per fold)."""
    def fit(self, X, y_):
        self.m_ = [rf().fit(X, y_), hgb().fit(X, y_)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.m_], axis=0)


# ---- step 1: reproduce exp087 ---------------------------------------------------------------------------
t = time.time()
B87, oof87 = ep.repeated_group_kfold(rf(), data0["rf"][0], y, g)
lo87, hi87 = ep.bootstrap_ci(y, oof87[0], g)
print(f"REPRO exp087: B {B87.mean():.4f} CI [{lo87:.4f}, {hi87:.4f}] (record B 0.3109, CI [0.1459, 0.4666]) in {time.time()-t:.0f}s")
if not (abs(B87.mean() - 0.3109) <= 0.002 and abs(lo87 - 0.1459) <= 0.002 and abs(hi87 - 0.4666) <= 0.002):
    print("BLOCKED: exp087 not reproduced within 0.002; stopping.")
    sys.exit(2)
print("Reproduction within 0.002: True")

# ---- F6 base foam ------------------------------------------------------------------------------------------
RE_BIO = re.compile(r"soy|rapeseed|castor|tung|palm|lignin|canola|bio|HSBP|sucrose-free vegetable", re.I)
RE_PES = re.compile(r"polyester|Stepanpol|PS-?2352|Isoexter|aromatic", re.I)
RE_PET = re.compile(r"polyether|PET-4110|LY-4110|Voranol|Rokopol|4110", re.I)


def ptype(s):
    s = str(s) if pd.notna(s) else ""
    if RE_BIO.search(s):
        return "bio_based"
    if RE_PES.search(s):
        return "polyester_aromatic"
    if RE_PET.search(s):
        return "polyether"
    return "unknown"


def iso_num(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s)
    if re.search(r"by weight|g/g", s, re.I):
        return np.nan
    m = re.search(r"NCO\s*[:/]\s*OH\D{0,3}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", s, re.I)
    if m:
        v = float(m.group(1)) / float(m.group(2))
    else:
        m = re.search(r"\d+(?:\.\d+)?", s)
        if not m:
            return np.nan
        v = float(m.group(0))
    return v / 100 if v > 10 else v


PT = ["polyether", "polyester_aromatic", "bio_based", "unknown"]
raw_ptype = raw["polyol"].map(ptype)
F6 = pd.DataFrame({f"pt_{k}": (raw_ptype == k).astype(float) for k in PT})
F6["iso_index_num"] = raw["iso_index"].map(iso_num)
txt = (raw["polyol"].fillna("") + " | " + raw["isocyanate"].fillna("") + " | " + raw["notes"].fillna(""))
F6["is_PIR"] = ((F6["iso_index_num"] >= 1.8) | txt.str.contains(r"PIR|isocyanurate", regex=True)
                | txt.str.contains(r"isocyanurate", case=False, regex=True)).astype(float)
F6COLS = list(F6.columns)

# ---- F7 class priors ---------------------------------------------------------------------------------------
mentions = []  # (class, {el: frac%})
for _, r in raw.iterrows():
    for k in ("fr1", "fr2", "fr3"):
        name = str(r.get(f"{k}_name", "") or "").strip()
        if name.lower() in ("", "nan", "none"):
            continue
        c = str(r.get(f"{k}_class", "") or "other").strip()
        e = lookup_table(name)
        if e is not None:
            mentions.append((c, {x: float(e[x]) for x in ELEMS}))
        elif name in SMAP:
            mentions.append((c, dict(SMAP[name]["frac"])))
PRI = pd.DataFrame([dict(cls=c, **f) for c, f in mentions])
prior = {c: {x: float(grp[x].median()) for x in ELEMS} for c, grp in PRI.groupby("cls")}
prior_n = PRI.groupby("cls").size().to_dict()
glob_prior = {x: float(PRI[x].median()) for x in ELEMS}
print(f"\nF7 class-prior table (median element mass fraction %, over {len(PRI)} resolved FR mentions; no LOI used):")
print(f"{'class':28}{'n':>5}" + "".join(f"{x:>9}" for x in ELEMS))
for c in sorted(prior):
    print(f"{c:28}{prior_n[c]:>5}" + "".join(f"{prior[c][x]:>9.2f}" for x in ELEMS))
print(f"{'GLOBAL (fallback)':28}{len(PRI):>5}" + "".join(f"{glob_prior[x]:>9.2f}" for x in ELEMS))


def fill_row(r):
    """row_features(use_pubchem=True) arithmetic, with class priors for still-unresolved FRs."""
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
    ok_load, tot, imp = True, 0.0, 0.0
    for name, c, L, u in frs:
        b, e = basis(u), lookup_table(name)
        fr = SMAP[name]["frac"] if (e is None and name in SMAP) else None
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
        tot += w
        if e is not None:
            f = {x: float(e[x]) for x in ELEMS}
        elif fr is not None:
            f = fr
        else:
            f = prior.get(c, glob_prior)
            imp += w
        for x in ELEMS:
            elem[x] += w * f[x] / 100
    out = {f"el_{x}": (v if ok_load else np.nan) for x, v in elem.items()}
    out["el_imputed_share"] = (imp / tot if tot > 0 else 0.0) if ok_load else np.nan
    return out


filled = pd.DataFrame([fill_row(r) for _, r in raw.iterrows()])
for c in EL_COLS:  # rows complete before F7 must be unchanged
    a, b_ = df1[c].to_numpy(float), filled[c].to_numpy(float)
    m = ~np.isnan(a)
    assert np.allclose(a[m], b_[m]), c
df3 = df2.copy()
for c in EL_COLS:
    df3[c] = filled[c].to_numpy()
df3["el_imputed_share"] = filled["el_imputed_share"].to_numpy()
# F8 synergy
df3["syn_PxN"] = df3.el_P * df3.el_N
df3["syn_P_over_N"] = df3.el_P / (df3.el_N + 0.1)
df3["syn_EG_x_P"] = df3["cls_expandable_graphite"] * df3.el_P
df3["syn_totfr_x_dens"] = df3.total_fr_wtpct * df3.density_kgm3 / 1000
F8COLS = ["syn_PxN", "syn_P_over_N", "syn_EG_x_P", "syn_totfr_x_dens"]
for c in F6COLS:
    df2[c] = F6[c].to_numpy()
    df3[c] = F6[c].to_numpy()

bb = df2[bm]
print("\n=== Coverage (benchmark 460 rows) ===")
print("polyol_type counts:", {k: int(F6.loc[bm, f"pt_{k}"].sum()) for k in PT})
print(f"iso_index_num non-null: {int(F6.loc[bm, 'iso_index_num'].notna().sum())}/460 "
      f"(raw iso_index non-empty: {int(raw.loc[bm, 'iso_index'].notna().sum())}); is_PIR=1: {int(F6.loc[bm, 'is_PIR'].sum())}")
n_before = int(df2[bm][EL_COLS].notna().all(axis=1).sum())
n_after = int(df3[bm][EL_COLS].notna().all(axis=1).sum())
print(f"el_* all non-NaN (before F7 / after F7): {n_before}/460 -> {n_after}/460; rows imputed: "
      f"{int((df3[bm].el_imputed_share > 0).sum())}; mean imputed share over those: "
      f"{df3[bm].loc[df3[bm].el_imputed_share > 0, 'el_imputed_share'].mean():.3f}")

# neat control (baseline-aware variant)
neat = df[bm & (df.n_frs == 0).to_numpy()].groupby(PAPER).loi_pct.mean()
df3["neat_loi"] = df3[PAPER].map(neat)
print(f"neat-control LOI available: {int(df3[bm].neat_loi.notna().sum())}/460 rows, "
      f"{df3[bm][df3[bm].neat_loi.notna()][PAPER].nunique()}/83 papers")

FEAT_C1 = FEAT2 + F6COLS
FEAT_C2 = FEAT2 + F6COLS + ["el_imputed_share"] + F8COLS
FEAT_BA = FEAT_C2 + ["neat_loi"]
FEATS = {"c1": (df2, FEAT_C1), "c2": (df3, FEAT_C2), "c3": (df3, FEAT_C2), "ba": (df3, FEAT_BA)}
MODEL = {"c1": rf, "c2": rf, "c3": Blend, "ba": rf}
DATA = {k: prep(v[0][bm], v[1]) for k, v in FEATS.items()}
for k in DATA:
    assert (DATA[k][1] == y).all() and (DATA[k][2] == g).all()
print(f"features: exp087 {len(FEAT2)}, c1 {len(FEAT_C1)}, c2/c3 {len(FEAT_C2)}, baseline-aware {len(FEAT_BA)}")

res, runs = {}, []
for cid in ("c1", "c2", "c3", "ba"):
    X = DATA[cid][0]
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, MODEL[cid]().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    Bk, oof = ep.repeated_group_kfold(MODEL[cid](), X, y, g)
    for i, v in enumerate(Bk):
        runs.append(dict(protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    plo, phi = pooled_ci(y, oof, g)
    if cid == "ba":
        Ck = np.array([np.nan])
    else:
        Ck, _ = ep.repeated_group_kfold(MODEL[cid](), X, y, dom)
    wins = int((Bk > B87).sum())
    res[cid] = dict(A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5), B_mean=Bk.mean(),
                    B_std=Bk.std(), ci_lo=lo, ci_hi=hi, p_lo=plo, p_hi=phi, C_mean=Ck.mean(), wins=wins,
                    delta=float((Bk - B87).mean()))
    print(f"  {cid} done in {time.time()-t:.0f}s")

NAME = {"c1": "c1 incumbent+F6", "c2": "c2 +F6+F7+F8", "c3": "c3 RF/HGB avg on c2", "ba": "baseline-aware (c2+neat)"}
print("\n=== Summary (460 rows / 83 papers) ===")
print(f"{'cand':26}{'A mean [p2.5, p97.5]':24}{'B mean ± std':15}{'B CI rep0':18}{'B CI pooled':18}{'C':7}{'wins':6}{'dB'}")
rows = [("exp087 (record)", None)] + [(NAME[c], c) for c in ("c1", "c2", "c3")]
print(f"{'exp087 (record)':26}{'0.658 [0.476, 0.788]':24}{B87.mean():.3f} ± {B87.std():.3f}".ljust(65)
      + f"[{lo87:.3f}, {hi87:.3f}]".ljust(18) + f"[0.109, 0.442]".ljust(18) + "0.345")
for cid in ("c1", "c2", "c3"):
    r = res[cid]
    print(f"{NAME[cid]:26}" + f"{r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(24)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(15) + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(18)
          + f"[{r['p_lo']:.3f}, {r['p_hi']:.3f}]".ljust(18) + f"{r['C_mean']:.3f}  ".ljust(7)
          + f"{r['wins']}/10".ljust(6) + f"{r['delta']:+.4f}")

best = max(("c1", "c2", "c3"), key=lambda c: res[c]["B_mean"])
rb = res[best]
adopt = rb["wins"] >= 8 and rb["ci_lo"] >= 0.146
print(f"\nSELECTION (highest B mean): {best} B {rb['B_mean']:.4f}; paired wins {rb['wins']}/10 (need >= 8), "
      f"B ci_lo {rb['ci_lo']:.4f} (need >= 0.146) -> "
      + ("ADOPT candidate over exp087 pending critic" if adopt else "NOT adopted; exp087 stays the record"))
trig = []
for cid in ("c1", "c2", "c3"):
    r = res[cid]
    if r["ci_lo"] > SOTA:
        trig.append(f"TRIGGER: B ci_lo ({cid}) {r['ci_lo']:.3f} > {SOTA}")
    if r["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({cid}) {r['A_mean']:.3f} > {SOTA}")
print("Triggers:", trig if trig else "none")

r = res["ba"]
print("\n=== REPORTED-NOT-SELECTABLE: baseline-aware variant (c2 + paper's neat-control LOI; a DIFFERENT task, "
      "never compared with Sun/Fu) ===")
print(f"A mean {r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]; B {r['B_mean']:.3f} ± {r['B_std']:.3f}; "
      f"B CI rep0 [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]; pooled [{r['p_lo']:.3f}, {r['p_hi']:.3f}]; "
      f"paired B wins vs exp087 {r['wins']}/10, mean delta {r['delta']:+.4f} (no result JSON written)")

pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round4_all.csv", index=False)
SHORT = {"c1": "f6", "c2": "f6f7f8", "c3": "rfhgb_avg"}
DESC = {"c1": "RF (exp085 config) + F6 base foam (polyol_type one-hot, iso_index_num, is_PIR)",
        "c2": "RF (exp085 config) + F6 + F7 (el_* class-prior fill, el_imputed_share) + F8 synergy",
        "c3": "equal-weight avg of c2 RF and HistGradientBoosting(lr=0.05, max_iter=300, max_leaf_nodes=15, "
              "min_samples_leaf=5, random_state=0) on c2 features, per fold"}
for cid in ("c1", "c2", "c3"):
    r = res[cid]
    note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}]; pooled-over-repeats CI [{r['p_lo']:.3f},{r['p_hi']:.3f}]; "
            f"C {r['C_mean']:.3f}; {len(y)} rows/{len(np.unique(g))} papers; round 4 per loi_round4.json")
    out = dict(slug=f"loi_r4_{cid}_{SHORT[cid]}", target="LOI (%) RPUF literature",
               round_config="experiments/configs/loi_round4.json",
               dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (460 rows / 83 papers)",
               features=FEATS[cid][1], model="HistGradientBoostingRegressor+RandomForestRegressor" if cid == "c3" else "RandomForestRegressor",
               hyperparameters="n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native; " + DESC[cid],
               code="data/v2/train_loi_round4.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
               protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
               ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), pooled_ci_lo=round(float(r["p_lo"]), 4),
               pooled_ci_hi=round(float(r["p_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
               c_mean=round(float(r["C_mean"]), 4), struct_rows=41, struct_papers=14,
               paired_vs="exp087", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10", sota=SOTA,
               sota_source="sun_fu_2026", status="ok", blocked_reason="", note=note)
    (ROOT / f"experiments/results/loi_r4_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 3 result JSONs; runtime {time.time()-T0:.0f}s")
