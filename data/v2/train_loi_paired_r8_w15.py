#!/usr/bin/env python3
"""LOI snapshot re-score of the current record configs (loi_snapshot_update2 amendment): exp113 = r8d blend on exp092 features, exp114 = r8d blend on proxy-free c3 features. Unchanged models.

Everything up to the round-6 fit section is train_loi_round6.py verbatim (data prep, features, exp089/exp092 checks).
Meant to run on snapshot 2 via: python3 data/v2/rescore_loi_snapshot.py --stage data/v2/loi_rpuf_parts/wave4_staging
--script data/v2/train_loi_round8.py (which generalises the 460/83 asserts and turns the snapshot-1 repro stops
into reports). All candidates use the exp092 (round-5 c2) feature matrix.
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



from sklearn.model_selection import train_test_split
PAPER = "ref_id"
data0 = {"rf": prep(df2[bm], FEAT2)}
y, g, dom = data0["rf"][1], data0["rf"][2], data0["rf"][3]
assert len(y) == 460 and len(np.unique(g)) == 83

# ---- round-4 F6 (verbatim) -- needed for the exp089 reproduction and the audit "before" column -----------
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


PT4 = ["polyether", "polyester_aromatic", "bio_based", "unknown"]
raw_ptype = raw["polyol"].map(ptype)
F6 = pd.DataFrame({f"pt_{k}": (raw_ptype == k).astype(float) for k in PT4})
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

# ---- step 1: reproduce exp089 (round-4 c2 = F6(r4) + F7 + F8) --------------------------------------------------
FEAT_R4C2 = FEAT2 + F6COLS + ["el_imputed_share"] + F8COLS
dR4 = df3.copy()
for c in F6COLS:
    dR4[c] = F6[c].to_numpy()
X89 = prep(dR4[bm], FEAT_R4C2)[0]
t = time.time()
B89, oof89 = ep.repeated_group_kfold(rf(), X89, y, g)
lo89, hi89 = ep.bootstrap_ci(y, oof89[0], g)
print(f"REPRO exp089: B {B89.mean():.4f} CI [{lo89:.4f}, {hi89:.4f}] (record B 0.3542, CI [0.1694, 0.4666]) in {time.time()-t:.0f}s")
if not (abs(B89.mean() - 0.3542) <= 0.002 and abs(lo89 - 0.1694) <= 0.002 and abs(hi89 - 0.4666) <= 0.002):
    print("BLOCKED: exp089 not reproduced within 0.002; stopping.")
    sys.exit(2)
print("Reproduction within 0.002: True")

# ---- F6_fixed ------------------------------------------------------------------------------------------------
RX_PES = re.compile(r"polyester|Stepanpol|PS-?2352|Isoexter|glycoly[sz]ed PET|PET oligo|terephthal", re.I)
RX_PET = re.compile(r"polyether|Rokopol|RF-?551|Voranol|Voracor|LY-?4110|PET-?4110|PTMEG|PPG|PEG|Laprol|oxypropyl|"
                    r"oligoetherol|4110", re.I)
RX_BIO = re.compile(r"soy|rapeseed|castor|tung|palm|lignin|canola|bio|HSBP|vegetable", re.I)
RX_PLA = re.compile(r"\bPLA")
RX_UNIT = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)\s*(?:g|pbw|parts?|php|phr|pph|wt\.?\s*%|%)(?![A-Za-z/])", re.I)


def comp_label(c):
    if RX_PES.search(c):
        return "polyester_aromatic"
    if RX_PET.search(c):
        return "polyether"
    if RX_BIO.search(c):
        return "bio_based"
    if RX_PLA.search(c):
        return "other_polyester"
    return "unknown"


def split_components(s):
    m = re.match(r"(.+?) / (.+?) wt% = (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?)$", s)
    if m:
        return [(m.group(1), float(m.group(3))), (m.group(2), float(m.group(4)))]
    parts, depth, cur, i = [], 0, "", 0
    while i < len(s):
        ch = s[i]
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and s[i:i + 3] == " + ":
            parts.append(cur)
            cur, i = "", i + 3
            continue
        cur += ch
        i += 1
    parts.append(cur)
    out = []
    for p in parts:
        u = RX_UNIT.search(p)
        if u:
            out.append((p, float(u.group(1)), "unit"))
            continue
        b = re.search(r"(?<![\w.:-])(\d+(?:\.\d+)?)$", re.sub(r"\([^)]*\)", "", p).strip())
        out.append((p, float(b.group(1)) if b else None, "bare" if b else None))
    n_amt = sum(o[1] is not None for o in out)
    n_unit = sum(o[2] == "unit" for o in out)
    if not (n_amt >= 2 or n_unit == 1):
        return [(o[0], None) for o in out]
    return [(o[0], o[1]) for o in out]


def ptype_fixed(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown", ""
    comps = split_components(str(s))
    stated = [(p, a) for p, a in comps if a is not None and a > 0]
    if any(a is not None for _, a in comps):
        if not stated:
            return "unknown", ""      # all stated amounts are 0
        chosen = max(stated, key=lambda pa: pa[1])[0]  # ties -> first listed
    else:
        chosen = comps[0][0]
    return comp_label(chosen), chosen.strip()


def iso_num_fixed(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s)
    m = re.search(r"\bR\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*:\s*1(?:\.0+)?(?![.\d])", s)
    if m:
        return float(m.group(1))
    return iso_num(s)


PT5 = ["polyether", "polyester_aromatic", "bio_based", "other_polyester", "unknown"]
fx = raw["polyol"].map(ptype_fixed)
fixed_lab = fx.map(lambda t: t[0])
fixed_chosen = fx.map(lambda t: t[1])
F6f = pd.DataFrame({f"pt_{k}": (fixed_lab == k).astype(float) for k in PT5})
F6f["iso_index_num"] = raw["iso_index"].map(iso_num_fixed)
txt_pir = txt.str.replace(r"isocyanurate-based\)? polyether polyol", "", case=False, regex=True)
F6f["is_PIR"] = ((F6f["iso_index_num"] >= 1.8) | txt_pir.str.contains(r"PIR", regex=True)
                 | txt_pir.str.contains(r"polyisocyanurate", case=False, regex=True)).astype(float)
F6COLS5 = list(F6f.columns)

# ---- audit (printed BEFORE any round-5 model is fit) --------------------------------------------------------------
print("\n=== AUDIT: polyol_type, every unique polyol string (n = rows in benchmark 460 / all rows) ===")
au = pd.DataFrame(dict(polyol=raw["polyol"].fillna("(NaN)"), r4=raw_ptype, fx=fixed_lab, chosen=fixed_chosen, bm=bm))
for s, grp in au.groupby("polyol", sort=False):
    ch = "" if grp.chosen.iloc[0] == s or not grp.chosen.iloc[0] else f"  [chosen comp: {grp.chosen.iloc[0][:60]}]"
    flag = "CHANGED " if grp.r4.iloc[0] != grp.fx.iloc[0] else "        "
    print(f"{flag}{int(grp.bm.sum()):>3}/{len(grp):<3} {grp.r4.iloc[0]:19} -> {grp.fx.iloc[0]:19} | {s[:110]}{ch}")
print(f"\nunique polyol strings: {au.polyol.nunique()}; changed strings: "
      f"{int(au.drop_duplicates('polyol').pipe(lambda d: (d.r4 != d.fx).sum()))}; changed benchmark rows: "
      f"{int(((au.r4 != au.fx) & au.bm).sum())}/460")
print("polyol_type counts, benchmark 460 rows (round 4 -> fixed):")
for k in PT5:
    nb = int(((au.r4 == k) & au.bm).sum())
    na = int(((au.fx == k) & au.bm).sum())
    print(f"  {k:20}{nb:>5} -> {na:>4}")
print("\n=== AUDIT: iso_index_num (rows where round-4 and fixed differ) ===")
ch_iso = ~((F6.iso_index_num == F6f.iso_index_num) | (F6.iso_index_num.isna() & F6f.iso_index_num.isna()))
for s, grp in raw[ch_iso].groupby("iso_index"):
    i = grp.index[0]
    print(f"  {len(grp):>3} rows  '{s}': {F6.iso_index_num[i]:.3f} -> {F6f.iso_index_num[i]:.3f}")
print(f"  changed rows (all/benchmark): {int(ch_iso.sum())}/{int((ch_iso & bm).sum())}")
print("\n=== AUDIT: is_PIR (rows where round-4 and fixed differ) ===")
ch_pir = F6.is_PIR != F6f.is_PIR
for (a, b), grp in pd.DataFrame(dict(a=F6.is_PIR[ch_pir], b=F6f.is_PIR[ch_pir], p=raw.polyol[ch_pir].str[:70],
                                     iso=raw.iso_index[ch_pir].str[:30])).groupby(["a", "b"]):
    print(f"  {int(a)} -> {int(b)}: {len(grp)} rows; strings: {sorted(set(grp.p + ' / iso=' + grp.iso.fillna('')))[:6]}")
print(f"  is_PIR=1 count, benchmark rows: {int(F6.loc[bm, 'is_PIR'].sum())} -> {int(F6f.loc[bm, 'is_PIR'].sum())}; "
      f"changed rows (all/benchmark): {int(ch_pir.sum())}/{int((ch_pir & bm).sum())}")
print(f"iso_index_num non-null benchmark: {int(F6.loc[bm,'iso_index_num'].notna().sum())} -> "
      f"{int(F6f.loc[bm,'iso_index_num'].notna().sum())}")

# ---- c2 extras: loi_standard one-hot, log(density) -------------------------------------------------------------
STD_RX = [("ASTM_D2863", re.compile(r"ASTM\s*D\s*2863", re.I)), ("GB_T_2406", re.compile(r"GB\s*/?\s*T\s*2406", re.I)),
          ("ISO_4589", re.compile(r"ISO\s*4589", re.I))]


def std_label(s):
    s = str(s) if pd.notna(s) else ""
    hits = [(m.start(), k) for k, rx in STD_RX for m in [rx.search(s)] if m]
    return min(hits)[1] if hits else "other_unknown"


std_lab = raw["loi_standard"].map(std_label)
STDC = {f"std_{k}": (std_lab == k).astype(float) for k in ["ASTM_D2863", "GB_T_2406", "ISO_4589", "other_unknown"]}
print("\nloi_standard one-hot counts (benchmark): " + ", ".join(f"{k}={int(v[bm].sum())}" for k, v in STDC.items()))

dC = df3.copy()
for c in F6COLS5:
    dC[c] = F6f[c].to_numpy()
for k, v in STDC.items():
    dC[k] = v.to_numpy()
dC["log_density"] = np.log(dC.density_kgm3.astype(float))
FEAT_C1 = FEAT2 + F6COLS5 + ["el_imputed_share"] + F8COLS
FEAT_C2 = FEAT_C1 + list(STDC) + ["log_density"]
FEATS = {"c1": FEAT_C1, "c2": FEAT_C2}
DATA = {k: prep(dC[bm], f) for k, f in FEATS.items()}
for k in DATA:
    assert (DATA[k][1] == y).all() and (DATA[k][2] == g).all()
print(f"features: exp089 {len(FEAT_R4C2)}, c1 {len(FEAT_C1)}, c2 {len(FEAT_C2)}")


# ==== ROUND 6 ======================================================================================================
res, runs = {}, []


def run_cand(cid, X, pre_B=None):
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, rf().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    Bk, oof = pre_B if pre_B is not None else ep.repeated_group_kfold(rf(), X, y, g)
    for i, v in enumerate(Bk):
        runs.append(dict(protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    plo, phi = pooled_ci(y, oof, g)
    Ck, _ = ep.repeated_group_kfold(rf(), X, y, dom)
    for i, v in enumerate(Ck):
        runs.append(dict(protocol="C", cand=cid, rep=i, r2=v))
    print(f"  {cid} done in {time.time()-t:.0f}s")
    return dict(A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5), B_mean=Bk.mean(),
                B_std=Bk.std(), ci_lo=lo, ci_hi=hi, p_lo=plo, p_hi=phi, C_mean=Ck.mean(), Bk=Bk, oof0=oof[0])


# ---- step 1: reproduce exp092 (round-5 c2) -------------------------------------------------------------------------
t = time.time()
B92, oof92 = ep.repeated_group_kfold(rf(), DATA["c2"][0], y, g)
lo92, hi92 = ep.bootstrap_ci(y, oof92[0], g)
print(f"\nREPRO exp092: B {B92.mean():.4f} CI [{lo92:.4f}, {hi92:.4f}] (record B 0.3738, CI [0.2007, 0.4700]) "
      f"in {time.time()-t:.0f}s")
if not (abs(B92.mean() - 0.3738) <= 0.002 and abs(lo92 - 0.2007) <= 0.002 and abs(hi92 - 0.4700) <= 0.002):
    print("BLOCKED: exp092 not reproduced within 0.002; stopping.")
    sys.exit(2)
print("Reproduction within 0.002: True")

# ---- F6_v6 (config R1-R4, applied on top of F6_fixed) ---------------------------------------------------------------
RX_PES6 = re.compile(RX_PES.pattern + r"|PET glycolysis|glycoly[sz]ed (?:waste )?PET", re.I)        # R1
RX_PET6 = re.compile(RX_PET.pattern + r"|polypropylene glycol|polyethylene glycol", re.I)             # R2
RX_BIO_FIRST = re.compile(r"bio-?polyol|castor oil polyol|MCOP|ESBO|epoxidi[sz]ed soy", re.I)         # R3


def comp_label6(c):
    if RX_BIO_FIRST.search(c):
        return "bio_based"
    if RX_PES6.search(c):
        return "polyester_aromatic"
    if RX_PET6.search(c):
        return "polyether"
    if RX_BIO.search(c):
        return "bio_based"
    if RX_PLA.search(c):
        return "other_polyester"
    return "unknown"


def ptype_v6(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown", ""
    comps = split_components(str(s))
    has = [a is not None for _, a in comps]
    if sum(has) == 1 and not has[0]:                                                                    # R4
        chosen = comps[0][0]
    elif any(has):
        stated = [(p, a) for p, a in comps if a is not None and a > 0]
        if not stated:
            return "unknown", ""
        chosen = max(stated, key=lambda pa: pa[1])[0]
    else:
        chosen = comps[0][0]
    return comp_label6(chosen), chosen.strip()


v6 = raw["polyol"].map(ptype_v6)
v6_lab = v6.map(lambda t: t[0])
print("\n=== AUDIT (round 6): polyol strings whose label changes F6_fixed -> F6_v6 (benchmark rows / all rows) ===")
au6 = pd.DataFrame(dict(polyol=raw["polyol"].fillna("(NaN)"), fx=fixed_lab, v6=v6_lab, ch=v6.map(lambda t: t[1]), bm=bm))
for s, grp in au6[au6.fx != au6.v6].groupby("polyol", sort=False):
    print(f"  {int(grp.bm.sum()):>3}/{len(grp):<3} {grp.fx.iloc[0]:19} -> {grp.v6.iloc[0]:19} | {s[:100]}"
          f"  [chosen: {grp.ch.iloc[0][:50]}]")
print(f"changed benchmark rows: {int(((au6.fx != au6.v6) & au6.bm).sum())}/460")
for k in PT5:
    print(f"  {k:20}{int(((au6.fx == k) & au6.bm).sum()):>5} -> {int(((au6.v6 == k) & au6.bm).sum()):>4}")

# ---- 5-level standard (c5): PN-EN ISO 4589-3 split from ISO 4589 -------------------------------------------------
RX_43 = re.compile(r"ISO\s*4589-3", re.I)
std5 = std_lab.where(~((std_lab == "ISO_4589") & raw["loi_standard"].fillna("").str.contains(RX_43)), "ISO_4589_3")
STD5 = {f"std5_{k}": (std5 == k).astype(float) for k in ["ASTM_D2863", "GB_T_2406", "ISO_4589", "ISO_4589_3", "other_unknown"]}
print("5-level standard counts (benchmark): " + ", ".join(f"{k}={int(v[bm].sum())}" for k, v in STD5.items()))

d6 = dC.copy()
for k in PT5:
    d6[f"pt_{k}"] = (v6_lab == k).astype(float).to_numpy()
for k, v in STD5.items():
    d6[k] = v.to_numpy()
FEAT_C3 = FEAT_C1 + ["log_density"]
FEATS6 = {"c3": (dC, FEAT_C3), "c4": (d6, FEAT_C3), "c5": (d6, FEAT_C3 + list(STD5))}
for cid, (frame, f) in FEATS6.items():
    Xc, yc, gc, _ = prep(frame[bm], f)
    assert (yc == y).all() and (gc == g).all()
    DATA[cid] = (Xc,)
print(f"features: exp092 {len(FEAT_C2)}, c3 {len(FEAT_C3)}, c4 {len(FEAT_C3)}, c5 {len(FEAT_C3) + len(STD5)}")

# ==== ROUND 8 ======================================================================================================
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Avg(BaseEstimator, RegressorMixin):
    def __init__(self, ests):
        self.ests = ests

    def fit(self, X, y):
        self.f_ = [clone(e).fit(X, y) for e in self.ests]
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.f_], axis=0)


def et():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10.0))



# ==== PAIRED TEST (loi_snapshot_update6_paired) =====================================================================
# Folds come from the ORIGINAL rows only (GroupKFold shuffle, random_state = repeat, identical to eval_protocol); the
# wave-14/15 rows (row_id W9-*, W15-*) are never in a test fold. Arm 0 trains on original training rows; arm 1 on the same rows
# plus all wave-14 rows. Both arms score the same original test rows with the same features and model.
from sklearn.model_selection import GroupKFold
ids = df.loc[bm, "row_id"].to_numpy().astype(str)
is_new = np.array([i.startswith(("W9-", "W15-")) for i in ids])
old_idx, new_idx = np.flatnonzero(~is_new), np.flatnonzero(is_new)
print(f"PAIRED: {len(old_idx)} original rows / {len(np.unique(g[old_idx]))} papers; {len(new_idx)} wave-14 rows / {len(np.unique(g[new_idx]))} papers")
yo, go = y[old_idx], g[old_idx]
OUT = {}
for cid in ("c2", "c3"):
    X2 = DATA[cid][0]
    o0, o1 = np.empty((10, len(old_idx))), np.empty((10, len(old_idx)))
    t = time.time()
    for r in range(10):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X2[old_idx], yo, go):
            a, b = old_idx[tr], old_idx[te]
            assert set(g[a]).isdisjoint(g[b])
            o0[r, te] = Avg([rf(), et(), ridge()]).fit(X2[a], y[a]).predict(X2[b])
            ab = np.concatenate([a, new_idx])
            o1[r, te] = Avg([rf(), et(), ridge()]).fit(X2[ab], y[ab]).predict(X2[b])
    R0 = np.array([r2_score(yo, p) for p in o0])
    R1 = np.array([r2_score(yo, p) for p in o1])
    D = R1 - R0
    rng = np.random.default_rng(0)
    papers = np.unique(go)
    rows_of = {m: np.flatnonzero(go == m) for m in papers}
    bd = []
    for _ in range(2000):
        rr = np.concatenate([rows_of[m] for m in rng.choice(papers, len(papers))])
        bd.append(np.mean([r2_score(yo[rr], o1[k][rr]) - r2_score(yo[rr], o0[k][rr]) for k in range(10)]))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    OUT[cid] = dict(arm0_B=float(R0.mean()), arm1_B=float(R1.mean()), delta_mean=float(D.mean()), delta_sd=float(D.std()),
                    wins=int((D > 0).sum()), delta_ci_lo=float(lo), delta_ci_hi=float(hi), per_repeat_arm0=R0.tolist(),
                    per_repeat_arm1=R1.tolist())
    print(f"PAIRED {cid}: arm0 (606-only training) B {R0.mean():.4f}  arm1 (+wave14 training) B {R1.mean():.4f}  "
          f"delta {D.mean():+.4f} (sd {D.std():.4f}) wins {int((D > 0).sum())}/10  paper-bootstrap CI of delta [{lo:+.4f}, {hi:+.4f}]  ({time.time()-t:.0f}s)", flush=True)
    print(f"PAIRED {cid} per-repeat delta: " + " ".join(f"{v:+.4f}" for v in D), flush=True)
(ROOT / "experiments/results/loi_paired_wave14_15.json").write_text(json.dumps(OUT, indent=2))
print(f"wrote paired result; runtime {time.time()-T0:.0f}s")
