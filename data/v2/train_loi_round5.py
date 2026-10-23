#!/usr/bin/env python3
"""LOI round 5 -- follows experiments/configs/loi_round5.json exactly (c1, c2; no extra candidates, no tuning).

Data loading / structures / element features / RF config / protocols A, B, C / seeds / CIs / F7 / F8 are copied verbatim
from train_loi_round4.py. Step 1 reproduces exp089 (= round-4 c2: F6 round-4 + F7 + F8; B 0.3542, CI [0.1694, 0.4666],
tolerance 0.002) or exits blocked. Then F6_fixed is built and its audit printed BEFORE any round-5 model is fit.

F6_fixed implementation notes (declared rule, literal regexes, re.I except PIR which is case-sensitive as in round 4):
 - polyol text is split into components at top-level " + " (outside parentheses); "A / B wt% = a / b" gives two components.
 - amount of a component (number not preceded by a letter/dot/hyphen, so grades like '400-4G' are not amounts): first "<num> g|pbw|part(s)|php|phr|pph|wt%|%" in the component text; else, a trailing bare number
   (after removing parentheses). Amounts are used if >= 2 components have one, or exactly one has a unit amount; else the
   first-listed component is used. Zero amounts are ignored; ties -> first listed; components without an amount are
   ignored when others have one.
 - component label: polyester_aromatic, then polyether, then bio_based, then PLA (\\bPLA) -> other_polyester, else unknown
   (declared order; the polyether regex is checked before the bio rule).
 - iso_index_num: "R a:b" -> a/b, "a:1" -> a; else round-4 parse.
 - is_PIR: iso_index_num >= 1.8 or /PIR|polyisocyanurate/ in polyol|isocyanate|notes, with the phrase
   "isocyanurate-based polyether polyol" removed before the test.
c2 extra: loi_standard one-hot {ASTM D2863, GB/T 2406, ISO 4589, other/unknown}: the EARLIEST regex match in the text wins
(strings naming two standards go to the first-named); plus log(density_kgm3).

Run: OMP_NUM_THREADS=1 python3 data/v2/train_loi_round5.py | tee experiments/runs/train_loi_round5.log
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

# ---- run c1, c2 (+ exp089 A/C for the table) ---------------------------------------------------------------------
res, runs = {}, []


def run_cand(cid, X, pre_B=None):
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, rf().fit(Xtr, ytr).predict(Xte)))
        runs.append(dict(protocol="A", cand=cid, rep=s, r2=A[-1]))
    A = np.array(A)
    if pre_B is None:
        Bk, oof = ep.repeated_group_kfold(rf(), X, y, g)
    else:
        Bk, oof = pre_B
    for i, v in enumerate(Bk):
        runs.append(dict(protocol="B", cand=cid, rep=i, r2=v))
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    plo, phi = pooled_ci(y, oof, g)
    Ck, _ = ep.repeated_group_kfold(rf(), X, y, dom)
    for i, v in enumerate(Ck):
        runs.append(dict(protocol="C", cand=cid, rep=i, r2=v))
    r = dict(A_mean=A.mean(), A_lo=np.percentile(A, 2.5), A_hi=np.percentile(A, 97.5), B_mean=Bk.mean(), B_std=Bk.std(),
             ci_lo=lo, ci_hi=hi, p_lo=plo, p_hi=phi, C_mean=Ck.mean(), Bk=Bk)
    print(f"  {cid} done in {time.time()-t:.0f}s")
    return r


res["exp089"] = run_cand("exp089", X89, pre_B=(B89, oof89))
for cid in ("c1", "c2"):
    res[cid] = run_cand(cid, DATA[cid][0])
    res[cid]["wins"] = int((res[cid]["Bk"] > B89).sum())
    res[cid]["delta"] = float((res[cid]["Bk"] - B89).mean())

NAME = {"exp089": "exp089 (record)", "c1": "c1 r4c2 + F6_fixed", "c2": "c2 c1 + standard + logdens"}
print("\n=== Summary (460 rows / 83 papers) ===")
print(f"{'cand':28}{'A mean [p2.5, p97.5]':24}{'B mean ± std':15}{'B CI rep0':18}{'B CI pooled':18}{'C':8}{'wins':6}{'dB'}")
for cid in ("exp089", "c1", "c2"):
    r = res[cid]
    print(f"{NAME[cid]:28}" + f"{r['A_mean']:.3f} [{r['A_lo']:.3f}, {r['A_hi']:.3f}]".ljust(24)
          + f"{r['B_mean']:.3f} ± {r['B_std']:.3f}".ljust(15) + f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]".ljust(18)
          + f"[{r['p_lo']:.3f}, {r['p_hi']:.3f}]".ljust(18) + f"{r['C_mean']:.3f}".ljust(8)
          + (f"{r['wins']}/10".ljust(6) + f"{r['delta']:+.4f}" if "wins" in r else "-"))
best = max(("c1", "c2"), key=lambda c: res[c]["B_mean"])
rb = res[best]
adopt = rb["wins"] >= 8 and rb["ci_lo"] >= 0.169
print(f"\nSELECTION (highest B mean): {best} B {rb['B_mean']:.4f}; paired wins {rb['wins']}/10 (need >= 8), "
      f"B ci_lo {rb['ci_lo']:.4f} (need >= 0.169) -> "
      + ("ADOPT candidate over exp089 pending critic" if adopt else "NOT adopted; exp089 stays the record"))
trig = []
for cid in ("c1", "c2"):
    r = res[cid]
    if r["ci_lo"] > SOTA:
        trig.append(f"TRIGGER: B ci_lo ({cid}) {r['ci_lo']:.3f} > {SOTA}")
    if r["A_mean"] > SOTA:
        trig.append(f"TRIGGER: A mean ({cid}) {r['A_mean']:.3f} > {SOTA}")
print("Triggers:", trig if trig else "none")

pd.DataFrame(runs).to_csv(ROOT / "experiments/runs/loi_round5_all.csv", index=False)
SHORT = {"c1": "f6fixed", "c2": "f6fixed_std_logdens"}
DESC = {"c1": "RF (exp085 config) + round-4 c2 features with F6 replaced by F6_fixed (polyol_type 5-level one-hot, "
              "iso_index_num ratio fix, is_PIR fix)",
        "c2": "c1 + loi_standard one-hot (ASTM D2863 / GB/T 2406 / ISO 4589 / other) + log(density_kgm3)"}
for cid in ("c1", "c2"):
    r = res[cid]
    note = (f"A {r['A_mean']:.3f} [{r['A_lo']:.3f},{r['A_hi']:.3f}]; pooled-over-repeats CI [{r['p_lo']:.3f},{r['p_hi']:.3f}]; "
            f"C {r['C_mean']:.3f}; {len(y)} rows/{len(np.unique(g))} papers; round 5 per loi_round5.json")
    out = dict(slug=f"loi_r5_{cid}_{SHORT[cid]}", target="LOI (%) RPUF literature",
               round_config="experiments/configs/loi_round5.json",
               dataset="data/v2/loi_rpuf_literature.csv, loi_pct notna and extraction != figure (460 rows / 83 papers)",
               features=FEATS[cid], model="RandomForestRegressor",
               hyperparameters="n_estimators=500, min_samples_leaf=2, random_state=0, NaN-native; " + DESC[cid],
               code="data/v2/train_loi_round5.py", n_rows=int(len(y)), n_groups=int(len(np.unique(g))),
               protocol="gkf_paper5x10", rkf_mean=round(float(r["B_mean"]), 4), rkf_std=round(float(r["B_std"]), 4),
               ci_lo=round(float(r["ci_lo"]), 4), ci_hi=round(float(r["ci_hi"]), 4), pooled_ci_lo=round(float(r["p_lo"]), 4),
               pooled_ci_hi=round(float(r["p_hi"]), 4), gss_mean=round(float(r["A_mean"]), 4),
               c_mean=round(float(r["C_mean"]), 4), struct_rows=41, struct_papers=14,
               paired_vs="exp089", paired_delta=round(r["delta"], 4), paired_wins=f"{r['wins']}/10", sota=SOTA,
               sota_source="sun_fu_2026", status="ok", blocked_reason="", note=note)
    (ROOT / f"experiments/results/loi_r5_{cid}_{SHORT[cid]}.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote 2 result JSONs; runtime {time.time()-T0:.0f}s")
