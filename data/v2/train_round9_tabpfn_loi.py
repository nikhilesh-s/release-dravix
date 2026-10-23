#!/usr/bin/env python3
"""Round 9 LOI arm (B2 TabPFN p1, p2) -- experiments/configs/round9_tabpfn_smearing.json, followed literally.

train_loi_rescore_r8.py verbatim (its lines 9-637: data prep, features, the exp089/exp092 checks that
rescore_loi_snapshot.patch_round6 turns into reports, and the round-8 Avg / et / ridge definitions), then:
step 1 = exp115 (Avg(rf, et, ridge) on DATA['c2'][0]) reproduced within 0.002 or exit 2; then p1 = TabPFN V2 alone and
p2 = Avg([exp115 model, TabPFN V2]) on the same 51 FEAT_C2 columns, raw LOI, protocols A / B / C as round 8.
Writes experiments/results/r9_tabpfn_loi_p{1,2}.json and experiments/runs/round9_loi_oof.npz (inside the scratch copy).

Run (never directly):
  python data/v2/rescore_loi_snapshot.py --stage data/v2/loi_rpuf_parts/wave4_staging \
      --stage data/v2/loi_rpuf_parts/wave5_staging --script data/v2/train_round9_tabpfn_loi.py --keep
Built with PriorLabs-TabPFN.
"""
import os
from pathlib import Path as _P

# TabPFN / HF environment (round9 tabpfn_model.environment_for_every_tabpfn_process), set before tabpfn is imported.
# rescore_loi_snapshot.py runs this script with a stripped environment, so everything is set here explicitly.
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(_P.home() / "Library" / "Caches" / "tabpfn")
os.environ["TABPFN_EXCLUDE_DEVICES"] = "cuda,mps"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# compute only (worker limit): n_jobs=-1 forests resolve to this many threads; forest output does not depend on n_jobs
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# ---- train_loi_rescore_r8.py lines 9-637 verbatim from here --------------------------------------------------------
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


# ---- end of the verbatim train_loi_rescore_r8.py prefix -------------------------------------------------------------

# ==== ROUND 9 (B2): exp115 reproduction, then TabPFN p1, p2 ===========================================================
import hashlib  # noqa: E402

import sklearn  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import tabpfn  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

CONFIG = "experiments/configs/round9_tabpfn_smearing.json"
CKPT = _P(os.environ["TABPFN_MODEL_CACHE_DIR"]) / "tabpfn-v2-regressor.ckpt"
SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
ATTR = "Built with PriorLabs-TabPFN"
FEAT_CFG = ["total_fr_wtpct", "cls_P-additive", "cls_P-reactive", "cls_N-additive", "cls_P-N", "cls_intumescent(APP-based)",
            "cls_expandable_graphite", "cls_metal_hydroxide", "cls_metal_oxide/salt", "cls_silicon", "cls_boron", "cls_halogen",
            "cls_nanoclay", "cls_carbon_nano(graphene/CNT)", "cls_bio-based", "cls_other", "el_P", "el_N", "el_Si", "el_B",
            "el_halogen", "el_metal", "density_kgm3", "n_frs", "has_EG", "has_APP", "reactive", "fr_arom_frac", "fr_fsp3",
            "fr_PO_per_kDa", "fr_PN_per_kDa", "fr_mollogp", "fr_tpsa_per_kDa", "resolved_fr_wt_share", "pt_polyether",
            "pt_polyester_aromatic", "pt_bio_based", "pt_other_polyester", "pt_unknown", "iso_index_num", "is_PIR",
            "el_imputed_share", "syn_PxN", "syn_P_over_N", "syn_EG_x_P", "syn_totfr_x_dens", "std_ASTM_D2863", "std_GB_T_2406",
            "std_ISO_4589", "std_other_unknown", "log_density"]
assert list(FEAT_C2) == FEAT_CFG, "FEAT_C2 differs from the config's 51 columns"
X2 = DATA["c2"][0]
N, NP = int(len(y)), int(len(np.unique(g)))
print(f"\n=== ROUND 9 LOI: {N} rows / {NP} papers / {X2.shape[1]} features; torch threads {torch.get_num_threads()}, "
      f"LOKY_MAX_CPU_COUNT {os.environ.get('LOKY_MAX_CPU_COUNT')}, HF_HUB_OFFLINE {os.environ['HF_HUB_OFFLINE']}")
assert (N, NP) == (606, 107), (N, NP)
h = hashlib.sha256()
with open(CKPT, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 20), b""):
        h.update(blk)
CK_SHA = h.hexdigest()
assert CK_SHA == SHA256, "checkpoint hash mismatch"


def tabpfn_model():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


def run_all(mk):
    """Protocols as the round-8 loop: A (50 random 80/20 splits), B (paper GKF 5x10), repeat-0 CI, pooled CI, C (dom GKF)."""
    t = time.time()
    A = []
    for s in range(50):
        Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.2, random_state=s)
        A.append(r2_score(yte, mk().fit(Xtr, ytr).predict(Xte)))
    A = np.array(A)
    Bk, oof = ep.repeated_group_kfold(mk(), X2, y, g)
    lo, hi = ep.bootstrap_ci(y, oof[0], g)
    plo, phi = pooled_ci(y, oof, g)
    Ck, _ = ep.repeated_group_kfold(mk(), X2, y, dom)
    return dict(A_mean=float(A.mean()), Bk=Bk, oof=oof, B_mean=float(Bk.mean()), B_std=float(Bk.std()), ci_lo=float(lo),
                ci_hi=float(hi), p_lo=float(plo), p_hi=float(phi), C_mean=float(Ck.mean()), wall=time.time() - t)


def show(cid, r, inc=None):
    s = (f"{cid:6s} A {r['A_mean']:.4f}  B {r['B_mean']:.4f} +- {r['B_std']:.4f} CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
         f"pooled [{r['p_lo']:.4f}, {r['p_hi']:.4f}]  C {r['C_mean']:.4f}")
    if inc is not None:
        d = r["Bk"] - inc["Bk"]
        s += f"  dB {d.mean():+.4f} wins {(d > 0).sum()}/10"
    print(s + f"  ({r['wall']:.0f}s)", flush=True)


def topk(oof_inc, oof_c):
    """reported_only.top_k_removal with papers as groups. Diagnostic only."""
    row = ((y - oof_inc) ** 2 - (y - oof_c) ** 2).mean(axis=0)
    con = {str(m): float(row[g == m].sum()) for m in np.unique(g)}
    ranked = sorted(con, key=lambda m: (-con[m], m))
    tot = sum(con.values())
    out = dict(top5=[[m, con[m]] for m in ranked[:5]], top5_share_of_net=sum(con[m] for m in ranked[:5]) / tot if tot else None)
    for k in (1, 3, 5):
        keep = ~np.isin(g.astype(str), ranked[:k])
        d = np.array([r2_score(y[keep], oof_c[r, keep]) - r2_score(y[keep], oof_inc[r, keep]) for r in range(10)])
        out[f"drop_top{k}"] = dict(mean_gain=float(d.mean()), wins=int((d > 0).sum()))
    out["not_established"] = bool(out["drop_top5"]["mean_gain"] <= 0 or out["drop_top5"]["wins"] < 8)
    return out


# ---- step 1: exp115 ----
inc = run_all(lambda: Avg([rf(), et(), ridge()]))
show("exp115", inc)
if not (abs(inc["B_mean"] - 0.3875) <= 0.002 and abs(inc["ci_lo"] - 0.3124) <= 0.002 and abs(inc["ci_hi"] - 0.5289) <= 0.002):
    print("BLOCKED: exp115 reproduction outside 0.002")
    sys.exit(2)
print("REPRO exp115 ok (record B 0.3875 [0.3124, 0.5289])")
for name, v, ref in (("A", inc["A_mean"], 0.7175), ("C", inc["C_mean"], 0.2475)):
    print(f"  protocol {name} {v:.4f} vs record {ref}: " + ("within 0.002" if abs(v - ref) <= 0.002 else "FLAG: outside 0.002"))

# ---- candidates ----
res = {}
res["p1"] = run_all(tabpfn_model)
show("p1", res["p1"], inc)
res["p2"] = run_all(lambda: Avg([Avg([rf(), et(), ridge()]), tabpfn_model()]))
show("p2", res["p2"], inc)
tr0 = next(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X2, y, g))[0]
fm = tabpfn_model().fit(X2[tr0], y[tr0])  # metadata only
attrs = dict(n_estimators_=int(fm.n_estimators_), devices_=[str(d) for d in fm.devices_], use_autocast_=bool(fm.use_autocast_),
             forced_inference_dtype_=None if fm.forced_inference_dtype_ is None else str(fm.forced_inference_dtype_))
print("TabPFN fitted attributes:", attrs)
print("Triggers:", [c for c in res if res[c]["ci_lo"] > SOTA or res[c]["A_mean"] > SOTA] or "none")

DESC9 = {"p1": "TabPFN V2 alone on the 51 FEAT_C2 columns, raw LOI",
         "p2": "Avg([exp115 model (Avg(RF, ET, Ridge)), TabPFN V2]) raw scale, nested (RF/ET/Ridge 1/6 each, TabPFN 1/2)"}
for cid in ("p1", "p2"):
    r = res[cid]
    d = r["Bk"] - inc["Bk"]
    tk = topk(inc["oof"], r["oof"])
    slug = f"r9_tabpfn_loi_{cid}"
    note = f"round 9 LOI {cid} (provisional id {slug}); report vs exp115 on identical folds; A {r['A_mean']:.3f}; C {r['C_mean']:.3f}"
    if tk["not_established"]:
        note += "; gain concentrated in <= 5 papers, not established"
    note += f"; {ATTR}"
    out = dict(slug=slug, target="LOI (%) RPUF literature", round_config=CONFIG,
               dataset=f"LOI snapshot 3 ({N} rows / {NP} papers), built by rescore_loi_snapshot.py (wave4 + wave5 staging); exp115 rows",
               features=FEAT_C2, model=DESC9[cid],
               hyperparameters="TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=0); package defaults, tabpfn 9.1.0",
               code="data/v2/train_round9_tabpfn_loi.py", n_rows=N, n_groups=NP, protocol="gkf_paper5x10",
               rkf_mean=round(r["B_mean"], 4), rkf_std=round(r["B_std"], 4), ci_lo=round(r["ci_lo"], 4), ci_hi=round(r["ci_hi"], 4),
               pooled_ci_lo=round(r["p_lo"], 4), pooled_ci_hi=round(r["p_hi"], 4), gss_mean=round(r["A_mean"], 4),
               c_mean=round(r["C_mean"], 4), paired_vs="exp115", paired_delta=round(float(d.mean()), 4),
               paired_wins=f"{int((d > 0).sum())}/10", sota=SOTA, sota_source="sun_fu_2026", status="ok", blocked_reason="",
               note=note, rkf_mean_full=r["B_mean"], ci_lo_full=r["ci_lo"], ci_hi_full=r["ci_hi"],
               per_repeat_r2=[float(v) for v in r["Bk"]], top_k_removal=tk, wall_time_s=round(r["wall"], 1),
               tabpfn_version=tabpfn.__version__, torch_version=torch.__version__, sklearn_version=sklearn.__version__,
               checkpoint=CKPT.name, checkpoint_sha256=CK_SHA, attribution=ATTR, **attrs)
    (ROOT / f"experiments/results/{slug}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote experiments/results/{slug}.json  top-k: {json.dumps({k: tk[k] for k in ('drop_top1', 'drop_top3', 'drop_top5')})}")
npz = ROOT / "experiments/runs/round9_loi_oof.npz"
np.savez(npz, y=y, groups=np.asarray(g).astype(str), row_id=np.arange(N), oof_incumbent=inc["oof"],
         inc_per_repeat_r2=inc["Bk"], inc_ci=np.array([inc["ci_lo"], inc["ci_hi"]]), order=np.array(["p1", "p2"]),
         selectable=np.array(["p1", "p2"]), oof_p1=res["p1"]["oof"], oof_p2=res["p2"]["oof"])
print(f"wrote {npz} (copy into the repo's experiments/runs/, then run the gate)")
print(f"runtime {time.time()-T0:.0f}s")
