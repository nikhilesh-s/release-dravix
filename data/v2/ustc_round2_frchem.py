"""Amendment 11: v4 = v3 + FR element/class loadings + RDKit descriptors of the main FR. USTC-style protocol
(GroupKFold by FR identity, 5x10, pHRR MAE) on (A) all EP rows and (B) USTC-matched: EP papers using P-containing FRs."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import train_litcone_gt_round1 as T
T.TOPN, T.PHYS, T.CLIP = 100, True, True
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

ch = pd.read_csv(T.PARTS / "fr_chemistry.csv", dtype=str).set_index("name")
EL = ["P", "N", "Si", "B", "S", "halogen", "metal"]
CL = sorted(ch["fr_class"].dropna().unique())
def num(x):
    try: return float(x)
    except Exception: return 0.0
def desc(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return [np.nan] * 4
    return [Descriptors.MolWt(m), rdMolDescriptors.CalcNumAromaticRings(m),
            sum(a.GetSymbol() == "P" for a in m.GetAtoms()), Descriptors.TPSA(m)]

def frchem(df):
    rows = []
    for comp in df["components"]:
        cs = T.parse_components(comp)
        units = {c[3] for c in cs if c[1] != "matrix" and not np.isnan(c[2])}
        tot = 100.0 + sum(c[2] for c in cs if c[1] != "matrix" and not np.isnan(c[2])) if units == {"phr"} else None
        f = dict.fromkeys([f"el_{e}" for e in EL] + [f"cl_{c}" for c in CL] + ["reactive_wt"], 0.0)
        main, mainw = None, -1
        for name, role, v, u in cs:
            if role not in ("fr", "synergist") or np.isnan(v): continue
            w = 100 * v / tot if tot else (v if u == "wt%" else np.nan)
            if np.isnan(w): continue
            n = T._norm(name)
            if n in ch.index:
                r = ch.loc[n]
                for e in EL: f[f"el_{e}"] += w * num(r[e])
                if isinstance(r["fr_class"], str): f[f"cl_{r['fr_class']}"] += w
                if r["reactive"] == "1": f["reactive_wt"] += w
                if w > mainw: main, mainw = r["smiles"], w
        d = desc(main)
        f.update(dict(fr_mw=d[0], fr_arom=d[1], fr_nP=d[2], fr_tpsa=d[3]))
        rows.append(f)
    return pd.DataFrame(rows, index=df.index)

def fr_group(comp):
    for name, role, v, u in T.parse_components(comp):
        if role in ("fr", "synergist"): return T._norm(name)
    return "none"

def has_P(comp):
    return any(role in ("fr", "synergist") and num(ch.loc[T._norm(n), "P"]) > 0
               for n, role, v, u in T.parse_components(comp) if T._norm(n) in ch.index)

df = pd.read_csv(T.SNAP, low_memory=False)
for c in T.NUM: df[c] = pd.to_numeric(df[c], errors="coerce")
cm = pd.read_csv(T.CLASS_MAP, dtype=str); cmap = dict(zip(cm["component_name"].str.strip().str.lower(), cm["class"]))
ppm = df.groupby("matrix_canonical")["group"].nunique()
X3 = T.build_X(df, cmap, sorted(ppm[ppm >= 3].index.difference(["other"])), sorted(df["material_form"].dropna().unique()))
X4 = pd.concat([X3, frchem(df)], axis=1)
ep = ((df["matrix_canonical"] == "EP") & (df["phrr_kWm2"] > 0) & df["heat_flux_kWm2"].notna()).to_numpy()
pP = df["components"].map(has_P).to_numpy()
p_papers = set(df.loc[ep & pP, "group"])
popB = ep & (pP | ((df["is_control"] == 1) | (df["components"].map(fr_group) == "none")).to_numpy()) & df["group"].isin(p_papers).to_numpy()

def run(mask, X, model_fn, label):
    sub = df[mask].reset_index(drop=True); Xm = X[mask].to_numpy(float); y = sub["phrr_kWm2"].to_numpy(float)
    g = sub["components"].map(fr_group).to_numpy(); ug = np.unique(g)
    maes, r2s, oof0 = [], [], None
    for r in range(10):
        perm = np.random.default_rng(r).permutation(ug); fold = {k: i % 5 for i, k in enumerate(perm)}
        f = np.array([fold[k] for k in g]); oof = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            oof[te] = T.fit_pred(model_fn(), Xm[tr], y[tr], Xm[te])
        maes.append(mean_absolute_error(y, oof)); r2s.append(r2_score(y, oof))
        if r == 0: oof0 = oof
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(g == k)[0] for k in pick])
        bs.append(mean_absolute_error(y[idx], oof0[idx]))
    print(f"{label}: n={len(y)} papers={sub['group'].nunique()} FR groups={len(ug)} | MAE {np.mean(maes):.1f} +- {np.std(maes):.1f} "
          f"CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] | R2 {np.mean(r2s):.3f}", flush=True)

print("amendment 11 -- USTC Li et al. 2025 pHRR MAE 125.5 (unconfirmed)")
run(popB, X4, T.primary, "HEADLINE primary v4, B (EP + P-FRs, USTC-matched)")
run(popB, X3, T.primary, "  ref primary v3, B")
run(popB, X4, T.gt_protocol_et, "  gt_protocol_et v4, B")
run(ep, X4, T.primary, "  primary v4, A (all EP)")
run(ep, X4, T.gt_protocol_et, "  gt_protocol_et v4, A")
