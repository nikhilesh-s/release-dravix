"""Flame-retardant chemistry features and FR grouping, shared by the USTC scripts and the runner export.
fr_group(): first-FR normalised name (legacy, has a synonym leak; kept so old results reproduce).
strict_group(): synonym alias map (fr_group_alias_all.csv) + InChIKey-skeleton merge (fr_smiles_extra.csv),
as in ustc-challenger run_attempt.py strict_groups(); use this for every new score."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import train_litcone_gt_round1 as T
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


_alias = pd.read_csv(T.PARTS / "fr_group_alias_all.csv", dtype=str).set_index("fr_group_current")["fr_identity"]
_sm = pd.read_csv(T.PARTS / "fr_smiles_extra.csv", dtype=str).drop_duplicates("name").set_index("name")["smiles"]

def strict_groups(components):
    """Leak-free FR identity per row: alias map over first-FR keys, then merge identities sharing an InChIKey skeleton."""
    first = [fr_group(c) for c in components]
    ident = {k: _alias.get(k, k) for k in set(first)}
    seen, merge = {}, {}
    for k in sorted(ident):
        smi = _sm.get(k)
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None: continue
        merge[ident[k]] = seen.setdefault(Chem.MolToInchiKey(m).split("-")[0], ident[k])
    return np.array([merge.get(ident[k], ident[k]) for k in first])
