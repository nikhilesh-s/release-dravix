"""USTC challenger harness. Usage: python run_attempt.py <attempt_id> [--seeds 0-9] [--dev]
Reads the declaration from attempts.json (must exist BEFORE scoring), reuses Arnav's amendment-11/13 setup
(data/v2/ustc_round2_frchem.py) and keeps his exact protocol: GroupKFold-5 by FR identity, fold map = seed-r
permutation of the test population's FR groups, 10 repeats, pooled OOF MAE + repeat-0 group-bootstrap CI + R2.
--dev scores on the dev population (EP rows with a control whose FR has no P; disjoint from B') instead."""
import sys, json, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1] / "data/v2"
sys.path.insert(0, str(V2))
_src = (V2 / "ustc_round2_frchem.py").read_text().split('print("amendment 11')[0]
exec(compile(_src, str(V2 / "ustc_round2_frchem.py"), "exec"))  # defines T, df, X3, X4, ep, pP, fr_group, frchem ...
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ap = argparse.ArgumentParser(); ap.add_argument("aid"); ap.add_argument("--seeds", default="0-9"); ap.add_argument("--dev", action="store_true")
a = ap.parse_args()
s0, s1 = map(int, a.seeds.split("-")); SEEDS = range(s0, s1 + 1)
D = json.load(open(HERE / "attempts.json"))[a.aid]
print(f"attempt {a.aid}: {D['declaration']}\nseeds {a.seeds}{' DEV' if a.dev else ''}", flush=True)

# ---- inputs (no targets used anywhere below until y/t)
ctrl = np.expm1(T.control_feature(df, "phrr_kWm2"))          # same paper's neat control, same flux (leak-fixed)
has = ~np.isnan(ctrl) & (df["is_control"] != 1).to_numpy()
alias = pd.read_csv(HERE / "fr_group_alias.csv", dtype=str).set_index("fr_group_current")
g_first = df["components"].map(fr_group).to_numpy()

def wt_shares(cs):
    """(name, role, wt%) for non-matrix components, same unit logic as T.class_features (phr / wt% / g)."""
    units = {c[3] for c in cs if c[1] != "matrix" and not np.isnan(c[2])}
    if units == {"phr"}:
        tot = 100.0 + sum(c[2] for c in cs if c[1] != "matrix" and not np.isnan(c[2])); f = lambda v: 100 * v / tot
    elif units == {"wt%"}: f = lambda v: v
    elif units == {"g"}:
        tot = sum(c[2] for c in cs if not np.isnan(c[2])); f = lambda v: 100 * v / tot if tot else np.nan
    else: f = lambda v: np.nan
    return [(T._norm(n), r, f(v)) for n, r, v, u in cs if r != "matrix" and not np.isnan(v)]

sm = pd.read_csv(HERE / "fr_smiles_extra.csv", dtype=str).drop_duplicates("name").set_index("name")
def strict_groups():
    """alias_all: synonym map over every first-FR key in the all-matrix pool (names-only), then merge identities whose
    SMILES share an InChIKey skeleton (first block)."""
    al = pd.read_csv(HERE / "fr_group_alias_all.csv", dtype=str).set_index("fr_group_current")["fr_identity"]
    ident = {k: al.get(k, k) for k in set(g_first)}
    first, merge = {}, {}
    for k in sorted(ident):
        m = Chem.MolFromSmiles(sm["smiles"].get(k)) if isinstance(sm["smiles"].get(k), str) else None
        if m is None: continue
        ik = Chem.MolToInchiKey(m).split("-")[0]
        merge[ident[k]] = first.setdefault(ik, ident[k])    # identity -> first identity seen with this skeleton
    global _al, _merge; _al, _merge = al, merge
    return np.array([merge.get(ident[k], ident[k]) for k in g_first])
G = {"first_fr": lambda: g_first, "alias": lambda: np.array([alias["fr_identity"].get(k, k) for k in g_first]),
     "alias_all": strict_groups}[D.get("groups", "first_fr")]()
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=D.get("ecfp_bits", 128))
def ecfp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return None
    return fpg.GetFingerprintAsNumPy(m).astype(float)

def extra_feats():
    nb = D.get("ecfp_bits", 128); rows = []
    for comp in df["components"]:
        sh = [s for s in wt_shares(T.parse_components(comp)) if s[1] in ("fr", "synergist")]
        pl, best, bw = 0.0, None, -1
        for n, r, w in sh:
            pf = pd.to_numeric(sm["P_mass_frac"].get(n), errors="coerce")
            if not np.isnan(w) and not np.isnan(pf): pl += w * pf
            if not np.isnan(w) and w > bw and isinstance(sm["smiles"].get(n), str): best, bw = sm["smiles"].get(n), w
        fr_tot = np.nansum([w for _, _, w in sh]) if sh else 0.0
        fp = ecfp(best) if "ecfp" in D["features"] else None
        row = {"P_load_wt": pl, "fr_load_wt": fr_tot}
        if "ecfp" in D["features"]:
            row.update({f"ecfp_{i}": (fp[i] if fp is not None else np.nan) for i in range(nb)})
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)

X = X4.copy()
if "noA" in D["features"]: X = X[[c for c in X.columns if not c.startswith("a_")]]
if "pload" in D["features"] or "ecfp" in D["features"]: X = pd.concat([X, extra_feats()], axis=1)
if "ctrlbase" in D["features"]:                              # same paper's neat-control cone values (inputs, not this row's outcomes)
    for c in ["tti_s", "thr_MJm2", "avg_hrr_kWm2", "time_to_phrr_s", "char_residue_pct", "tsr_m2m2", "ehc_MJkg"]:
        X[f"ctrl_{c}"] = T.control_feature(df, c)
Xa = X.to_numpy(float)

# ---- populations
if a.dev:
    test = ep & has & ~pP                                    # dev: non-P EP rows with a control (disjoint from B')
elif D["population"] == "Bprime":
    test = ep & has & pP                                     # Arnav amendment 13 B'
elif D["population"] == "v2":                                # Data Matcher primary: EP bulk, organophosphorus first FR
    orgP = np.array([alias["organophosphorus"].get(k, "0") == "1" for k in g_first])
    test = ep & has & (df["material_form"] == "bulk").to_numpy() & orgP
pool = {"test_pop": test, "ep_ctrl": ep & has, "all_ctrl": has & (df["phrr_kWm2"] > 0).to_numpy() & df["heat_flux_kWm2"].notna().to_numpy()}[D["train_pool"]]

y_all = df["phrr_kWm2"].to_numpy(float)

def model():
    if D["model"] == "primary": return T.primary()
    if D["model"] == "et": return T.gt_protocol_et()
    if D["model"] == "hgb_et": return T.Avg(T.primary().members[:2])  # blend without the Ridge member
    if D["model"] == "hgb": return T.primary().members[0]
    if D["model"] == "hgb_l1":
        from sklearn.pipeline import make_pipeline
        return make_pipeline(T.imp(), HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                                    max_leaf_nodes=15, min_samples_leaf=5, random_state=0))
    raise ValueError(D["model"])

def fit_predict(tr, te):
    c_tr, c_te = ctrl[tr], ctrl[te]
    if D["target"] == "logratio":                            # amendment 13 target
        t = np.log(y_all[tr] / c_tr); m = clone(model()).fit(Xa[tr], t)
        return c_te * np.exp(np.clip(m.predict(Xa[te]), t.min(), t.max()))
    if D["target"] == "ratio_w":                             # ratio, weighted by control => loss in kW/m2 units
        t = y_all[tr] / c_tr; m = clone(model())
        m.fit(Xa[tr], t, **{f"{m.steps[-1][0]}__sample_weight": c_tr} if hasattr(m, "steps") else {})
        return c_te * np.clip(m.predict(Xa[te]), t.min(), t.max())
    raise ValueError(D["target"])

def row_ids(comp):                                               # every FR/synergist identity in a row (RC1: exclude_any_fr)
    out = set()
    for n, r, v, u in T.parse_components(comp):
        if r in ("fr", "synergist"):
            k = T._norm(n); k = _al.get(k, k) if D.get("groups") == "alias_all" else k
            out.add(_merge.get(k, k) if D.get("groups") == "alias_all" else k)
    return out
IDS = [row_ids(c) for c in df["components"]] if D.get("exclude_any_fr") else None

ti = np.where(test)[0]; y = y_all[ti]; g = G[ti]; ug = np.unique(g)
assert not np.isnan(ctrl[ti]).any() and (df["is_control"].to_numpy()[ti] != 1).all()
M, R, B, oof0 = [], [], [], None
for r in SEEDS:
    perm = np.random.default_rng(r).permutation(ug); fold = {k: i % 5 for i, k in enumerate(perm)}
    f = np.array([fold[k] for k in g]); oof = np.zeros(len(y)); base = np.zeros(len(y))
    for k in range(5):
        te_groups = set(ug[[fold[u] == k for u in ug]])
        te = ti[f == k]
        drop = np.isin(G, list(te_groups))                          # no test-fold FR identity anywhere in training
        if D.get("exclude_test_papers"): drop |= df["group"].isin(set(df["group"].iloc[te])).to_numpy()
        if IDS is not None:                                         # RC1: also drop any row sharing ANY FR/synergist identity with the test fold
            tid = set().union(*(IDS[i] for i in te)); drop |= np.array([bool(x & tid) for x in IDS])
        tr = np.where(pool & ~drop)[0]
        assert not set(G[tr]) & set(G[te]) and not set(tr) & set(te)
        oof[f == k] = fit_predict(tr, te)
        base[f == k] = ctrl[te] * np.median(y_all[tr] / ctrl[tr])
    M.append(mean_absolute_error(y, oof)); R.append(r2_score(y, oof)); B.append(mean_absolute_error(y, base))
    if oof0 is None: oof0 = oof
    print(f"  seed {r}: MAE {M[-1]:.1f} R2 {R[-1]:.3f}", flush=True)
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(g == k)[0] for k in pick])
    bs.append(mean_absolute_error(y[idx], oof0[idx]))
sub = df.iloc[ti]
print(f"RESULT {a.aid}{' DEV' if a.dev else ''}: n={len(y)} papers={sub['group'].nunique()} FR groups={len(ug)} features={Xa.shape[1]} "
      f"| MAE {np.mean(M):.1f} +- {np.std(M):.1f} CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] R2 {np.mean(R):.3f} "
      f"| baseline ctrl x median ratio MAE {np.mean(B):.1f}", flush=True)
