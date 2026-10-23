#!/usr/bin/env python3
"""Dravix V2 -- final figures for the numbers of record (plan section 11).

For each target in experiments/numbers_of_record.json (pHRR exp074, TTI exp046, THR exp045, Char Residue exp081,
LOI exp092, FIGRA exp073) this writes, to reports/figures/v2_final/:
  <target>_pred_vs_actual.png   repeat-0 out-of-fold predictions vs measured, identity line
  <target>_residuals.png        repeat-0 OOF residual (predicted - measured) vs predicted
and, for pHRR and LOI, <target>_learning_curve.png (R2 of repeated GroupKFold 5x10 on random subsets of 25/50/75/100%
of the groups, 5 subset seeds, mean +/- sd).

Data and model come from each record's training script. Importing is used only where the import has no side effects:
  pHRR  -> train_phrr_promote_c2 (main() is guarded; module level only sets constants)   [imported]
  TTI   -> train_tti_v5_physics, imported with the sys.argv flags of the record (--onset-bc --replicate-table);
           module level only loads data, main is guarded                                  [imported]
  THR   -> train_thr_v1 (guarded) provides build_features/scalar_table/LIN; the record's wrapper
           train_thr_v1_replicates.py runs its whole evaluation at import, so its 10 loading lines are replicated
  CR    -> train_char_residue_round1b.py writes result JSONs and may sys.exit at import   [loading replicated verbatim]
  LOI   -> train_loi_round5.py (exp092, candidate c2) runs everything and writes JSONs at import [loading replicated verbatim]
  FIGRA -> train_figra_round1b.py rewrites a result JSON and reports/figra_same_polymer_table.md at import
                                                                                          [loading replicated verbatim]
Every target asserts the regenerated rkf_mean (mean over 10 repeats of the pooled-OOF R2) is within 0.002 of the record
before anything is plotted. Nothing under experiments/ is written.

Run: OMP_NUM_THREADS=1 python3 data/v2/make_figures.py [--no-lc] [--only KEY]
--only KEY (e.g. --only LOI) regenerates just that target's PNGs and updates only its entries in learning_curves.json and README.md.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import textwrap
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from eval_protocol import repeated_group_kfold

OUT = os.path.join(ROOT, "reports", "figures", "v2_final")
RECORDS = json.load(open(os.path.join(ROOT, "experiments", "numbers_of_record.json")))["targets"]
TOL = 0.002
BLUE, ORANGE, VERM, GREY = "#0072B2", "#E69F00", "#D55E00", "#666666"  # Okabe-Ito


# ----------------------------------------------------------------------------------------------------------------
# Loaders: each returns dict(X, y, groups, model, log_target, unit, source)
# ----------------------------------------------------------------------------------------------------------------
def load_phrr():
    import train_phrr_promote_c2 as P  # sets T.EXTRAS exactly as --bc does
    T = P.T
    full = T.load(replicate_target=True)
    df = full[full["has_mcc"]].reset_index(drop=True)
    grp = P.merged(df["material"].to_numpy())
    assert (len(df), len(np.unique(grp))) == (191, 64)
    return dict(X=df[P.FEATS].to_numpy(float), y=df[P.TARGET].to_numpy(float), groups=grp, model=T.hgb(),
                log_target=True, unit="kW/m$^2$", source="train_phrr_promote_c2.py (imported)")


def load_tti():
    argv = sys.argv
    sys.argv = [argv[0], "--onset-bc", "--replicate-table"]  # record's flags; module reads sys.argv at import
    try:
        import train_tti_v5_physics as V
    finally:
        sys.argv = argv
    name, model, feats, logt = [m for m in V.MODELS if m[0].startswith("C6")][0]
    d = V.df
    return dict(X=d[feats].to_numpy(float), y=d[V.TARGET].to_numpy(float), groups=d["material"].to_numpy(), model=model,
                log_target=logt, unit="s", source="train_tti_v5_physics.py --onset-bc --replicate-table, C6 (imported)")


def load_thr():
    from train_thr_v1 import LIN, build_features, scalar_table  # guarded; same imports as the record's wrapper
    # replicated verbatim from train_thr_v1_replicates.py (that file evaluates at import)
    r = pd.read_csv(f"{HERE}/cone_replicates_fsri.csv")
    r = r[~r["did_not_ignite"] & ~r["early_stop"]]
    thr = r.groupby(["material", "heat_flux_kWm2"], as_index=False).agg(
        total_heat_release_MJm2=("thr_MJm2", "mean"), n_reps=("thr_MJm2", "size"), n_trunc=("truncated_test", "sum"))
    df = build_features(thr, scalar_table())
    df["phys_MJm2"] = df.mpa_flux_kgm2 * df.hc_bc_kJg
    core = df.dropna(subset=["phys_MJm2"]).reset_index(drop=True)
    return dict(X=core[["phys_MJm2"]].to_numpy(float), y=core.total_heat_release_MJm2.to_numpy(float),
                groups=core.material.to_numpy(), model=LIN, log_target=False, unit="MJ/m$^2$",
                source="train_thr_v1_replicates.py (loading replicated; model/features imported from train_thr_v1)")


def load_cr():
    # verbatim from train_char_residue_round1b.py (--merge-groups), selected candidate c2 = HGB + 5 category flags
    cone = pd.read_csv(f"{HERE}/char_residue_cone_fsri.csv")
    sta = pd.read_csv(f"{HERE}/char_yield_sta.csv")
    mcc = pd.read_csv(f"{HERE}/mcc_hrr.csv")
    props = pd.read_csv(f"{HERE}/material_props_fsri.csv")
    sta_wide = (sta.pivot_table(index="material", columns="heating_rate_K_min", values="final_residual_mass_pct",
                                aggfunc="mean").rename(columns=lambda r: f"sta_res_{int(r)}K").reset_index())
    mcc_agg = mcc.groupby("material").agg(mcc_char_yield_pct=("char_yield_pct", "mean"),
                                          mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean")).reset_index()
    feat = (sta_wide.merge(mcc_agg, on="material", how="outer")
            .merge(props[["material", "density_kgm3", "categories"]], on="material", how="left"))
    df = cone[~cone.test_config.str.contains("noSpark") & cone.cone_residue_pct.notna()].merge(feat, on="material", how="left")
    STA_COLS = ["sta_res_3K", "sta_res_10K", "sta_res_30K"]
    df = df[df[STA_COLS].notna().any(axis=1) & df.mcc_char_yield_pct.notna()].reset_index(drop=True)
    y, groups = df.cone_residue_pct.values, df.material.values
    groups = np.where(groups == "PlasticC", "HIPS", groups)  # --merge-groups
    BASE = ["heat_flux_kWm2", "density_kgm3", "mcc_char_yield_pct"] + STA_COLS
    MINERAL_CATS = ["interior finish", "exterior finish", "insulation", "roofing", "structural"]
    cat_l = df.categories.fillna("").str.lower()
    FLAGS = []
    for c in MINERAL_CATS:
        col = "cat_" + c.replace(" ", "_")
        df[col] = cat_l.apply(lambda s, c=c: float(c in [t.strip() for t in s.split(";")]))
        FLAGS.append(col)
    HGB = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10,
                                        random_state=0)
    return dict(X=df[BASE + FLAGS].values, y=y, groups=groups, model=HGB, log_target=False, unit="%",
                source="train_char_residue_round1b.py --merge-groups, c2 (loading replicated verbatim)")


def load_loi():
    # verbatim from train_loi_snapshot_update.py / train_loi_round4.py / train_loi_round5.py (exp092 = round-5 c2; round-4 c2 = exp089 is its base): exp087's round-1 RF + 6 RDKit FR-structure features
    # + resolved_fr_wt_share, el_* recomputed with PubChem formulas (read from data/v2/fr_structures.csv, no network)
    import glob
    import re

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    from build_loi_rpuf import ELEMS, EXCLUDE_REFS, basis, iso_parts

    RDLogger.DisableLog("rdApp.*")
    CLS_PREFIX = "cls_"
    EL_COLS = ["el_P", "el_N", "el_Si", "el_B", "el_halogen", "el_metal"]
    BOOLS = ["has_EG", "has_APP", "reactive"]
    STRUCT_COLS = ["fr_arom_frac", "fr_fsp3", "fr_PO_per_kDa", "fr_PN_per_kDa", "fr_mollogp", "fr_tpsa_per_kDa"]

    df = pd.read_csv(f"{HERE}/loi_rpuf_literature.csv")
    cls_cols = [c for c in df.columns if c.startswith(CLS_PREFIX)]
    FEAT1 = ["total_fr_wtpct"] + cls_cols + EL_COLS + ["density_kgm3", "n_frs"] + BOOLS
    for b in BOOLS:
        df[b] = df[b].astype(str).str.lower().isin(["true", "1"]).astype(float)

    st = pd.read_csv(f"{HERE}/fr_structures.csv", dtype=str).fillna("")
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
    el_tab = pd.read_csv(f"{HERE}/fr_element_table.csv")

    def lookup_table(name):
        for _, e in el_tab.iterrows():
            if re.search(e["match"], name, flags=re.I):
                return e
        return None

    def row_features(r, use_pubchem):
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
        wts = []
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

    parts = sorted(glob.glob(f"{HERE}/loi_rpuf_parts/rows_w*.csv"))
    raw = pd.concat([pd.read_csv(p, dtype=str) for p in parts], ignore_index=True)
    raw = raw[~raw["ref_id"].isin(EXCLUDE_REFS)].reset_index(drop=True)  # as build_loi_rpuf.main()
    assert list(raw.row_id) == list(df.row_id)
    base = [row_features(r, False) for _, r in raw.iterrows()]
    newf = [row_features(r, True) for _, r in raw.iterrows()]
    for c in EL_COLS:
        a = np.array([b[c] for b in base], float)
        assert np.allclose(a, df[c].to_numpy(float), equal_nan=True), c
    df1 = df.copy()
    for c in EL_COLS:
        df1[c] = [n[c] for n in newf]

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
    assert bm.sum() == 460 and df[bm].ref_id.nunique() == 83 and len(FEAT2) == 34

    # ---- round 4 additions, verbatim from train_loi_round4.py (exp089 = candidate c2 "f6f7f8") ----
    # F6 base foam
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

    # F7 class priors
    mentions = []
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
    glob_prior = {x: float(PRI[x].median()) for x in ELEMS}

    def fill_row(r):
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
        df3[c] = F6[c].to_numpy()
    # ---- round 5 additions, verbatim from train_loi_round5.py (exp092 = candidate c2 "f6fixed_std_logdens") ----
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

    STD_RX = [("ASTM_D2863", re.compile(r"ASTM\s*D\s*2863", re.I)), ("GB_T_2406", re.compile(r"GB\s*/?\s*T\s*2406", re.I)),
              ("ISO_4589", re.compile(r"ISO\s*4589", re.I))]


    def std_label(s):
        s = str(s) if pd.notna(s) else ""
        hits = [(m.start(), k) for k, rx in STD_RX for m in [rx.search(s)] if m]
        return min(hits)[1] if hits else "other_unknown"


    std_lab = raw["loi_standard"].map(std_label)
    STDC = {f"std_{k}": (std_lab == k).astype(float) for k in ["ASTM_D2863", "GB_T_2406", "ISO_4589", "other_unknown"]}

    dC = df3.copy()
    for c in F6COLS5:
        dC[c] = F6f[c].to_numpy()
    for k, v in STDC.items():
        dC[k] = v.to_numpy()
    dC["log_density"] = np.log(dC.density_kgm3.astype(float))
    FEAT_C1 = FEAT2 + F6COLS5 + ["el_imputed_share"] + F8COLS
    FEAT_C2 = FEAT_C1 + list(STDC) + ["log_density"]

    d = dC[bm].reset_index(drop=True)
    assert d.ref_id.nunique() == 83 and len(FEAT_C2) == 51
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)
    return dict(X=d[FEAT_C2].astype(float).to_numpy(), y=d["loi_pct"].to_numpy(float), groups=d["ref_id"].to_numpy(),
                model=model, log_target=False, unit="% O$_2$ (LOI)",
                source="train_loi_round5.py, candidate c2 f6fixed_std_logdens (loading replicated verbatim)")


def load_figra():
    # verbatim from train_figra_round1b.py (row set A, F2, direct HGB, log1p target = exp012 config)
    reps = pd.read_csv(f"{HERE}/figra_fsri.csv")
    reps = reps[reps.spark_ignitor & reps.ignition_recorded]
    df = (reps.groupby(["material", "heat_flux_kWm2"])
          .agg(figra_kWm2s=("figra_kWm2s", "mean"), phrr_kWm2=("peak_hrrpua_kWm2", "mean"),
               t_phrr_s=("time_to_peak_s", "mean"), n_reps=("figra_kWm2s", "size")).reset_index())
    mcc = pd.read_csv(f"{HERE}/mcc_hrr.csv").groupby("material").agg(
        mcc_peak_hrr_Wg=("peak_specific_hrr_Wg", "mean"), mcc_char_yield_pct=("char_yield_pct", "mean"))
    extras = pd.read_csv(f"{HERE}/mcc_extras_fsri.csv").groupby("material").agg(
        mcc_hrc_JgK=("hrc_JgK", "mean"), mcc_total_hr_kJg=("total_hr_kJg", "mean"), mcc_t_peak_C=("t_peak_C", "mean"))
    props = pd.read_csv(f"{HERE}/material_props_fsri.csv").set_index("material")[
        ["categories", "density_kgm3", "cone_thickness_mm", "cone_mass_per_area_kgm2"]]
    df = df.join(mcc, on="material").join(extras, on="material").join(props, on="material")
    F2 = ["heat_flux_kWm2", "mcc_peak_hrr_Wg", "mcc_char_yield_pct", "density_kgm3", "cone_thickness_mm",
          "cone_mass_per_area_kgm2", "mcc_hrc_JgK", "mcc_total_hr_kJg", "mcc_t_peak_C"]
    d = df[df.mcc_peak_hrr_Wg.notna()].reset_index(drop=True)
    HGB = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5,
                                        random_state=0)
    return dict(X=d[F2].to_numpy(float), y=d.figra_kWm2s.to_numpy(), groups=d.material.to_numpy(), model=HGB,
                log_target=True, unit="kW/m$^2$/s", source="train_figra_round1b.py, exp012 config (loading replicated verbatim)")


TARGETS = [  # (record key, file stem, short axis name, loader, learning curve?)
    ("pHRR", "phrr", "Peak HRR", load_phrr, True),
    ("TTI", "tti", "Time to ignition", load_tti, False),
    ("THR", "thr", "Total heat release", load_thr, False),
    ("Char Residue", "char_residue", "Char residue", load_cr, False),
    ("LOI", "loi", "LOI", load_loi, True),
    ("FIGRA", "figra", "FIGRA", load_figra, False),
]


# ----------------------------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans"})


def header(key, rec, d, extra=""):
    ng = len(np.unique(d["groups"]))
    title = f"{key} ({rec['exp_id']}){extra}"
    sub = (f"record R$^2$ = {rec['value']:.3f}, 95% CI [{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]  |  "
           f"n = {len(d['y'])} rows / {ng} groups")
    status = "status: " + rec["status"]
    return title, sub, status


def decorate(fig, ax, title, sub, status, foot):
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left", y=0.985)
    ax.set_title(sub + "\n" + "\n".join(textwrap.wrap(status, 70)), fontsize=8.5, loc="left", color="#333333")
    fig.text(0.02, 0.012, "\n".join(textwrap.wrap(foot, 95)), fontsize=7, color=GREY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))


def plot_pred_actual(key, stem, name, rec, d, oof0, r2_0):
    y, u = d["y"], d["unit"]
    fig, ax = plt.subplots(figsize=(5.6, 5.9))
    ax.scatter(y, oof0, s=16, alpha=0.65, color=BLUE, edgecolor="white", linewidth=0.3)
    lo, hi = min(y.min(), oof0.min()), max(y.max(), oof0.max())
    pad = 0.04 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color=VERM, lw=1.2, label="identity (y = x)")
    ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
    ax.set_xlabel(f"Measured {name} ({u})"), ax.set_ylabel(f"Out-of-fold predicted {name} ({u})")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    t, s, st = header(key, rec, d, ": predicted vs actual")
    decorate(fig, ax, t, s, st, f"Repeat-0 OOF predictions (R$^2$ of this repeat = {r2_0:.3f}); record = mean over 10 repeats. "
                                f"Model/protocol: {rec['metric']}. Source: {d['source']}.")
    fig.savefig(os.path.join(OUT, f"{stem}_pred_vs_actual.png")), plt.close(fig)


def plot_resid(key, stem, name, rec, d, oof0, r2_0):
    y, u = d["y"], d["unit"]
    res = oof0 - y
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(oof0, res, s=16, alpha=0.65, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=VERM, lw=1.2, ls="--")
    ax.set_xlabel(f"Out-of-fold predicted {name} ({u})"), ax.set_ylabel(f"Residual, predicted $-$ measured ({u})")
    ax.text(0.98, 0.03, f"RMSE {np.sqrt(np.mean(res**2)):.3g}, mean bias {res.mean():+.3g}", transform=ax.transAxes,
            ha="right", fontsize=8, color="#333333")
    t, s, st = header(key, rec, d, ": residuals")
    decorate(fig, ax, t, s, st, f"Repeat-0 OOF predictions (R$^2$ of this repeat = {r2_0:.3f}). Source: {d['source']}.")
    fig.savefig(os.path.join(OUT, f"{stem}_residuals.png")), plt.close(fig)


def _lc_task(model, X, y, groups, log_target, frac, seed):
    if frac >= 1.0:
        idx = np.arange(len(y))
    else:
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        keep = rng.choice(ug, size=int(round(frac * len(ug))), replace=False)
        idx = np.flatnonzero(np.isin(groups, keep))
    m = clone(model)
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=1)
    r, _ = repeated_group_kfold(m, X[idx], y[idx], groups[idx], log_target=log_target)
    return frac, seed, float(r.mean()), len(idx), len(np.unique(groups[idx]))


def learning_curve(key, stem, rec, d, seeds=5, fracs=(0.25, 0.5, 0.75, 1.0)):
    tasks = [(f, s) for f in fracs if f < 1.0 for s in range(seeds)] + [(1.0, 0)]
    out = Parallel(n_jobs=6)(delayed(_lc_task)(d["model"], d["X"], d["y"], d["groups"], d["log_target"], f, s)
                             for f, s in tasks)
    res = {}
    for f in fracs:
        vals = [o[2] for o in out if o[0] == f]
        rows = [o[3] for o in out if o[0] == f]
        ngr = [o[4] for o in out if o[0] == f]
        res[f] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, vals=vals,
                      rows=float(np.mean(rows)), groups=float(np.mean(ngr)))
    fig, ax = plt.subplots(figsize=(6.0, 4.9))
    xs = [100 * f for f in fracs]
    for f in fracs:
        ax.scatter([100 * f] * len(res[f]["vals"]), res[f]["vals"], s=14, color=GREY, alpha=0.5, zorder=2)
    ax.errorbar(xs, [res[f]["mean"] for f in fracs], yerr=[res[f]["sd"] for f in fracs], marker="o", color=BLUE, lw=1.8,
                capsize=4, zorder=3, label="mean $\\pm$ sd over 5 subset seeds (100%: single full-data run)")
    ax.axhline(rec["value"], color=VERM, ls="--", lw=1, label=f"record R$^2$ = {rec['value']:.3f}")
    ax.set_xticks(xs), ax.set_xticklabels([f"{int(x)}%\n({res[f]['groups']:.0f} groups)" for x, f in zip(xs, fracs)])
    ax.set_xlabel("Share of groups used (random group subsets)"), ax.set_ylabel("R$^2$ (repeated GroupKFold 5x10, pooled OOF)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    t, s, st = header(key, rec, d, ": learning curve")
    decorate(fig, ax, t, s, st, "Each point: mean over 10 repeats of 5-fold GroupKFold on a random subset of groups (groups as in the record). "
                                "R$^2$ is also affected by the subset's target spread, not only by training size.")
    fig.savefig(os.path.join(OUT, f"{stem}_learning_curve.png")), plt.close(fig)
    return res


# ----------------------------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    do_lc = "--no-lc" not in sys.argv
    summary, lines, failed = {}, [], []
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for key, stem, name, loader, has_lc in TARGETS:
        if only and key != only:
            continue
        rec = RECORDS[key]
        d = loader()
        r, oof = repeated_group_kfold(d["model"], d["X"], d["y"], d["groups"], log_target=d["log_target"])
        diff = abs(r.mean() - rec["value"])
        n_rec = {"pHRR": (191, 64), "TTI": None, "THR": (191, 65), "Char Residue": (567, 61), "LOI": (460, 83),
                 "FIGRA": (197, 67)}[key]
        ng = len(np.unique(d["groups"]))
        print(f"[{key}] {len(d['y'])} rows / {ng} groups; regenerated rkf_mean {r.mean():.4f} vs record {rec['value']} "
              f"(diff {diff:.4f}); repeat-0 R2 {r[0]:.4f}")
        if diff > TOL or (n_rec and (len(d["y"]), ng) != n_rec):
            print(f"[{key}] NOT REPRODUCED (tol {TOL}); no figures written for this target")
            failed.append(key)
            continue
        plot_pred_actual(key, stem, name, rec, d, oof[0], r[0])
        plot_resid(key, stem, name, rec, d, oof[0], r[0])
        lines.append((key, stem, rec, d, len(d["y"]), ng, r.mean(), r[0], has_lc))
        if has_lc and do_lc:
            lc = learning_curve(key, stem, rec, d)
            summary[key] = lc
            print(f"[{key}] learning curve: " + "; ".join(
                f"{int(100 * f)}% {v['mean']:.3f} +/- {v['sd']:.3f}" for f, v in lc.items()))
    if do_lc:
        lcp = os.path.join(OUT, "learning_curves.json")
        merged = json.load(open(lcp)) if only and os.path.exists(lcp) else {}
        merged.update({k: {str(f): v for f, v in lc.items()} for k, lc in summary.items()})
        with open(lcp, "w") as fh:
            json.dump(merged, fh, indent=2)
    write_readme(lines, summary, failed, only)
    if failed:
        sys.exit(f"not reproduced: {failed}")


def write_readme(lines, lc, failed=(), only=None):
    md = ["# v2_final figures", "",
          "Generated by `data/v2/make_figures.py`. Predictions are repeat-0 out-of-fold predictions of the exact model and "
          "protocol behind each number of record (`experiments/numbers_of_record.json`); the regenerated mean R2 over the 10 "
          "repeats was asserted within 0.002 of the record before plotting. Titles carry the record's R2, CI, n and status.", ""]
    for key, stem, rec, d, n, ng, rm, r0, has_lc in lines:
        md.append(f"## {key} ({rec['exp_id']}), {n} rows / {ng} groups, status: {rec['status']}")
        md.append(f"- `{stem}_pred_vs_actual.png`: repeat-0 OOF predicted vs measured, identity line; regenerated mean R2 {rm:.4f} "
                  f"(record {rec['value']}), repeat-0 {r0:.3f}.")
        md.append(f"- `{stem}_residuals.png`: repeat-0 OOF residual (predicted minus measured) vs predicted, with RMSE and bias.")
        if has_lc:
            md.append(f"- `{stem}_learning_curve.png`: R2 at 25/50/75/100% of groups (random group subsets, 5 subset seeds, mean +/- sd)"
                      + (": " + ", ".join(f"{int(100 * f)}% {v['mean']:.3f}+/-{v['sd']:.3f}" for f, v in lc[key].items())
                         if key in lc else "."))
        md.append("")
    for k in failed:
        md.append(f"## {k}: NOT REPRODUCED within 0.002, no figures written\n")
    if only:  # keep every other section of the existing README, replace only this target's
        import re
        old = open(os.path.join(OUT, "README.md")).read()
        new = "\n".join(md[md.index(next(x for x in md if x.startswith("## " + only + " "))):]).split("\n\n")[0]
        old = re.sub(r"## " + re.escape(only) + r" \(.*?(?=\n\n## |\n\n`learning_curves)", lambda m: new.rstrip(), old, flags=re.S)
        open(os.path.join(OUT, "README.md"), "w").write(old)
        return
    md.append("`learning_curves.json` holds the per-seed learning-curve values. Colours: Okabe-Ito (colour-blind safe).")
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
