"""Derive Pomázi-style composition inputs for reconstructed rows that lack them.

Pomázi & Toldy 2023 Table 1 tabulates, per matrix, C/H/O/N/P % and aliphatic/cycloaliphatic/aromatic %
(normalised over C+H+O+N+P). Within a matrix series these are exactly linear in the P loading, i.e. a
mass-balance mixture: comp = (1 - w) * base + w * fr, with w = P% / P_fr.  We recover each FR's pure-component
vector `fr` from tabulated rows, validate the rule by predicting every single-FR tabulated row from FR vectors
fitted on OTHER matrices, then fill rows whose matrix base is tabulated:
  - composites: inputs of their matrix (fibre content stays a separate input, as in Pomázi 2026);
  - matrices/composites at loadings not tabulated: mixture rule.
Rows outside the four core systems (CE blends, ionic liquids, gelcoats, EG/ZnO, DOPO adducts, hemp) stay NaN
and are tagged core=False.

  python3 data/v2/derive_pomazi_inputs.py            # validate + write data/v2/pomazi_recon_inputs.csv
"""
import os
import re

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
COMP = ["C_pct", "H_pct", "O_pct", "N_pct", "P_pct", "aliphatic_pct", "cycloaliphatic_pct", "aromatic_pct"]
# P mass fraction (%) of each FR, normalised over C+H+O+N+P (all of these contain only those elements).
# APP (NH4PO3)n 31.93; RDP C30H24O8P2 10.78. Cross-checked below against the tabulated slopes.
P_FR = {"APP": 31.93, "RDP": 10.78}
CORE = {"PER/T58", "GER/T58", "SPE/T58", "GFTE/DETDA"}


def system(row):
    m, h = str(row["matrix_resin"]).strip(), str(row["hardener"])
    if re.search(r"T58|MH 3122", h) and m in ("PER", "GER", "SPE"):
        return f"{m}/T58"
    if m == "GFTE" and "DETDA" in h:
        return "GFTE/DETDA"
    return None


def fr_loads(row):
    """{'APP': P%, 'RDP': P%} from fr1/fr2 when the unit is P content of the matrix; None if not expressible."""
    out = {}
    for k in ("fr1", "fr2"):
        name, load, unit = row[f"{k}_name"], row[f"{k}_loading"], str(row[f"{k}_unit"])
        if pd.isna(name) or str(name).strip() in ("", "none"):
            continue
        name = str(name).strip()
        if name not in P_FR or pd.isna(load) or "P" not in unit.split("(")[0]:
            return None
        out[name] = out.get(name, 0.0) + float(load)
    return out


def mix(base, fr_vecs, loads):
    comp = np.array(base, float).copy()
    w_tot = sum(p / P_FR[f] for f, p in loads.items())
    comp = comp * (1 - w_tot)
    for f, p in loads.items():
        comp = comp + (p / P_FR[f]) * fr_vecs[f]
    return comp


def main():
    d = pd.read_csv(f"{D}/pomazi_recon.csv")
    d["system"] = d.apply(system, axis=1)
    tab = d[d["inputs_source"].eq("tabulated_pz2023")].copy()
    tab["loads"] = tab.apply(fr_loads, axis=1)

    bases = {s: g.loc[g["loads"].apply(lambda x: x == {}), COMP].iloc[0].to_numpy(float)
             for s, g in tab[tab["system"].isin(CORE)].groupby("system")}
    singles = tab[tab["system"].isin(CORE) & tab["loads"].apply(lambda x: x is not None and len(x) == 1)]

    def fit_fr(exclude_system=None):
        vecs = {}
        for f in P_FR:
            rows = singles[singles["loads"].apply(lambda x: f in x) & (singles["system"] != exclude_system)]
            est = []
            for _, r in rows.iterrows():
                b = bases[r["system"]]
                est.append(b + (r[COMP].to_numpy(float) - b) * P_FR[f] / r["loads"][f])
            vecs[f] = np.mean(est, axis=0)
        return vecs

    vec_all = fit_fr()
    print("FR pure-component vectors (C H O N P ali cyc aro):")
    for f, v in vec_all.items():
        print(f"  {f}: {np.round(v, 2)}")

    # Validation: every tabulated core row with expressible loads, FR vectors fitted without its own system.
    errs = []
    for _, r in tab[tab["system"].isin(CORE) & tab["loads"].notna()].iterrows():
        if r["loads"] == {}:
            continue
        pred = mix(bases[r["system"]], fit_fr(exclude_system=r["system"]), r["loads"])
        errs.append(np.abs(pred - r[COMP].to_numpy(float)))
    errs = np.array(errs)
    print(f"Leave-system-out validation on {len(errs)} tabulated rows: max abs error per input "
          f"{dict(zip(COMP, np.round(errs.max(axis=0), 2)))}; overall mean {errs.mean():.3f}")
    assert errs.max() < 1.5, "mixture rule does not reproduce Table 1 well enough; do not use derived inputs"

    # Fill
    d["loads"] = d.apply(fr_loads, axis=1)
    d["core"] = d["system"].isin(CORE) & d["loads"].notna()
    filled = 0
    for i, r in d[d["core"] & ~d["inputs_source"].eq("tabulated_pz2023")].iterrows():
        d.loc[i, COMP] = mix(bases[r["system"]], vec_all, r["loads"])
        d.loc[i, "inputs_source"] = "derived_mixture_rule"
        filled += 1
    # Pomázi 2026's other four inputs: epoxy / hardener / FR1 / FR2 mass ratios (% of matrix).
    # FR mass% = P% / P_FR * 100. Hardener share of the resin+hardener part from N balance (only the amine
    # hardener carries N): h = N_base / N_hardener, with T58 C15H30N2 (N 11.75 %) and DETDA C11H18N2 (N 15.71 %).
    # Check: PER/T58 gives 3.36 / 11.75 = 0.286 = 100:40, the ratio stated in Pomázi 2023 and Polymers 2018.
    n_hard = {"PER/T58": 11.75, "GER/T58": 11.75, "SPE/T58": 11.75, "GFTE/DETDA": 15.71}
    h_sys = {s: bases[s][COMP.index("N_pct")] / n_hard[s] for s in bases}
    print("Hardener mass share from N balance:", {s: round(h, 3) for s, h in h_sys.items()})
    for i, r in d[d["core"]].iterrows():
        frs = [(f, p / P_FR[f] * 100) for f, p in r["loads"].items()]
        fr_mass = sum(m for _, m in frs)
        name1, name2 = str(r["fr1_name"]).strip(), str(r["fr2_name"]).strip()
        d.loc[i, "fr1_ratio_pct"] = dict(frs).get(name1, 0.0)
        d.loc[i, "fr2_ratio_pct"] = dict(frs).get(name2, 0.0)
        d.loc[i, "hardener_ratio_pct"] = (100 - fr_mass) * h_sys[r["system"]]
        d.loc[i, "epoxy_ratio_pct"] = (100 - fr_mass) * (1 - h_sys[r["system"]])
    d["fibre_wtpct_input"] = np.where(d["sample_type"].eq("matrix"), 0.0, d["fibre_wtpct"])
    d["has_all4"] = d[["tti_s", "phrr_kWm2", "thr_MJm2", "residue_pct"]].notna().all(axis=1)
    d["has_inputs"] = d[COMP].notna().all(axis=1)
    d = d.drop(columns=["loads"])
    d.to_csv(f"{D}/pomazi_recon_inputs.csv", index=False)
    use = d[d["has_inputs"] & d["has_all4"] & d["dup_of"].isna()]
    print(f"Derived inputs for {filled} rows. Usable (inputs + all four outputs, not dup): {len(use)} "
          f"({use['sample_type'].value_counts().to_dict()}; heat flux {use['heat_flux_kWm2'].value_counts().to_dict()})")
    nonp = d[d["has_all4"] & d["dup_of"].isna() & ~d["has_inputs"]]
    print(f"Not usable for lack of inputs: {len(nonp)} (outside the four core systems or loading not in P%)")


if __name__ == "__main__":
    main()
