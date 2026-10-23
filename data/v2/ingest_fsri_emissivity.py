#!/usr/bin/env python3
"""Dravix -- FSRI integrating-sphere (FTIR IS) ingestion: total emissivity (= absorptance, Kirchhoff) per material,
Planck-weighted at source temperatures 600-2000 K, plus the in-depth absorption coefficient where FSRI measured
transmission. Same source/citation as ingest_fsri.py. Declared in experiments/configs/data_pass_oct4.json.

Reproduces FSRI's own 02_Scripts/plot_IS_emissivity_data_html.py (repo commit a432697) step for step: cubic
reindex to 1 cm^-1, light-trap correction eps = 1 - (r_s - r_LT)/(1 - r_LT) - tau (tau from the thickest
transmission sample, else 0), 10 nm wavelength grid, ratio of trapezoid integrals against Planck's law; absorption
coefficient = least-squares slope of -ln(mean total transmittance) vs thickness. Only the per-wavelength Python loop
is vectorised (same arithmetic). Data only: thermal-property joins stay banned (experiments/banned_moves.md).

Run: python3 data/v2/ingest_fsri_emissivity.py   -> data/v2/emissivity_fsri.csv
"""
import csv
import glob
import math
import os
import re

import numpy as np
import pandas as pd
from scipy import integrate

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "fsri_materials_database", "01_Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emissivity_fsri.csv")
SOURCE = "fsri_materials_database"

h = 6.62607015e-34  # constants exactly as FSRI's script (c is their "speed of light in air")
c = 299702547
kB = 1.380649e-23
T_RANGE = np.linspace(600, 2000, 8)


def reindex_data(data, min_x, max_x, dx):  # verbatim from FSRI
    data = data.reindex(data.index.union(np.arange(min_x, max_x + dx, dx)))
    data = data.astype("float64")
    data.index = data.index.astype("float64")
    data = data.interpolate(method="cubic")
    return data.loc[np.arange(min_x, max_x + dx, dx)]


def planck(wl_m, t):
    return (2 * h * c ** 2) / (wl_m ** 5 * (np.exp((h * c) / (wl_m * kB * t)) - 1))


def read_dpt(f, name):
    d = pd.read_csv(f, index_col=0, header=None, names=["wavenumber", name])
    return reindex_data(d, math.ceil(min(d.index)), math.floor(max(d.index)), 1)


def slope(x, y):  # lsqfity "my" and "smy"
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    den = n * (x ** 2).sum() - x.sum() ** 2
    my = (n * (x * y).sum() - x.sum() * y.sum()) / den
    by = ((x ** 2).sum() * y.sum() - x.sum() * (x * y).sum()) / den
    s2 = ((y - by - my * x) ** 2).sum() / (n - 2) if n > 2 else float("nan")
    return my, math.sqrt(n * s2 / den) if n > 2 else float("nan")


def one_material(material):
    files = sorted(glob.glob(f"{RAW}/{material}/FTIR/IS/*.dpt"))
    trans_f = [f for f in files if "Trans" in f]
    ref_f = [f for f in files if "Trans" not in f]
    mean_trans, abs_coeff, d_max = None, {}, None
    if trans_f:
        ts, tt, thick = {}, {}, []
        for f in trans_f:
            rep = f.split("_")[-1].split(".dpt")[0]
            th = [p for p in f.split("_") if re.match(r"\d\.\d*", p)][0]
            thick.append(th)
            col = read_dpt(f, "transmittance")["transmittance"]
            if "_S_" in f:
                ts[f"{th}_{rep}"] = col
            elif "_T_" in f:
                tt[f"{th}_{rep}"] = col
        ts, tt = pd.DataFrame(ts), pd.DataFrame(tt)
        mean_trans = pd.DataFrame()
        th_list = sorted(set(thick))
        for d in th_list:
            mean_trans[d] = ts.filter(axis="columns", regex=d).mean(axis="columns") - \
                tt.filter(axis="columns", regex=d).mean(axis="columns")
        mean_trans["wavelength"] = 10000000 / mean_trans.index
        mean_trans = mean_trans.set_index("wavelength")
        mean_trans = reindex_data(mean_trans, math.ceil(min(mean_trans.index)), math.floor(max(mean_trans.index)) - 10, 10)
        wl = mean_trans.index.to_numpy() * 1e-9
        x_m = [float(d) / 1000 for d in th_list]
        for t in T_RANGE:
            sp = planck(wl, t)
            tau = [integrate.trapezoid(sp * mean_trans[d].to_numpy(), x=mean_trans.index) /
                   integrate.trapezoid(sp, x=mean_trans.index) for d in th_list]
            abs_coeff[t] = slope(x_m, [-np.log(np.abs(v)) for v in tau])
        d_max = max(thick)  # FSRI takes max() of the thickness strings

    sd, td = {}, {}
    for f in ref_f:
        rep = f.split(".dpt")[0].split("_")[-1]
        if "_S_" in f:
            sd[rep] = read_dpt(f, "reflectivity")["reflectivity"]
        elif "_T_" in f:
            td[rep] = read_dpt(f, "reflectivity")["reflectivity"]
    if not sd or not td:
        return None
    sd, td = pd.DataFrame(sd), pd.DataFrame(td)
    n_s, n_t = sd.shape[1], td.shape[1]
    mr = pd.DataFrame({"Sample": sd.mean(axis=1), "Sample_std": sd.std(axis=1),
                       "LT": td.mean(axis=1), "LT_std": td.std(axis=1)})
    mr["wavelength"] = 10000000 / mr.index
    mr = mr.set_index("wavelength")
    mr = reindex_data(mr, math.ceil(min(mr.index)), math.floor(max(mr.index)) - 10, 10)
    if mean_trans is not None:
        tr = pd.DataFrame({"transmittance": mean_trans[d_max]})
    else:
        tr = pd.DataFrame(index=mr.index, data={"transmittance": np.zeros(len(mr.index))})
    tr = reindex_data(tr, math.ceil(min(mr.index)), math.floor(max(mr.index)), 10)
    mr["transmittance"] = tr["transmittance"]
    mr["measured"] = 1 - ((mr["Sample"] - mr["LT"]) / (1 - mr["LT"])) - mr["transmittance"]
    c1 = np.sqrt(mr["Sample_std"] ** 2 + mr["LT_std"] ** 2) / (mr["Sample"] - mr["LT"])
    c2 = mr["LT_std"] / mr["LT"]
    mr["measured_std"] = ((mr["Sample"] - mr["LT"]) / (1 - mr["LT"])) * np.sqrt(c1 ** 2 + c2 ** 2)
    wl = mr.index.to_numpy() * 1e-9
    for t in T_RANGE:
        sp = planck(wl, t)
        mr[f"{t}_spectrum"] = sp
        mr[f"{t}_measured"] = sp * mr["measured"]
        mr[f"{t}_measured_std"] = sp * mr["measured_std"]
    mr = mr.dropna(axis="index")
    row = dict(material=material, n_sample_reps=n_s, n_trap_reps=n_t,
               transmission_measured="y" if trans_f else "n",
               trans_thicknesses_mm=";".join(sorted(set(thick))) if trans_f else "")
    for t in T_RANGE:
        den = integrate.trapezoid(mr[f"{t}_spectrum"], x=mr.index)
        row[f"emissivity_{int(t)}K"] = round(integrate.trapezoid(mr[f"{t}_measured"], x=mr.index) / den, 3)
        row[f"emissivity_std_{int(t)}K"] = round(integrate.trapezoid(mr[f"{t}_measured_std"], x=mr.index) / den, 3)
    for t in T_RANGE:
        row[f"abs_coeff_{int(t)}K_per_m"] = round(abs_coeff[t][0], 1) if t in abs_coeff else ""
    row["source"] = SOURCE
    row["source_files"] = f"01_Data/{material}/FTIR/IS/*.dpt"
    return row


def main():
    rows, skipped = [], []
    for material in sorted((m for m in os.listdir(RAW) if not m.startswith(".")), key=str.lower):
        if not os.path.isdir(f"{RAW}/{material}/FTIR/IS"):
            continue
        r = one_material(material)
        (rows.append(r) if r else skipped.append(material))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} materials ({sum(r['transmission_measured'] == 'y' for r in rows)} with transmission); "
          f"skipped (empty sample or light-trap files): {skipped}")


if __name__ == "__main__":
    main()
