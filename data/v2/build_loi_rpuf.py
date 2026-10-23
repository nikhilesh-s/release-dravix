"""Merge LOI RPUF extraction parts into data/v2/loi_rpuf_literature.csv with normalized loadings and
FR-derived element features (plan §3 B1; config experiments/configs/loi_round1.json).

Loading normalization (per FR):
  - foam basis ("wt%", "%", "wt% of all raw materials", "g per 100 g (wt%)", "wt% (FR ratio)"): used as is.
  - polyol basis ("php", "phr", "parts by weight", "g/100 g polyol", "g (per 100 g polyol blend)",
    "wt% of polyol blend", bare "g" with a 100 g polyol batch): wt% of foam = L / (100 + iso + 5 + sum FR) * 100,
    iso = isocyanate parts parsed from the paper's isocyanate field, else 130 (typical pMDI for RPUF; flagged).
    The +5 covers water, catalyst, surfactant.
  - "wt% P": FR mass = P / P-fraction of the FR (needs the element table), foam basis.
Element features: foam-level P, N, Si, B, halogen, metal wt% = sum over FRs (wt% x mass fraction) from
data/v2/fr_element_table.csv; any FR with a loading but no table entry -> that row's element features are NaN.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
ELEMS = ["P", "N", "Si", "B", "halogen", "metal"]
CLASSES = ["P-additive", "P-reactive", "N-additive", "P-N", "intumescent(APP-based)", "expandable_graphite",
           "metal_hydroxide", "metal_oxide/salt", "silicon", "boron", "halogen", "nanoclay",
           "carbon_nano(graphene/CNT)", "bio-based", "other"]


# Population exclusions (decided by the orchestrator before any wave-3 scoring; plan §3 B1 population =
# flame-retarded RIGID PU / PIR foams). N019: open-cell spray foam (not rigid closed-cell).
EXCLUDE_REFS = {"N019": "open-cell spray foam, outside the rigid-foam population"}


def basis(unit):
    u = str(unit).lower().strip()
    if u in ("", "nan"):
        return None
    if "wt% p" in u or u.startswith("p wt"):
        return "P"
    if re.search(r"php|phr|parts|per 100|/100 ?g|of polyol", u) or u == "g":
        return "polyol"
    if "%" in u:
        return "foam"
    return None


def iso_parts(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:parts|phr|php|pbw|g\b)", str(s))
    return (float(m.group(1)), False) if m else (130.0, True)


def main():
    parts = sorted(glob.glob(f"{D}/loi_rpuf_parts/rows_w*.csv"))
    d = pd.concat([pd.read_csv(p, dtype=str) for p in parts], ignore_index=True)
    d = d[~d["ref_id"].isin(EXCLUDE_REFS)].reset_index(drop=True)
    el = pd.read_csv(f"{D}/fr_element_table.csv")
    man = pd.read_csv(f"{D}/../sources/sunfu_corpus_manifest.csv", dtype=str).set_index("ref_id")

    def lookup(name):
        for _, e in el.iterrows():
            if re.search(e["match"], name, flags=re.I):
                return e
        return None

    out = []
    for _, r in d.iterrows():
        rec = dict(row_id=r["row_id"], ref_id=r["ref_id"], source_doi=r["source_doi"],
                   in_sunfu_refs=man.loc[r["ref_id"], "in_sunfu_refs"] if r["ref_id"] in man.index else "",
                   extraction=r["extraction"], loi_pct=pd.to_numeric(r["loi_pct"], errors="coerce"),
                   density_kgm3=pd.to_numeric(r["density_kgm3"], errors="coerce"))
        frs = []
        for k in ("fr1", "fr2", "fr3"):
            name = str(r.get(f"{k}_name", "") or "").strip()
            if name.lower() in ("", "nan", "none"):
                continue
            frs.append((name, str(r.get(f"{k}_class", "") or "other").strip(),
                        pd.to_numeric(r.get(f"{k}_loading"), errors="coerce"), r.get(f"{k}_unit")))
        iso, iso_est = iso_parts(r.get("isocyanate"))
        polyol_fr_sum = sum(L for _, _, L, u in frs if basis(u) == "polyol" and pd.notna(L))
        elem = dict.fromkeys(ELEMS, 0.0)
        cls = dict.fromkeys(CLASSES, 0.0)
        ok_load, ok_elem, flags = True, True, []
        for name, c, L, u in frs:
            b, e = basis(u), lookup(name)
            if pd.isna(L) or b is None:
                ok_load = False
                flags.append(f"unparsed loading {name}: {L} {u}")
                continue
            if b == "foam":
                w = L
            elif b == "polyol":
                w = L / (100 + iso + 5 + polyol_fr_sum) * 100
                flags.append("iso_default_130" if iso_est else "iso_from_paper")
            else:  # wt% P
                if e is None or not e["P"]:
                    ok_load = False
                    flags.append(f"P-basis without P fraction: {name}")
                    continue
                w = L / e["P"] * 100
            cls[c if c in cls else "other"] += w
            if e is None:
                ok_elem = False
            else:
                for x in ELEMS:
                    elem[x] += w * float(e[x]) / 100
        rec.update(n_frs=len(frs), total_fr_wtpct=sum(cls.values()) if ok_load else np.nan,
                   loading_ok=ok_load, elements_ok=ok_elem, flags="; ".join(sorted(set(flags))))
        rec.update({f"cls_{c}": (v if ok_load else np.nan) for c, v in cls.items()})
        rec.update({f"el_{x}": (v if ok_load and ok_elem else np.nan) for x, v in elem.items()})
        rec["has_EG"] = any(c == "expandable_graphite" for _, c, _, _ in frs)
        rec["has_APP"] = any(re.search(r"APP|ammonium polyphosph", n, re.I) for n, _, _, _ in frs)
        rec["reactive"] = any(c == "P-reactive" for _, c, _, _ in frs)
        rec["fr_names"] = " | ".join(n for n, _, _, _ in frs)
        out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(f"{D}/loi_rpuf_literature.csv", index=False)
    bench = o[o["loi_pct"].notna() & (o["extraction"] != "figure")]
    print(f"merged {len(o)} rows / {o['ref_id'].nunique()} papers from {len(parts)} parts")
    print(f"benchmark of record (LOI present, not figure-read): {len(bench)} rows / {bench['ref_id'].nunique()} papers")
    print(f"  loading parsed: {bench['loading_ok'].mean():.0%}; element features complete: "
          f"{(bench['loading_ok'] & bench['elements_ok']).mean():.0%}; iso default used on "
          f"{bench['flags'].str.contains('iso_default').sum()} rows")
    unk = {}
    for n in o.loc[~o["elements_ok"], "fr_names"]:
        for x in n.split(" | "):
            if lookup(x) is None:
                unk[x] = unk.get(x, 0) + 1
    print(f"FR names without element entry ({len(unk)}):", sorted(unk.items(), key=lambda t: -t[1])[:40])


if __name__ == "__main__":
    main()
