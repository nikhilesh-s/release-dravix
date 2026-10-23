#!/usr/bin/env python3
"""LOI round 3 structure resolution (experiments/configs/loi_round3.json): FR name -> PubChem CID/SMILES/formula.

Query order per FR name: expansion inside the parentheses, then the abbreviation (text before the
parentheses), then the full string. First query that returns a compound is used; acceptance rules below.
No structure is guessed by hand. <= 5 requests/s. Output: data/v2/fr_structures.csv
Run: OMP_NUM_THREADS=1 python3 data/v2/resolve_fr_structures.py | tee experiments/runs/resolve_fr_structures.log
"""
import glob, json, re, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = Path(__file__).resolve().parent
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,MolecularFormula/JSON"

REJECT_NAME = [
    (r"poly|\b(MF|M)?APP\b|\bMPP\b", "polymer / polyol / polyphosphate (APP, MPP repeat units already in element table)"),
    (r"graphite|graphene|graph[a-z]* oxide|\bGO\b|CNT|MWCNT|MOF", "graphite/graphene/CNT/MOF"),
    (r"clay|cloisite|bentonite|halloysite|\bHNT\b|vermiculite|perlite|montmorillonite|hydrotalcite", "clay/mineral"),
    (r"slag|glass|aerogel|silica|SiO2|\bPOSS\b", "mineral/silica/glass"),
    (r"alginate|chitosan|cellulose|lignin|lignosulfonate|pulp|shell|walnut|eucalyptus|oil|starch|canola|tannin", "bio-material"),
    (r"resin|formaldehyde", "resin/mixture"),
    (r"capsule", "encapsulated product"),
    (r"modified|-based|grafted|coated|impregnated", "modified/based/grafted/coated product"),
    (r"\bfiller\b|\bpowder\b", "filler/powder"),
]
REJECT_CLASS = {"expandable_graphite", "carbon_nano(graphene/CNT)", "nanoclay", "bio-based"}  # FR class assigned at extraction
KEYWORD_ELEMENT = [(r"phosph", "P"), (r"chlor", "Cl"), (r"brom", "Br"), (r"alumin", "Al"), (r"magnes", "Mg"), (r"zinc", "Zn"), (r"boro|boric|borate", "B"), (r"silic|silox|silan", "Si")]  # expansion text must be consistent with the hit's formula
NONMETAL = set("H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe Ge".split())


def parts_of(name):
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return name.strip(), ""


def fetch(q):
    u = URL.format(urllib.parse.quote(q, safe=""))
    time.sleep(0.25)
    try:
        with urllib.request.urlopen(u, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:  # SSL etc -> curl fallback
        p = subprocess.run(["curl", "-s", "-m", "30", "-w", "\n%{http_code}", u], capture_output=True, text=True)
        body, code = p.stdout.rsplit("\n", 1)
        if code.strip() == "404":
            return None
        return json.loads(body)


def first_prop(js):
    try:
        p = js["PropertyTable"]["Properties"][0]
        smi = p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES") or ""
        return str(p["CID"]), smi, p.get("MolecularFormula", "")
    except Exception:
        return None


def main():
    d = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(glob.glob(str(D / "loi_rpuf_parts/rows_w*.csv")))])
    names = {}
    for k in ("fr1", "fr2", "fr3"):
        for n in d[f"{k}_name"].dropna():
            n = n.strip()
            if n.lower() not in ("", "nan", "none"):
                names[n] = names.get(n, 0) + 1
    cls_of = {}
    for k in ("fr1", "fr2", "fr3"):
        for n, c in zip(d[f"{k}_name"], d[f"{k}_class"]):
            if isinstance(n, str) and isinstance(c, str):
                cls_of.setdefault(n.strip(), set()).add(c.strip())
    out = []
    for name in sorted(names):
        abbr, exp = parts_of(name)
        rec = dict(fr_name=name, query_used="", cid="", smiles="", formula="", status="", reason="")
        why = next((w for pat, w in REJECT_NAME if re.search(pat, name, flags=re.I)), None)
        if not why and cls_of.get(name, set()) & REJECT_CLASS:
            why = "FR class " + ",".join(sorted(cls_of[name] & REJECT_CLASS)) + " (graphite/clay/bio-material)"
        if why:
            rec.update(status="rejected", reason="name: " + why)
            out.append(rec)
            continue
        qs = []
        for q in (exp, abbr, name):
            if q and q not in qs:
                qs.append(q)
        hit = None
        for q in qs:
            js = fetch(q)
            if js and first_prop(js):
                hit = (q,) + first_prop(js)
                break
        if not hit:
            rec.update(status="not_found", reason="no PubChem match for: " + " || ".join(qs))
            out.append(rec)
            continue
        q, cid, smi, formula = hit
        rec.update(query_used=q, cid=cid, smiles=smi, formula=formula)
        mol = Chem.MolFromSmiles(smi) if smi else None
        fe = set(re.findall(r"[A-Z][a-z]?", formula))
        bad_kw = [el for pat, el in KEYWORD_ELEMENT if re.search(pat, name, flags=re.I) and el not in fe]  # whole name, not just the expansion
        if mol is not None and mol.GetNumAtoms() < 2:
            rec.update(status="rejected", reason="single-atom hit (element symbol, not a compound)")
        elif bad_kw:
            rec.update(status="rejected", reason=f"hit formula lacks {bad_kw} implied by the expansion text")
        elif "." in smi:
            rec.update(status="rejected", reason="multi-component SMILES (salt/mixture)")
        elif mol is None:
            rec.update(status="rejected", reason="SMILES not parseable by RDKit")
        else:
            syms = {a.GetSymbol() for a in mol.GetAtoms()}
            if "C" not in syms and (syms - NONMETAL):
                rec.update(status="rejected", reason="inorganic metal compound (mineral-like)")
            else:
                rec.update(status="accepted", reason="single-component small molecule")
        out.append(rec)
        print(f"  [{rec['status']:9}] {name!r} <- {q!r} CID {cid} {formula}", flush=True)
    o = pd.DataFrame(out)
    o.to_csv(D / "fr_structures.csv", index=False)
    print(f"\n{len(o)} distinct FR names: " + ", ".join(f"{k} {v}" for k, v in o.status.value_counts().items()))
    print("\n=== ACCEPTED (audit list): fr_name | query | CID | formula | SMILES ===")
    for _, r in o[o.status == "accepted"].iterrows():
        print(f"{r.fr_name} | {r.query_used} | {r.cid} | {r.formula} | {r.smiles}")
    print("\n=== REJECTED ===")
    for _, r in o[o.status == "rejected"].iterrows():
        print(f"{r.fr_name} | {r.reason}" + (f" | CID {r.cid} {r.formula}" if r.cid else ""))
    print("\n=== NOT FOUND ===")
    for _, r in o[o.status == "not_found"].iterrows():
        print(f"{r.fr_name}")


if __name__ == "__main__":
    main()
