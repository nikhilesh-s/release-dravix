#!/usr/bin/env python3
"""Component NAME -> additive class (litcone_gt_round1 features). Reads only the `components` column of the staged
rows (names and roles; no target values). Rules are fixed regexes, first match wins; unmatched -> 'other'.
Writes data/v2/cone_lit_parts/component_class_map.csv (component_name, role_seen, class, rule)."""
import csv, glob, re
from collections import Counter
from pathlib import Path
P = Path(__file__).resolve().parent
RULES = [  # order matters
    ("APP_IFR", r"ammonium polyphosphate|\bapp\b|intumescent|\bifr\b|pentaerythritol|\bper\b|melamine polyphosphate|\bmpp\b|piperazine pyrophosphate|\bppap\b"),
    ("EG", r"expand(able|ed) graphite|\beg\b"),
    ("halogen_FR", r"brom|chlor(?!ide)|decabrom|\bdbdpo\b|\bdbdpe\b|tbbpa|hexabromo|\btcpp\b|\btcep\b|\bfluor"),
    ("Sb2O3", r"sb2o3|antimony"),
    ("metal_hydroxide", r"alumin(i)?um (tri)?hydroxide|\bath\b|magnesium hydroxide|\bmdh\b|\bmh\b|al\(oh\)3|mg\(oh\)2|boehmite|brucite"),
    ("boron", r"bor(ate|on|ic)|\bzb\b|zinc borate|\bbn\b"),
    ("nanoclay", r"montmorillonite|\bmmt\b|\bomt\b|clay|halloysite|\bhnt|sepiolite|kaolin|vermiculite|attapulgite|bentonite"),
    ("metal_oxide_LDH", r"\bldh\b|layered double|hydrotalcite|\bmof\b|zif|oxide|tio2|zno|fe2o3|fe3o4|cuo|mno2|ceo2|zro2|ferrite|\bmxene\b|ti3c2"),
    ("carbon_nano", r"graphene|\bgo\b|\brgo\b|carbon nanotube|\bcnt|mwcnt|swcnt|carbon black|fullerene|carbon nano"),
    ("Si", r"silic|siloxane|\bposs\b|silane|silica|sio2|\bkh-?\d"),
    ("P_FR", r"phosph|dopo|phyt|\bbdp\b|\brdp\b|\btpp\b|\bdmmp\b|\bdemp\b|alpi|aluminum diethyl|aluminium diethyl|hypophos|phytic|\bpa\b phytic"),
    ("N_FR", r"melamine|cyanur|\bmca\b|\bmc\b|urea|guanid|triazin|nitrogen"),
    ("glass_fibre", r"glass fib|\bgf\b|glass fabric|e-glass|basalt"),
    ("carbon_fibre", r"carbon fib|\bcf\b"),
    ("natural_fibre", r"wood|flax|kenaf|jute|hemp|ramie|bamboo|sisal|cellulose|lignin|chitosan|starch|rice husk|bagasse|coir|natural fib"),
    ("mineral_filler", r"talc|calcium carbonate|caco3|mica|wollastonite|zeolite|diatom|fly ash|glass bead|sepiolite|gypsum|dolomite|perlite|eggshell|seashell|barium sulfate"),
]
names = Counter(); roles = {}
for f in glob.glob(str(P / "oa_staging/rows_oa_*.csv")) + glob.glob(str(P / "lib_staging/rows_lib_*.csv")) + [str(P / "cone_lit_merged_raw.csv")]:
    for r in csv.DictReader(open(f)):
        comp = r.get("components") or ((r.get("fr_summary") or "") + "|FR|")
        for part in comp.split(";"):
            b = [x.strip() for x in part.split("|")]
            if b and b[0]:
                k = b[0].lower(); names[k] += 1; roles.setdefault(k, set()).add(b[1].lower() if len(b) > 1 else "")
with open(P / "component_class_map.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["component_name", "role_seen", "class", "rule", "n"])
    for k, n in names.most_common():
        cls, rule = "other", ""
        for c, rx in RULES:
            if re.search(rx, k):
                cls, rule = c, rx; break
        rs = roles[k]
        if cls == "other" and rs & {"matrix", "curing_agent"} and not rs & {"fr", "synergist", "filler", "reinforcement", "nanofiller"}:
            cls, rule = "not_additive", "role matrix/curing_agent"
        elif cls == "other" and rs & {"fr", "synergist"}:
            cls, rule = "other_FR", "role fallback"
        elif cls == "other" and rs & {"filler", "reinforcement", "nanofiller"}:
            cls, rule = "other_filler", "role fallback"
        if re.fullmatch(r"none.*|no fr.*|-", k):
            cls, rule = "not_additive", "none"
        w.writerow([k, ";".join(sorted(rs)), cls, rule[:40], n])
print(len(names), "component names")
