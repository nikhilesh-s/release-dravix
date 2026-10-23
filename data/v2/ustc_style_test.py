"""Amendment 10: USTC-style test -- epoxy only, GroupKFold by flame-retardant identity, pHRR MAE."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.insert(0, str(Path(__file__).parent))
import train_litcone_gt_round1 as T
T.TOPN, T.PHYS, T.CLIP = 100, True, True

df = pd.read_csv(T.SNAP, low_memory=False)
for c in T.NUM: df[c] = pd.to_numeric(df[c], errors="coerce")
cm = pd.read_csv(T.CLASS_MAP, dtype=str); cmap = dict(zip(cm["component_name"].str.strip().str.lower(), cm["class"]))
ppm = df.groupby("matrix_canonical")["group"].nunique()
X_all = T.build_X(df, cmap, sorted(ppm[ppm >= 3].index.difference(["other"])), sorted(df["material_form"].dropna().unique()))
m = ((df["matrix_canonical"] == "EP") & (df["phrr_kWm2"] > 0) & df["heat_flux_kWm2"].notna()).to_numpy()
sub, X = df[m].reset_index(drop=True), X_all[m].to_numpy(float)
y = sub["phrr_kWm2"].to_numpy(float)
def fr(comp):
    for name, role, v, u in T.parse_components(comp):
        if role in ("fr", "synergist") or "fr" in role: return T._norm(name)
    return "none"
g = np.array([fr(c) for c in sub["components"]])
print(f"EP pHRR rows {len(y)}, papers {sub['group'].nunique()}, FR groups {len(set(g))}")
maes, r2s, oof0 = [], [], None
for r in range(10):
    ug = np.unique(g); rng = np.random.default_rng(r); perm = rng.permutation(ug)
    fold = {k: i % 5 for i, k in enumerate(perm)}; f = np.array([fold[k] for k in g])
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        oof[te] = T.fit_pred(T.primary(), X[tr], y[tr], X[te])
    maes.append(mean_absolute_error(y, oof)); r2s.append(r2_score(y, oof))
    if r == 0: oof0 = oof
    print(f"  repeat {r}: MAE {maes[-1]:.1f}  R2 {r2s[-1]:.3f}", flush=True)
rng = np.random.default_rng(0); ug = np.unique(g); bs = []
for _ in range(2000):
    pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(g == k)[0] for k in pick])
    bs.append(mean_absolute_error(y[idx], oof0[idx]))
print(f"RESULT USTC-style (EP, grouped by FR): MAE {np.mean(maes):.1f} +- {np.std(maes):.1f} kW/m2, CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]; R2 {np.mean(r2s):.3f}; USTC 125.5 (unconfirmed)")
