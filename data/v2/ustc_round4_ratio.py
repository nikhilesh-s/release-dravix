"""Amendment 13: reduction-ratio model on the USTC-style test (predict log(pHRR/control), multiply back)."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "ustc_round2_frchem.py").read().split('print("amendment 11')[0])
ctrl = np.expm1(T.control_feature(df, "phrr_kWm2"))
has = ~np.isnan(ctrl) & (df["is_control"] != 1).to_numpy()
def run2(mask, label):
    sub = df[mask].reset_index(drop=True); X = X4[mask].to_numpy(float); y = sub["phrr_kWm2"].to_numpy(float); c = ctrl[mask]
    t = np.log(y / c); g = sub["components"].map(fr_group).to_numpy(); ug = np.unique(g)
    M, B, R, oof0 = [], [], [], None
    for r in range(10):
        perm = np.random.default_rng(r).permutation(ug); fold = {k: i % 5 for i, k in enumerate(perm)}
        f = np.array([fold[k] for k in g]); oof = np.zeros(len(y)); base = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            m = clone(T.primary()).fit(X[tr], t[tr])
            oof[te] = c[te] * np.exp(np.clip(m.predict(X[te]), t[tr].min(), t[tr].max()))
            base[te] = c[te] * np.exp(np.median(t[tr]))
        M.append(mean_absolute_error(y, oof)); B.append(mean_absolute_error(y, base)); R.append(r2_score(y, oof))
        if r == 0: oof0 = oof
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(g == k)[0] for k in pick]); bs.append(mean_absolute_error(y[idx], oof0[idx]))
    print(f"{label}: n={len(y)} papers={sub['group'].nunique()} FR groups={len(ug)} | ratio-model MAE {np.mean(M):.1f} +- {np.std(M):.1f} "
          f"CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] R2 {np.mean(R):.3f} | baseline ctrl x median ratio MAE {np.mean(B):.1f}", flush=True)
print("amendment 13 -- given the paper's measured neat-epoxy pHRR; USTC 125.5 (unconfirmed, different inputs)")
run2(ep & has & pP, "HEADLINE B' (EP + P-FR, with control)")
run2(ep & has, "  A' (all EP with control)")
