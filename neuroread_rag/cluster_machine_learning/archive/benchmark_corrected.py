"""
benchmark_corrected.py — Corrected LOPO & Holdout clustering evaluation with train-fitted scaler,
chance level baseline, and cross-task (NR vs TSR) classification ablation.
"""

import os
import glob
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score as ARI,
    normalized_mutual_info_score as NMI,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_P = str(REPO_ROOT / "neuroread_rag" / "data" / "precomputed_csvs")
P = os.environ.get("PRECOMPUTED_DIR", DEFAULT_P)
if not P.endswith("/"):
    P += "/"

F = ["EEI", "FMT", "ASM", "GD", "TRT", "FFD", "PupilSize", "FixationCount"]
EEG = F[:3]
EYE = F[3:]


def load(t):
    fs = sorted(g for g in glob.glob(P + t + "/features_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["pid"] = d.patient_id.astype(str).str.extract(r"results(Z[A-Z]+)")[0]
    return d[~d[F].isna().all(axis=1)].reset_index(drop=True)


nr = load("task2_NR")
tsr = load("task3_TSR")
print("NR", len(nr), nr.pid.nunique(), "TSR", len(tsr), tsr.pid.nunique())


def hung(y, p):
    ly, lp = np.unique(y), np.unique(p)
    M = np.zeros((len(ly), len(lp)), int)
    for i, a in enumerate(ly):
        for j, b in enumerate(lp):
            M[i, j] = ((y == a) & (p == b)).sum()
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y)


rng = np.random.default_rng(0)


def chance(y, p, n=100):
    return float(np.mean([hung(y, rng.permutation(p)) for _ in range(n)]))


def models():
    return {
        "GMM-full": GaussianMixture(6, covariance_type="full", n_init=10, reg_covar=1e-5, random_state=42),
        "GMM-tied": GaussianMixture(6, covariance_type="tied", n_init=10, reg_covar=1e-5, random_state=42),
        "GMM-diag": GaussianMixture(6, covariance_type="diag", n_init=10, reg_covar=1e-5, random_state=42),
        "K-Means": KMeans(6, n_init=10, random_state=42)
    }


def fold(tr, te):
    med = tr[F].median()
    sc = RobustScaler().fit(tr[F].fillna(med))
    Xtr = sc.transform(tr[F].fillna(med))
    Xte = sc.transform(te[F].fillna(med))
    y = te.cognitive_pattern.astype(str).to_numpy()
    out = {}
    maj = pd.Series(y).value_counts(normalize=True).max()
    for n, m in models().items():
        m.fit(Xtr)
        p = m.predict(Xte)
        out[n] = dict(
            sil=silhouette_score(Xte, p) if len(set(p)) > 1 else np.nan,
            db=davies_bouldin_score(Xte, p) if len(set(p)) > 1 else np.nan,
            ari=ARI(y, p),
            nmi=NMI(y, p),
            acc=hung(y, p),
            chance=chance(y, p),
            maj=maj
        )
    return out


res = {}
r = fold(nr[nr.pid != "ZPH"], nr[nr.pid == "ZPH"])
res["holdout_ZPH"] = r
print("\nHOLDOUT ZPH (train-fitted scaler)")
print(pd.DataFrame(r).T.round(3).to_string())

rows = []
for p in sorted(nr.pid.unique()):
    for n, v in fold(nr[nr.pid != p], nr[nr.pid == p]).items():
        rows.append(dict(model=n, pid=p, **v))
L = pd.DataFrame(rows)
L["acc_minus_chance"] = L.acc - L.chance
S = L.groupby("model")[["sil", "db", "ari", "nmi", "acc", "chance", "maj", "acc_minus_chance"]].agg(["mean", "std"]).round(3)
print("\nLOPO 12 folds mean/std")
print(S.to_string())
res["lopo"] = json.loads(L.to_json(orient="records"))
print("\nwins vs chance (acc>chance) per model:", (L.acc_minus_chance > 0).groupby(L.model).sum().to_dict())

# NR vs TSR, 12 participants
a = nr[F + ["pid"]].copy()
a["y"] = 0
b = tsr[F + ["pid"]].copy()
b["y"] = 1
D = pd.concat([a, b], ignore_index=True)


def prep(g):
    g = g.copy()
    for c in F:
        m = g[c].median()
        i = g[c].quantile(.75) - g[c].quantile(.25)
        g[c] = ((g[c] - m) / (i if i > 0 else 1)).fillna(0)
    return g


D = pd.concat([prep(g) for _, g in D.groupby("pid")], ignore_index=True)
R = {}
for fs, cols in {"EEG-only (3)": EEG, "Eye-only (5)": EYE, "All 8": F}.items():
    for mn, mk in {
        "LogReg": lambda: LogisticRegression(max_iter=1000),
        "RandForest": lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1)
    }.items():
        au = []
        for p in sorted(D.pid.unique()):
            t = D[D.pid != p]
            v = D[D.pid == p]
            au.append(roc_auc_score(v.y, mk().fit(t[cols], t.y).predict_proba(v[cols])[:, 1]))
        R[f"{fs} | {mn}"] = (round(np.mean(au), 3), round(np.std(au, ddof=1), 3), round(min(au), 3))

print("\nNR vs TSR, LOPO 12 participants, AUC mean/sd/min")
print(pd.DataFrame(R, index=["mean", "sd", "min"]).T.to_string())
res["nr_tsr"] = R

# Save JSON next to script and in current working dir
json.dump(res, open(SCRIPT_DIR / "final_eval_results.json", "w"), default=float, indent=2)
json.dump(res, open("final_eval_results.json", "w"), default=float, indent=2)
print(f"\n✓ Saved results to {SCRIPT_DIR / 'final_eval_results.json'} and final_eval_results.json")
