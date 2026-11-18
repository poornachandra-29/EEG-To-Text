"""
modality_ablation.py — Evaluates GMM (K=6) across full multimodal (8 features),
EEG-only (3 features), and Eye-tracking-only (5 features) subsets.
Must run from repository root or uses repo-relative paths.
"""

import os
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

F = ["EEI", "FMT", "ASM", "GD", "TRT", "FFD", "PupilSize", "FixationCount"]
EEG = ["EEI", "FMT", "ASM"]
EYE = ["GD", "TRT", "FFD", "PupilSize", "FixationCount"]

# Resolve paths
medians_path = REPO_ROOT / "artifacts" / "feature_medians.joblib"
scaler_path = REPO_ROOT / "artifacts" / "robust_scaler.joblib"
train_path = REPO_ROOT / "artifacts" / "train_nrei_preprocessed.csv"
test_path = REPO_ROOT / "test_nrei_csv" / "test_predictions.csv"

med = pd.Series(joblib.load(medians_path))
sc = joblib.load(scaler_path)
tr = pd.read_csv(train_path)
te = pd.read_csv(test_path)

Xtr = pd.DataFrame(sc.transform(tr[F].fillna(med)), columns=F)
Xte = pd.DataFrame(sc.transform(te[F].fillna(med)), columns=F)

sets = {"Full (8 feat.)": F, "EEG-only (3)": EEG, "Eye-only (5)": EYE}
out = {}
models = {}

for name, cols in sets.items():
    m = GaussianMixture(6, covariance_type="full", n_init=10, reg_covar=1e-5, random_state=42).fit(Xtr[cols])
    models[name] = m
    ptr = m.predict_proba(Xtr[cols])
    pte = m.predict_proba(Xte[cols])
    ltr = ptr.argmax(1)
    lte = pte.argmax(1)
    seeds = []
    bics = []
    for s in range(10):
        ms = GaussianMixture(6, covariance_type="full", n_init=10, reg_covar=1e-5, random_state=s).fit(Xtr[cols])
        seeds.append(ARI(ltr, ms.predict(Xtr[cols])))
        bics.append(ms.bic(Xtr[cols]))

    out[name] = dict(
        bic=round(m.bic(Xtr[cols]), 1),
        sil_train=round(silhouette_score(Xtr[cols], ltr, sample_size=2000, random_state=0), 3),
        sil_test=round(silhouette_score(Xte[cols], lte), 3),
        ll_train=round(m.score(Xtr[cols]), 2),
        ll_test=round(m.score(Xte[cols]), 2),
        maxp_train=round(ptr.max(1).mean(), 3),
        maxp_test=round(pte.max(1).mean(), 3),
        unc_test=round((pte.max(1) < 0.55).mean(), 3),
        seed_ARI_mean=round(float(np.mean(seeds)), 3),
        seed_ARI_min=round(float(np.min(seeds)), 3),
        seeds_same=int(sum(a > 0.95 for a in seeds)),
        bic_min=round(min(bics), 1),
        bic_max=round(max(bics), 1),
        n_clusters_test=int(len(set(lte)))
    )
    out[name]["_lte"] = lte.tolist()
    out[name]["_ltr"] = ltr.tolist()

# Agreement with full model
for k in ["EEG-only (3)", "Eye-only (5)"]:
    out[k]["ARI_vs_full_train"] = round(ARI(out["Full (8 feat.)"]["_ltr"], out[k]["_ltr"]), 3)
    out[k]["ARI_vs_full_test"] = round(ARI(out["Full (8 feat.)"]["_lte"], out[k]["_lte"]), 3)

for v in out.values():
    v.pop("_lte")
    v.pop("_ltr")

print("\n=== MODALITY ABLATION RESULTS (GMM K=6) ===")
print(pd.DataFrame(out).T.to_string())

# Save results next to script and in current working dir
json.dump(out, open(SCRIPT_DIR / "ablation_results.json", "w"), indent=2)
json.dump(out, open("ablation_results.json", "w"), indent=2)
print(f"\n✓ Saved ablation results to {SCRIPT_DIR / 'ablation_results.json'} and ablation_results.json")
