"""
cluster_mapper.py — Cognitive State Mapping & Interpretation Engine
===================================================================
Provides neurophysiologically grounded interpretation and cognitive state naming
for clusters discovered across the 10 multimodal brain and eye-tracking features.

Features (10):
    [EEI, FMT, ASM, GD, TRT, FFD, PupilSize, FixationCount, AlphaPower, BetaPower]

Cognitive States:
    1. Concentrated Fluent Reading (Category: Concentrated)
    2. Deep Cognitive Processing   (Category: Concentrated)
    3. Cognitive Fatigue           (Category: Fatigued)
    4. High Cognitive Workload     (Category: Workload)
    5. Comprehension Bottleneck    (Category: Bottleneck)
    -1. Unclassified Outlier Noise  (Category: Noise / Indeterminate)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


STANDARD_COGNITIVE_STATES = [
    {
        "name": "Concentrated Fluent Reading",
        "category": "Concentrated",
        "description": "High executive engagement (positive EEI) with rapid, fluent eye movements; efficient lexical and semantic parsing without cognitive struggle.",
    },
    {
        "name": "Deep Cognitive Processing",
        "category": "Concentrated",
        "description": "Sustained executive engagement (positive EEI) with deliberate, analytical reading pace and focused fixation duration; deep semantic integration.",
    },
    {
        "name": "Cognitive Fatigue",
        "category": "Fatigued",
        "description": "Depressed executive index (negative EEI), elevated alpha/theta band power, and autonomic pupil constriction indicating mental depletion and low vigilance.",
    },
    {
        "name": "High Cognitive Workload",
        "category": "Workload",
        "description": "Elevated frontal midline theta and dilated pupils; indicates intense working memory maintenance, high mental effort, and cognitive strain.",
    },
    {
        "name": "Comprehension Bottleneck",
        "category": "Bottleneck",
        "description": "Severely prolonged total reading times with high fixation counts and frequent saccadic regressions; lexical or syntactic comprehension breakdown.",
    }
]


def build_affinity_matrix(centroids: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Constructs a k x 5 affinity matrix between cluster centroids and the 5 standard
    cognitive reading states, enforcing physiological discriminators (e.g. pupil constriction
    and negative EEI for fatigue; positive EEI and low alpha for concentration).
    """
    k = centroids.shape[0]
    A = np.zeros((k, 5))

    for i in range(k):
        c = dict(zip(feature_names, centroids[i]))
        eei = c.get("EEI", 0.0)
        fmt = c.get("FMT", 0.0)
        trt = c.get("TRT", 0.0)
        gd = c.get("GD", 0.0)
        pupil = c.get("PupilSize", 0.0)
        fix = c.get("FixationCount", 0.0)
        alpha = c.get("AlphaPower", 0.0)
        beta = c.get("BetaPower", 0.0)

        # State 0: Concentrated Fluent Reading (EEI >= 0, swift eye movements, low alpha)
        if eei >= 0.0:
            A[i, 0] = (eei * 3.0) - (trt * 1.5) - (gd * 1.5) - (alpha * 1.0)
        else:
            A[i, 0] = -50.0 - abs(eei) * 10.0

        # State 1: Deep Cognitive Processing (EEI >= 0, deliberate reading pace, low alpha)
        if eei >= 0.0:
            A[i, 1] = (eei * 3.0) + (trt * 1.5) + (gd * 1.5) - (alpha * 0.5)
        else:
            A[i, 1] = -50.0 - abs(eei) * 10.0

        # State 2: Cognitive Fatigue (depressed EEI, constricted pupil, alpha/theta slowing)
        if eei <= 0.05 and pupil <= 0.15:
            A[i, 2] = (-eei * 3.0) - (pupil * 2.5) + (alpha * 0.8) + (fmt * 0.5)
        else:
            A[i, 2] = -50.0

        # State 3: High Cognitive Workload (high FMT, high pupil dilation, high beta/alpha)
        A[i, 3] = (fmt * 2.5) + (pupil * 2.5) + (beta * 1.5) + (alpha * 1.0)

        # State 4: Comprehension Bottleneck (prolonged TRT/GD and high fixations)
        A[i, 4] = (trt * 2.5) + (fix * 2.0) + (gd * 1.5) - (eei * 1.0)

    return A


def assign_cognitive_states(centroids: np.ndarray, feature_names: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Assigns unique, distinctive cognitive state names to each of the K clusters
    using Kuhn-Munkres (Hungarian) optimal bipartite matching.
    """
    k = centroids.shape[0]
    A = build_affinity_matrix(centroids, feature_names)

    # Use Hungarian algorithm to maximize total affinity
    row_ind, col_ind = linear_sum_assignment(-A)

    assigned_mapping: Dict[int, Dict[str, Any]] = {}
    for r, c in zip(row_ind, col_ind):
        state_info = STANDARD_COGNITIVE_STATES[c]
        assigned_mapping[int(r)] = {
            "name": state_info["name"],
            "category": state_info["category"],
            "description": state_info["description"],
            "affinity_score": round(float(A[r, c]), 3),
            "centroid": {fn: round(float(centroids[r, idx]), 3) for idx, fn in enumerate(feature_names)}
        }

    # Add special entry for noise / outlier (-1) if present (e.g. for HDBSCAN)
    assigned_mapping[-1] = {
        "name": "Unclassified Outlier / Noise",
        "category": "Noise",
        "description": "Atypical physiological readings that do not conform to coherent cognitive reading clusters.",
        "affinity_score": 0.0,
        "centroid": {}
    }

    return assigned_mapping


def compute_sentence_percentages(labels: np.ndarray, cluster_mapping: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the exact percentage of sentences assigned to each cognitive state,
    and sums up the broad categories: Concentrated (%), Fatigued (%), Workload (%), Bottleneck (%).
    """
    total = len(labels)
    if total == 0:
        return {}

    counts = pd.Series(labels).value_counts()
    percentages = {}
    
    concentrated_pct = 0.0
    fatigued_pct = 0.0
    workload_pct = 0.0
    bottleneck_pct = 0.0
    noise_pct = 0.0

    for cluster_id, count in counts.items():
        c_info = cluster_mapping.get(int(cluster_id), cluster_mapping.get(str(cluster_id), {}))
        c_name = c_info.get("name", f"Cluster {cluster_id}")
        c_cat = c_info.get("category", "Other")
        pct = round((count / total) * 100.0, 2)
        percentages[c_name] = pct

        if c_cat == "Concentrated":
            concentrated_pct += pct
        elif c_cat == "Fatigued":
            fatigued_pct += pct
        elif c_cat == "Workload":
            workload_pct += pct
        elif c_cat == "Bottleneck":
            bottleneck_pct += pct
        elif c_cat == "Noise":
            noise_pct += pct

    percentages["Total_Concentrated_Percentage"] = round(concentrated_pct, 2)
    percentages["Total_Fatigued_Percentage"] = round(fatigued_pct, 2)
    percentages["Total_Workload_Percentage"] = round(workload_pct, 2)
    percentages["Total_Bottleneck_Percentage"] = round(bottleneck_pct, 2)
    percentages["Total_Noise_Percentage"] = round(noise_pct, 2)

    return percentages
