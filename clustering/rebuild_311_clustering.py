"""Rebuild NYC 311 ZIP-code x complaint-type clustering.

Uses only filing-time volume and complaint-timing features. Outcome/closure
fields are deliberately excluded. Run from the project folder:
    python rebuild_311_clustering.py --input nyc311_clean.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

CUTOFF = "2026-10-25"
MIN_GROUP_COUNT = 20
FEATURES = ["log_volume", "weekend_share", "overnight", "morning", "afternoon", "evening"]
EXCLUDED = ["slow_48h", "closed_date", "hours_to_close", "status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="nyc311_clean.csv", help="Cleaned NYC 311 CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--replicates", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["incident_zip"] = pd.to_numeric(df["incident_zip"], errors="coerce").astype("Int64")
    train = df.loc[df.created_date < CUTOFF].dropna(
        subset=["created_date", "incident_zip", "complaint_type"]
    ).copy()

    train["daypart"] = pd.cut(
        train.created_date.dt.hour, [-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"]
    )
    train["weekend"] = (train.created_date.dt.dayofweek >= 5).astype(int)
    groups = train.groupby(["incident_zip", "complaint_type"], observed=True).agg(
        total_complaints=("unique_key", "size"),
        weekend_share=("weekend", "mean")
    ).reset_index()
    shares = pd.crosstab(
        [train.incident_zip, train.complaint_type], train.daypart, normalize="index"
    ).reset_index()
    groups = groups.merge(shares, on=["incident_zip", "complaint_type"], how="left")
    groups = groups.loc[groups.total_complaints >= MIN_GROUP_COUNT].copy().reset_index(drop=True)
    groups["log_volume"] = np.log1p(groups.total_complaints)

    scaler = StandardScaler()
    X = scaler.fit_transform(groups[FEATURES])

    comparison = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        comparison.append({
            "k": k, "method": "KMeans",
            "silhouette": float(silhouette_score(X, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_))
        })
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        comparison.append({
            "k": k, "method": "Agglomerative Ward",
            "silhouette": float(silhouette_score(X, ag.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X, ag.labels_))
        })

    # K-Means K=2 retained as the operational model because it supports
    # assigning future ZIP/type groups via predict(). Ward K=2 scores
    # marginally better on both internal separation metrics.
    model = KMeans(n_clusters=2, random_state=42, n_init=20).fit(X)
    groups["cluster_id"] = model.labels_

    # Bootstrap out-of-bag stability: fit each bootstrap model on an 80%
    # with-replacement sample, then compare labels ONLY on observations not
    # drawn in that replicate. This avoids evaluating on bootstrap training rows.
    rng = np.random.default_rng(42)
    aris = []
    n = len(X)
    for b in range(args.replicates):
        idx = rng.choice(n, int(0.8 * n), replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(idx)] = False
        if oob_mask.sum() < 10:
            continue
        boot_model = KMeans(n_clusters=2, random_state=100 + b, n_init=20).fit(X[idx])
        aris.append(float(adjusted_rand_score(
            model.labels_[oob_mask], boot_model.predict(X[oob_mask])
        )))

    profiles = {}
    for cid, part in groups.groupby("cluster_id"):
        profiles[str(int(cid))] = {
            "group_count": int(len(part)),
            "mean_complaints_per_zip_type_group": round(float(part.total_complaints.mean()), 2),
            "median_complaints_per_zip_type_group": round(float(part.total_complaints.median()), 2),
            "mean_weekend_share": round(float(part.weekend_share.mean()), 3),
            "mean_daypart_shares": {
                c: round(float(part[c].mean()), 3)
                for c in ["overnight", "morning", "afternoon", "evening"]
            },
            "top_complaint_types_by_volume": {
                str(k): int(v) for k, v in
                part.groupby("complaint_type").total_complaints.sum().sort_values(ascending=False).head(8).items()
            },
            "top_zip_codes_by_volume": {
                str(int(k)): int(v) for k, v in
                part.groupby("incident_zip").total_complaints.sum().sort_values(ascending=False).head(8).items()
            }
        }

    method = {
        "unit_of_analysis": "ZIP code x complaint type",
        "minimum_complaints_per_group": MIN_GROUP_COUNT,
        "features": FEATURES,
        "excluded_fields": EXCLUDED,
        "scaling": "StandardScaler",
        "algorithms_compared": ["KMeans", "Agglomerative Ward"],
        "k_values": [2, 3, 4, 5],
        "selected_model": "KMeans",
        "selected_k": 2,
        "selection_note": (
            "KMeans K=2 is retained as the operational model because it supports predict() "
            "for assigning future ZIP/type groups. Agglomerative Ward K=2 has marginally "
            "better internal separation scores (silhouette and Davies-Bouldin); this trade-off "
            "is disclosed rather than claiming KMeans is the metric winner."
        )
    }
    stability = {
        "method": "80% bootstrap with-replacement; ARI evaluated on out-of-bag groups only",
        "replicates": len(aris),
        "sample_fraction": 0.8,
        "mean_adjusted_rand_index": float(np.mean(aris)),
        "std_adjusted_rand_index": float(np.std(aris)),
        "p10": float(np.quantile(aris, 0.1)),
        "median": float(np.median(aris)),
        "p90": float(np.quantile(aris, 0.9))
    }
    result = {
        "project": "NYC 311 urban risk clustering",
        "data_file": input_path.name,
        "training_window": {
            "start": str(train.created_date.min()),
            "end": str(train.created_date.max()),
            "cutoff_exclusive": CUTOFF
        },
        "method": method,
        "data_counts": {
            "source_rows": int(len(df)),
            "training_rows_with_zip_and_type": int(len(train)),
            "zip_type_groups_before_minimum_support": int(
                train.groupby(["incident_zip", "complaint_type"], observed=True).ngroups
            ),
            "groups_after_minimum_support": int(len(groups))
        },
        "comparison": comparison,
        "bootstrap_stability": stability,
        "profiles": profiles,
        "caveats": [
            "Clusters describe ZIP/type complaint timing and volume patterns, not causal risk.",
            "Only ZIP/type groups with at least 20 training-period complaints are included.",
            "The data covers a short period and should not be interpreted as seasonal or long-term citywide patterns.",
            "slow_48h is not included in the feature matrix and was not used to fit clusters."
        ]
    }

    (out_dir / "clusters_311_rebuilt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    groups.to_csv(out_dir / "zip_type_cluster_assignments.csv", index=False)
    joblib.dump({
        "scaler": scaler,
        "model": model,
        "feature_columns": FEATURES,
        "training_cutoff_exclusive": CUTOFF,
        "minimum_group_count": MIN_GROUP_COUNT,
        "selection_note": method["selection_note"]
    }, out_dir / "cluster_model_311_rebuilt.joblib")
    print(json.dumps({
        "groups": len(groups),
        "selected_model": "KMeans K=2",
        "ward_k2_silhouette": comparison[1]["silhouette"],
        "kmeans_k2_silhouette": comparison[0]["silhouette"],
        "oob_bootstrap_mean_ari": stability["mean_adjusted_rand_index"],
        "oob_bootstrap_replicates": stability["replicates"]
    }, indent=2))

if __name__ == "__main__":
    main()
