"""Train a reproducible K-means customer segmentation model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "Mall_Customers.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
REQUIRED_COLUMNS = ["CustomerID", "Gender", "Age", *FEATURES]
RANDOM_STATE = 42


def segment_name(income: float, spending: float) -> str:
    """Provide readable labels for the five segments in this mall dataset."""
    if income >= 75 and spending >= 65:
        return "High-income, high-spending"
    if income >= 75 and spending < 35:
        return "High-income, low-spending"
    if income <= 40 and spending >= 65:
        return "Lower-income, high-spending"
    if income <= 40 and spending < 35:
        return "Lower-income, low-spending"
    return "Mid-range, moderate-spending"


def save_cluster_plot(df: pd.DataFrame, output: Path) -> None:
    segments = list(df["Segment"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis", max(1, len(segments)))
    for i, name in enumerate(segments):
        part = df[df["Segment"] == name]
        ax.scatter(part[FEATURES[0]], part[FEATURES[1]], s=52, alpha=.82,
                   label=name, color=cmap(i), edgecolor="white", linewidth=.35)
    ax.set_title("Customer Segments by Annual Income and Spending Score", fontsize=14, pad=14)
    ax.set_xlabel("Annual income (thousand USD)")
    ax.set_ylabel("Spending score (1–100)")
    ax.grid(True, alpha=.22)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def run(data_path: Path, output_dir: Path, k_min: int = 2, k_max: int = 10) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    df = df.copy()
    if df[FEATURES].isna().any().any():
        raise ValueError("Annual income or spending score has missing values. Clean these rows before training.")
    if df.duplicated(subset=["CustomerID"]).any():
        raise ValueError("CustomerID must be unique for this dataset.")
    if not 2 <= k_min <= k_max < len(df):
        raise ValueError("Require 2 <= k_min <= k_max < number of customers.")

    feature_frame = df[FEATURES].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    k_rows: list[dict[str, float | int]] = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        k_rows.append({"k": k, "inertia": float(model.inertia_),
                       "silhouette_score": float(silhouette_score(scaled, labels))})
    score_df = pd.DataFrame(k_rows)
    best_row = score_df.loc[score_df["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])

    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_rows = []
    cluster_to_segment: dict[int, str] = {}
    for cluster_id, (income, spending) in enumerate(centers):
        name = segment_name(float(income), float(spending))
        cluster_to_segment[int(cluster_id)] = name
        center_rows.append({"Cluster": int(cluster_id), "Segment": name,
                            "Annual Income (k$)": round(float(income), 2),
                            "Spending Score (1-100)": round(float(spending), 2)})

    clustered = df.copy()
    clustered["Cluster"] = labels.astype(int)
    clustered["Segment"] = clustered["Cluster"].map(cluster_to_segment)
    clustered.to_csv(output_dir / "clustered_customers.csv", index=False)

    profiles = clustered.groupby(["Cluster", "Segment"], as_index=False).agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Female_Percent=("Gender", lambda s: 100 * (s.astype(str).str.lower() == "female").mean()),
        Average_Income_kUSD=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean"),
    )
    profiles["Average_Age"] = profiles["Average_Age"].round(1)
    profiles["Female_Percent"] = profiles["Female_Percent"].round(1)
    profiles["Average_Income_kUSD"] = profiles["Average_Income_kUSD"].round(2)
    profiles["Average_Spending_Score"] = profiles["Average_Spending_Score"].round(2)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    score_df.to_csv(output_dir / "cluster_selection_scores.csv", index=False)
    centers_df = pd.DataFrame(center_rows).sort_values("Cluster")
    centers_df.to_csv(output_dir / "cluster_centers.csv", index=False)

    # Elbow curve
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(score_df["k"], score_df["inertia"], marker="o", linewidth=2, color="#2878b5")
    ax.axvline(best_k, color="#e07a5f", linestyle="--", label=f"Selected k={best_k} (best silhouette)")
    ax.set(title="Elbow Method: Inertia by Number of Clusters", xlabel="Number of clusters (k)", ylabel="Inertia")
    ax.grid(True, alpha=.25); ax.legend(frameon=False)
    fig.savefig(output_dir / "elbow_method.png", dpi=170, facecolor="white"); plt.close(fig)

    # Silhouette comparison
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(score_df["k"], score_df["silhouette_score"], marker="o", linewidth=2, color="#2a9d8f")
    ax.scatter([best_k], [float(best_row["silhouette_score"])], s=100, color="#e76f51", zorder=4,
               label=f"Selected k={best_k}: {float(best_row['silhouette_score']):.3f}")
    ax.set(title="Silhouette Score by Number of Clusters", xlabel="Number of clusters (k)", ylabel="Silhouette score", ylim=(0, 1))
    ax.grid(True, alpha=.25); ax.legend(frameon=False)
    fig.savefig(output_dir / "silhouette_scores.png", dpi=170, facecolor="white"); plt.close(fig)

    save_cluster_plot(clustered, output_dir / "customer_segments.png")

    # Segment sizes chart
    segment_counts = clustered["Segment"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    segment_counts.plot(kind="barh", ax=ax, color="#3a86a8")
    ax.set(title="Customers per Segment", xlabel="Number of customers", ylabel="")
    ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
    fig.savefig(output_dir / "segment_sizes.png", dpi=170, facecolor="white"); plt.close(fig)

    # Feature distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].hist(df[FEATURES[0]], bins=14, color="#3a86a8", edgecolor="white")
    axes[0].set(title="Annual Income Distribution", xlabel="Annual income (k$)", ylabel="Customers")
    axes[1].hist(df[FEATURES[1]], bins=14, color="#2a9d8f", edgecolor="white")
    axes[1].set(title="Spending Score Distribution", xlabel="Spending score (1–100)", ylabel="Customers")
    for ax in axes: ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(output_dir / "feature_distributions.png", dpi=170, facecolor="white"); plt.close(fig)

    bundle = {
        "scaler": scaler,
        "model": kmeans,
        "feature_columns": FEATURES,
        "cluster_to_segment": cluster_to_segment,
        "segment_centers": centers_df.to_dict(orient="records"),
        "selected_k": best_k,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, output_dir / "customer_segmentation_model.joblib")
    metrics = {
        "dataset_file": data_path.name,
        "customer_rows": int(len(df)),
        "columns": list(df.columns),
        "features_used_for_clustering": FEATURES,
        "features_not_used_for_clustering": ["Gender", "Age", "CustomerID"],
        "missing_values_in_clustering_features": int(df[FEATURES].isna().sum().sum()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "selected_k": best_k,
        "silhouette_score": float(best_row["silhouette_score"]),
        "inertia": float(kmeans.inertia_),
        "random_state": RANDOM_STATE,
        "k_search": k_rows,
        "segment_counts": {str(k): int(v) for k, v in clustered["Segment"].value_counts().items()},
        "segment_centers": center_rows,
        "note": "K-means clusters annual income and spending score after standardization. Gender, age, and CustomerID are retained for descriptive profiling only.",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Customer segmentation training completed.")
    print(f"Dataset: {data_path} ({len(df)} customers)")
    print(f"Selected clusters: k={best_k}")
    print(f"Silhouette score: {float(best_row['silhouette_score']):.4f}")
    print(f"Inertia: {float(kmeans.inertia_):.4f}")
    print("Segment profiles:")
    print(profiles.to_string(index=False))
    print(f"Artifacts saved to: {output_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Mall_Customers.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for model and reports")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k to evaluate (default: 2)")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum k to evaluate (default: 10)")
    args = parser.parse_args()
    run(args.data, args.output_dir, args.k_min, args.k_max)


if __name__ == "__main__":
    main()
