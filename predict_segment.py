"""Predict a customer segment from annual income and spending score."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "customer_segmentation_model.joblib"


def predict_segment(income_k: float, spending_score: float, model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run python train_model.py first.")
    if income_k <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if not 1 <= spending_score <= 100:
        raise ValueError("Spending score must be between 1 and 100.")
    bundle = joblib.load(model_path)
    row = pd.DataFrame([[income_k, spending_score]], columns=bundle["feature_columns"])
    scaled = bundle["scaler"].transform(row)
    cluster_id = int(bundle["model"].predict(scaled)[0])
    centers = bundle["scaler"].inverse_transform(bundle["model"].cluster_centers_)
    center = centers[cluster_id]
    return {
        "cluster": cluster_id,
        "segment": bundle["cluster_to_segment"][cluster_id],
        "center_income_k": float(center[0]),
        "center_spending_score": float(center[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--income", type=float, required=True, help="Annual income in thousands of USD (k$)")
    parser.add_argument("--spending-score", type=float, required=True, help="Spending score from 1 to 100")
    args = parser.parse_args()
    result = predict_segment(args.income, args.spending_score)
    print(f"Cluster: {result['cluster']}")
    print(f"Customer segment: {result['segment']}")
    print(f"Segment center: income ${result['center_income_k']:.1f}k, spending score {result['center_spending_score']:.1f}")


if __name__ == "__main__":
    main()
