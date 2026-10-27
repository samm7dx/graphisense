"""Metrics loading service for model evaluation results."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

RF_METRICS_CSV = ROOT / "outputs" / "reports" / "model_evaluation_metrics.csv"
CNN_EVAL_JSON = ROOT / "outputs" / "reports" / "cnn_evaluation" / "evaluation_results.json"


def load_rf_metrics() -> Optional[pd.DataFrame]:
    """Load Random Forest cross-validation metrics from CSV."""
    try:
        return pd.read_csv(RF_METRICS_CSV)
    except Exception:
        return None


def load_rf_metrics_dict() -> Optional[Dict[str, Any]]:
    """Load Random Forest metrics as a dictionary for the Random Forest (Set A - Normalized 9) configuration."""
    df = load_rf_metrics()
    if df is None or df.empty:
        return None
    # Get the Random Forest (Set A - Normalized 9) row
    rf_row = df[(df["model"] == "random_forest") & (df["feature_set"] == "Set A (Normalized 9)")]
    if rf_row.empty:
        # Fallback to first random_forest row
        rf_row = df[df["model"] == "random_forest"]
        if rf_row.empty:
            return None
    return rf_row.iloc[0].to_dict()


def load_cnn_evaluation() -> Optional[Dict[str, Any]]:
    """Load CNN held-out test evaluation results from JSON."""
    try:
        with open(CNN_EVAL_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_rf_confusion_matrix() -> Optional[List[List[int]]]:
    """Load the out-of-fold confusion matrix for Random Forest (Set A - Normalized 9).

    Note: The CSV doesn't store confusion matrices directly.
    We could load from the saved figure or compute from OOF predictions if available.
    For now, return a placeholder structure or load from a saved JSON if it exists.
    """
    # Try to load from a saved JSON file if it exists
    cm_path = ROOT / "outputs" / "reports" / "rf_confusion_matrix.json"
    try:
        import json
        with open(cm_path, "r") as f:
            return json.load(f)
    except Exception:
        pass

    # Fallback: return None if not available
    return None


def load_rf_metrics_for_comparison() -> Optional[Dict[str, Any]]:
    """Load RF metrics formatted for the comparison page."""
    metrics = load_rf_metrics_dict()
    if metrics is None:
        return None

    # The CSV has mean/std columns, format for comparison
    return {
        "accuracy_mean": metrics.get("accuracy_mean", 0),
        "accuracy_std": metrics.get("accuracy_std", 0),
        "balanced_accuracy_mean": metrics.get("balanced_accuracy_mean", 0),
        "balanced_accuracy_std": metrics.get("balanced_accuracy_std", 0),
        "f1_score_mean": metrics.get("f1_score_mean", 0),
        "f1_score_std": metrics.get("f1_score_std", 0),
        "roc_auc_mean": metrics.get("roc_auc_mean", 0),
        "roc_auc_std": metrics.get("roc_auc_std", 0),
        "precision_pd_mean": metrics.get("precision_mean", 0),  # macro precision
        "precision_lpd_mean": metrics.get("precision_mean", 0),
        "recall_pd_mean": metrics.get("recall_pd_mean", 0),
        "recall_lpd_mean": metrics.get("recall_lpd_mean", 0),
        "f1_pd_mean": metrics.get("f1_score_mean", 0),
        "f1_lpd_mean": metrics.get("f1_score_mean", 0),
    }


def load_cnn_metrics_for_comparison() -> Optional[Dict[str, Any]]:
    """Load CNN metrics formatted for the comparison page."""
    eval_data = load_cnn_evaluation()
    if eval_data is None:
        return None

    test_metrics = eval_data.get("test_metrics", {})
    if not test_metrics:
        return None

    return {
        "test_metrics": {
            "accuracy": test_metrics.get("accuracy", 0),
            "balanced_accuracy": test_metrics.get("balanced_accuracy", 0),
            "f1_pd": test_metrics.get("f1_pd", 0),
            "f1_lpd": test_metrics.get("f1_lpd", 0),
            "roc_auc": test_metrics.get("roc_auc", 0),
            "precision_pd": test_metrics.get("precision_pd", 0),
            "precision_lpd": test_metrics.get("precision_lpd", 0),
            "recall_pd": test_metrics.get("recall_pd", 0),
            "recall_lpd": test_metrics.get("recall_lpd", 0),
        },
        "test_confusion_matrix": test_metrics.get("confusion_matrix", []),
    }