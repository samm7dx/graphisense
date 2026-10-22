"""Tests for machine learning pipelines and evaluation procedures."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models import build_pipeline as package_build_pipeline
from src.models.inference import load_saved_model
from src.models.train_evaluate import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    build_pipeline,
    evaluate_model_cv,
    load_feature_dataset,
    run_all_evaluations,
    train_and_save_final_model,
)

CSV_PATH = Path("outputs/features/features.csv")
REPORT_CSV = Path("outputs/reports/model_evaluation_metrics.csv")
REPORT_MD = Path("outputs/reports/model_evaluation_report.md")
FIG_DIR = Path("outputs/figures")


@pytest.fixture
def sample_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """Fixture returning features DataFrame and binary target array."""
    if not CSV_PATH.is_file():
        pytest.skip("Generated feature dataset is not available")
    df = pd.read_csv(CSV_PATH)
    y = np.where(df["class_label"] == "PD", 1, 0)
    return df, y


def test_build_pipeline_types():
    """Verify Pipeline construction and invalid model handling."""
    pipe_lr = build_pipeline("logistic_regression")
    assert isinstance(pipe_lr, Pipeline)
    assert "scaler" in pipe_lr.named_steps
    assert "clf" in pipe_lr.named_steps

    pipe_rf = build_pipeline("random_forest")
    assert isinstance(pipe_rf, Pipeline)
    assert "clf" in pipe_rf.named_steps
    assert package_build_pipeline is build_pipeline

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline("unsupported_classifier")


def test_load_feature_dataset_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Run 'python -m src.features.build_feature_dataset'"):
        load_feature_dataset(tmp_path / "missing.csv")


def test_training_saves_inference_artifacts(tmp_path: Path):
    """Training on a temporary synthetic feature table saves the inference contract."""
    values = np.arange(10 * len(FEATURE_SET_B), dtype=float).reshape(10, len(FEATURE_SET_B))
    feature_table = pd.DataFrame(values, columns=FEATURE_SET_B)
    feature_table["class_label"] = ["LPD"] * 5 + ["PD"] * 5
    feature_table["status"] = "SUCCESS"
    csv_path = tmp_path / "features.csv"
    feature_table.to_csv(csv_path, index=False)

    output_dir = tmp_path / "models"
    model_path, metadata_path = train_and_save_final_model(
        csv_path=csv_path,
        output_dir=output_dir,
    )

    assert model_path == output_dir / "model_pipeline.joblib"
    assert metadata_path == output_dir / "model_metadata.json"
    pipeline, metadata = load_saved_model(model_path, metadata_path)
    assert hasattr(pipeline, "predict")
    assert metadata["feature_names"] == FEATURE_SET_A


def test_feature_sets_validity(sample_dataset):
    """Verify predefined feature sets exist in CSV and adhere to constraints."""
    df, _ = sample_dataset

    assert len(FEATURE_SET_A) == 9
    assert len(FEATURE_SET_B) == 25

    # Feature Set A must strictly exclude raw crop dimensions
    assert "image_width" not in FEATURE_SET_A
    assert "image_height" not in FEATURE_SET_A
    assert "foreground_pixels" not in FEATURE_SET_A

    # All features in both sets must exist in features.csv
    for f in FEATURE_SET_A:
        assert f in df.columns, f"Feature {f} missing from CSV"
    for f in FEATURE_SET_B:
        assert f in df.columns, f"Feature {f} missing from CSV"


def test_stratified_split_integrity(sample_dataset):
    """Verify that cross-validation folds are stratified, disjoint, and cover all samples."""
    df, y = sample_dataset
    X = df[FEATURE_SET_A]

    res = evaluate_model_cv(X, y, "logistic_regression", n_splits=5, random_state=42)

    fold_metrics = res["fold_metrics"]
    assert len(fold_metrics) == 5

    oof_preds = res["oof_predictions"]
    assert len(oof_preds) == len(y)

    cm = res["confusion_matrix"]
    assert cm.sum() == len(y)  # All 249 predictions accounted for


def test_evaluation_reproducibility(sample_dataset):
    """Verify exact reproducibility given the same random seed."""
    df, y = sample_dataset
    X = df[FEATURE_SET_A]

    res1 = evaluate_model_cv(X, y, "logistic_regression", n_splits=5, random_state=42)
    res2 = evaluate_model_cv(X, y, "logistic_regression", n_splits=5, random_state=42)

    assert np.array_equal(res1["oof_predictions"], res2["oof_predictions"])
    assert np.allclose(res1["oof_probabilities"], res2["oof_probabilities"])
    assert res1["mean_metrics"] == res2["mean_metrics"]


def test_metric_bounds(sample_dataset):
    """Verify all calculated classification metrics fall within theoretical [0, 1] bounds."""
    df, y = sample_dataset
    X = df[FEATURE_SET_A]

    res = evaluate_model_cv(X, y, "logistic_regression", n_splits=5, random_state=42)
    for fold in res["fold_metrics"]:
        for metric_name in ["accuracy", "balanced_accuracy", "precision", "recall_pd", "recall_lpd", "f1_score", "roc_auc"]:
            val = fold[metric_name]
            assert 0.0 <= val <= 1.0, f"{metric_name} out of bounds: {val}"


def test_model_artifacts_created():
    """Verify that all Phase 4 report and figure deliverables exist and are non-empty."""
    if not REPORT_CSV.exists() or not REPORT_MD.exists() or not FIG_DIR.exists():
        pytest.skip("Generated model evaluation artifacts are not available")
    assert REPORT_CSV.exists()
    assert REPORT_CSV.stat().st_size > 100

    assert REPORT_MD.exists()
    assert REPORT_MD.stat().st_size > 500

    expected_figures = [
        "confusion_matrices.png",
        "roc_curves.png",
        "model_comparison_metrics.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIG_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is empty"
