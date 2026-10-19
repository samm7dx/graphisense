"""Tests for exploratory data analysis functions."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.analyze_features import (
    FEATURE_COLUMNS,
    compute_class_comparisons,
    compute_distribution_summary,
    find_highly_correlated_pairs,
    load_and_validate_dataset,
)

CSV_PATH = Path("outputs/features/features.csv")
REPORT_PATH = Path("outputs/reports/eda_report.md")
FIG_DIR = Path("outputs/figures")


@pytest.fixture
def dataset_df() -> pd.DataFrame:
    """Fixture returning the validated dataset DataFrame."""
    return load_and_validate_dataset(CSV_PATH)


def test_load_and_validate_dataset_success(dataset_df: pd.DataFrame):
    """Verify loading and validating the actual features CSV."""
    assert len(dataset_df) == 249
    for col in FEATURE_COLUMNS:
        assert col in dataset_df.columns
    assert dataset_df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_load_and_validate_dataset_errors(tmp_path: Path):
    """Verify validation errors on missing files and invalid schemas."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_and_validate_dataset(tmp_path / "missing.csv")

    # Incomplete schema
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Expected 249 rows"):
        load_and_validate_dataset(bad_csv)


def test_compute_distribution_summary(dataset_df: pd.DataFrame):
    """Verify distribution summary calculation and structure."""
    summary_df = compute_distribution_summary(dataset_df, FEATURE_COLUMNS)
    assert len(summary_df) == len(FEATURE_COLUMNS)
    required_cols = {"feature", "min", "median", "max", "mean", "std", "skewness", "outlier_count_iqr"}
    assert required_cols.issubset(set(summary_df.columns))

    # Min should always be <= Max
    assert (summary_df["min"] <= summary_df["max"]).all()


def test_find_highly_correlated_pairs():
    """Verify identification of high correlations on synthetic data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    df_syn = pd.DataFrame({
        "feat_a": x,
        "feat_b": 2.0 * x + 1.0,  # perfect correlation rho = 1.0
        "feat_c": -3.0 * x,       # perfect inverse correlation rho = -1.0
        "feat_d": np.random.randn(50),  # low correlation
    })

    high_corr = find_highly_correlated_pairs(df_syn, ["feat_a", "feat_b", "feat_c", "feat_d"], threshold=0.90)
    assert len(high_corr) == 3  # (a,b), (a,c), (b,c)
    assert high_corr[0]["abs_rho"] == 1.0


def test_compute_class_comparisons(dataset_df: pd.DataFrame):
    """Verify non-parametric comparison metrics and FDR bounds."""
    comp_df = compute_class_comparisons(dataset_df, FEATURE_COLUMNS)
    assert len(comp_df) == len(FEATURE_COLUMNS)

    # Check bounds of Cliff's delta [-1, 1]
    assert (comp_df["cliffs_delta"] >= -1.0).all()
    assert (comp_df["cliffs_delta"] <= 1.0).all()

    # Check bounds of p-values [0, 1]
    assert (comp_df["p_value_raw"] >= 0.0).all() and (comp_df["p_value_raw"] <= 1.0).all()
    assert (comp_df["p_value_fdr"] >= 0.0).all() and (comp_df["p_value_fdr"] <= 1.0).all()
    assert (comp_df["p_value_bonferroni"] >= 0.0).all() and (comp_df["p_value_bonferroni"] <= 1.0).all()

    # FDR p-value should be >= raw p-value
    assert (comp_df["p_value_fdr"] >= comp_df["p_value_raw"] - 1e-6).all()


def test_eda_artifacts_exist():
    """Verify that report and figure artifacts exist and are non-empty."""
    assert REPORT_PATH.exists()
    assert REPORT_PATH.stat().st_size > 500

    expected_figures = [
        "feature_correlation_heatmap.png",
        "class_comparison_boxplots.png",
        "crop_dependency_analysis.png",
    ]
    for fig_name in expected_figures:
        fig_file = FIG_DIR / fig_name
        assert fig_file.exists(), f"Figure {fig_name} missing"
        assert fig_file.stat().st_size > 1000, f"Figure {fig_name} is empty"
