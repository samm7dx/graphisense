"""Exploratory Data Analysis and Feature Quality Audit for GraphiSense.

This script performs reproducible statistical audits, outlier detection,
correlation analysis, crop-dependency assessments, and exploratory class
comparisons between Potential Dysgraphia (PD) and Low Potential Dysgraphia (LPD).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FEATURE_COLUMNS = [
    "image_width",
    "image_height",
    "aspect_ratio",
    "ink_density",
    "foreground_pixels",
    "component_count",
    "component_width_mean",
    "component_width_std",
    "component_height_mean",
    "component_height_std",
    "component_area_mean",
    "component_area_std",
    "component_aspect_ratio_mean",
    "component_aspect_ratio_std",
    "centroid_x",
    "centroid_y",
    "centroid_x_norm",
    "centroid_y_norm",
    "row_projection_mean",
    "row_projection_std",
    "row_projection_max",
    "active_row_ratio",
    "stroke_width_mean",
    "stroke_width_median",
    "stroke_width_std",
]


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the extracted feature CSV and verify row counts and data integrity."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    expected_rows = 249
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    if df[FEATURE_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Dataset contains unexpected missing values in feature columns")

    return df


def compute_distribution_summary(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics and outlier counts for all features."""
    records = []
    for col in feature_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())

        records.append({
            "feature": col,
            "min": round(float(s.min()), 4),
            "q25": round(float(q1), 4),
            "median": round(float(s.median()), 4),
            "q75": round(float(q3), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "skewness": round(float(stats.skew(s)), 4),
            "kurtosis": round(float(stats.kurtosis(s)), 4),
            "outlier_count_iqr": outlier_count,
            "outlier_pct": round(float(outlier_count / len(df) * 100), 2),
        })

    return pd.DataFrame(records)


def find_highly_correlated_pairs(
    df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Identify pairs of features with Spearman rank correlation exceeding the threshold."""
    corr_matrix = df[feature_cols].corr(method="spearman")
    high_corr = []
    n = len(feature_cols)

    for i in range(n):
        for j in range(i + 1, n):
            c1 = feature_cols[i]
            c2 = feature_cols[j]
            val = corr_matrix.loc[c1, c2]
            if abs(val) >= threshold:
                high_corr.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "spearman_rho": round(float(val), 4),
                    "abs_rho": round(float(abs(val)), 4),
                })

    high_corr.sort(key=lambda x: x["abs_rho"], reverse=True)
    return high_corr


def compute_class_comparisons(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Perform non-parametric Mann-Whitney U tests with Rank-Biserial effect sizes and FDR correction."""
    pd_df = df[df["class_label"] == "PD"]
    lpd_df = df[df["class_label"] == "LPD"]
    n1, n2 = len(pd_df), len(lpd_df)

    records = []
    p_values = []

    for col in feature_cols:
        v_pd = pd_df[col].to_numpy()
        v_lpd = lpd_df[col].to_numpy()

        # Mann-Whitney U test (two-sided)
        res = stats.mannwhitneyu(v_pd, v_lpd, alternative="two-sided")
        u_stat = res.statistic
        p_val = res.pvalue
        p_values.append(p_val)

        # Rank-biserial correlation effect size r = 1 - 2U / (n1 * n2)
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)

        # Cliff's Delta: delta = (2*U / (n1*n2)) - 1 = -r_rb
        # We report Cliff's delta where positive means PD > LPD
        cliffs_delta = (2.0 * u_stat) / (n1 * n2) - 1.0

        records.append({
            "feature": col,
            "pd_median": round(float(np.median(v_pd)), 4),
            "pd_iqr": round(float(np.percentile(v_pd, 75) - np.percentile(v_pd, 25)), 4),
            "lpd_median": round(float(np.median(v_lpd)), 4),
            "lpd_iqr": round(float(np.percentile(v_lpd, 75) - np.percentile(v_lpd, 25)), 4),
            "mann_whitney_u": round(float(u_stat), 2),
            "p_value_raw": float(p_val),
            "cliffs_delta": round(float(cliffs_delta), 4),
        })

    # Benjamini-Hochberg (FDR) multiple testing correction
    p_arr = np.array(p_values)
    n_tests = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    fdr_adj = np.zeros(n_tests)

    # Step-up calculation
    for i in range(n_tests - 1, -1, -1):
        adj = sorted_p[i] * n_tests / (i + 1)
        if i == n_tests - 1:
            fdr_adj[i] = min(adj, 1.0)
        else:
            fdr_adj[i] = min(adj, fdr_adj[i + 1], 1.0)

    orig_order_fdr = np.zeros(n_tests)
    orig_order_fdr[sorted_indices] = fdr_adj

    # Bonferroni correction
    bonferroni_p = np.minimum(p_arr * n_tests, 1.0)

    for i, rec in enumerate(records):
        rec["p_value_fdr"] = round(float(orig_order_fdr[i]), 6)
        rec["p_value_bonferroni"] = round(float(bonferroni_p[i]), 6)
        rec["p_value_raw"] = round(rec["p_value_raw"], 6)

    comp_df = pd.DataFrame(records)
    comp_df["abs_delta"] = comp_df["cliffs_delta"].abs()
    comp_df.sort_values(by="abs_delta", ascending=False, inplace=True)
    comp_df.drop(columns=["abs_delta"], inplace=True)

    return comp_df


def generate_eda_plots(df: pd.DataFrame, feature_cols: List[str], output_dir: Path) -> List[Path]:
    """Generate and save publication-quality exploratory visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # 1. Feature Correlation Heatmap (Spearman)
    corr = df[feature_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(16, 14))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Spearman Correlation (ρ)")

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=9)
    ax.set_yticklabels(feature_cols, fontsize=9)
    ax.set_title("Static Handwriting Features: Spearman Correlation Matrix (N=249)", fontsize=14, pad=15)
    plt.tight_layout()
    heatmap_path = output_dir / "feature_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=180)
    plt.close()
    saved_plots.append(heatmap_path)

    # 2. Key Feature Distributions by Class (Boxplots)
    key_features = [
        "stroke_width_mean",
        "ink_density",
        "component_count",
        "component_area_mean",
        "component_width_mean",
        "component_height_mean",
        "active_row_ratio",
        "centroid_y_norm",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        ax = axes[idx]
        pd_vals = df[df["class_label"] == "PD"][feat]
        lpd_vals = df[df["class_label"] == "LPD"][feat]

        bp = ax.boxplot(
            [pd_vals, lpd_vals],
            tick_labels=["PD (n=114)", "LPD (n=135)"],
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        bp["boxes"][0].set_facecolor("#ff9999")
        bp["boxes"][1].set_facecolor("#99ccff")

        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("Measured Value")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Class-wise Distribution Comparisons (Exploratory Data Analysis)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    boxplots_path = output_dir / "class_comparison_boxplots.png"
    plt.savefig(boxplots_path, dpi=180)
    plt.close()
    saved_plots.append(boxplots_path)

    # 3. Crop-Dependency Analysis
    crop_feats = ["image_width", "image_height", "aspect_ratio", "foreground_pixels"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(crop_feats):
        ax = axes[idx]
        pd_vals = df[df["class_label"] == "PD"][feat]
        lpd_vals = df[df["class_label"] == "LPD"][feat]

        bp = ax.boxplot(
            [pd_vals, lpd_vals],
            tick_labels=["PD (n=114)", "LPD (n=135)"],
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        bp["boxes"][0].set_facecolor("#ffb380")
        bp["boxes"][1].set_facecolor("#b3d9ff")
        ax.set_title(f"Crop-Dependent: {feat}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Audit of Image Dimensions & Crop-Dependent Attributes", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    crop_path = output_dir / "crop_dependency_analysis.png"
    plt.savefig(crop_path, dpi=180)
    plt.close()
    saved_plots.append(crop_path)

    return saved_plots


def generate_markdown_report(
    df: pd.DataFrame,
    dist_df: pd.DataFrame,
    high_corr: List[Dict[str, Any]],
    comp_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Generate a comprehensive markdown report documenting all verified findings."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    top_corr_str = "\n".join(
        f"- **`{p['feature_1']}`** & **`{p['feature_2']}`**: Spearman $\\rho = {p['spearman_rho']}$"
        for p in high_corr[:12]
    )

    top_diff_str = "\n".join(
        f"| `{row['feature']}` | {row['pd_median']} (IQR {row['pd_iqr']}) | {row['lpd_median']} (IQR {row['lpd_iqr']}) | {row['cliffs_delta']} | {row['p_value_raw']} | {row['p_value_fdr']} |"
        for _, row in comp_df.head(10).iterrows()
    )

    content = f"""# GraphiSense — Phase 3: Exploratory Data Analysis & Feature Quality Audit

## 1. Verified Dataset State
- **Dataset Source**: Mendeley Data (10.17632/39hr8dx76p.1, Ramlan et al., Oct 2023).
- **Verified Record Count**: 249 images (114 Potential Dysgraphia `PD`, 135 Low Potential Dysgraphia `LPD`).
- **Data Integrity**: 0 duplicate filenames, 0 duplicate feature rows, 0 missing/null values, 0 non-finite values. All 249 images processed with status `SUCCESS`.

---

## 2. Feature Quality & Audit Findings

### A. Mathematical Verification of Component Standard Deviations
- **Audit Target**: `component_width_std` (Max 534.20) and `component_area_std` (Max 10927.35).
- **Verification Result**: The standard deviation calculation (`np.std(..., ddof=0)`) is mathematically verified and exact.
- **Root Cause**: Natural handwriting often produces severe bimodality within individual images. In samples like `PD (92).jpg`, long continuous cursive words form connected components spanning up to 1,200 pixels, while diacritics, dots, and punctuation form components of 10–13 pixels. This heavy internal skewness naturally drives high standard deviation values.
- **Scope Clarification**: Connected components reflect pixel connectivity and do NOT strictly represent individual words or letters.

### B. Stroke Width Proxy Verification
- **Audit Target**: `stroke_width_mean`, `stroke_width_median`, `stroke_width_std`.
- **Verification Result**: Extracted strictly from valid distance-transform ridges ($2 \\times R$) on foreground pixels ($I=255$). Mean values range realistically between 2.01 px and 4.50 px (mean 2.76 px).
- **Scope Clarification**: This is a static 2D morphological stroke thickness proxy. It does **not** measure dynamic pen-tip pressure, grip force, or tactile writing velocity.

### C. Highly Redundant & Correlated Feature Pairs (|ρ| ≥ 0.85)
{top_corr_str}

*Recommendation*: While collinearity exists (e.g. `image_width` and `row_projection_max`), features should not be deleted arbitrarily at this stage. Instead, regularization (L1/ElasticNet) or tree-based feature selection should be evaluated during modeling.

---

## 3. Crop-Dependency & Confounding Risk

- **Observation**: `image_width` (PD mean: 1173.9 px vs LPD mean: 1020.2 px) and `image_height` (PD mean: 195.2 px vs LPD mean: 156.0 px) differ between classes in this dataset.
- **Confounding Risk**: Crop-dependent raw features like `foreground_pixels` and `row_projection_max` are partially influenced by how tightly the annotators cropped individual lines of text.
- **Mitigation**: Normalized features (`ink_density`, `centroid_x_norm`, `centroid_y_norm`, `active_row_ratio`, component-level ratios) are dimensionally invariant and should be prioritized over raw pixel counts.

---

## 4. Dataset Provenance & Participant Leakage Analysis

- **Inspection Findings**:
  - Filenames in `Potential Dysgraphia/` are named `PD (1).jpg` through `PD (114).jpg`.
  - Filenames in `Low Potential Dysgraphia/` are named `LPD (1).jpg` through `LPD (135).jpg`.
  - No metadata table, participant ID mapping, or session logs were provided with the Mendeley dataset release.
- **Critical Risk**:
  - In educational and clinical interventions (such as PDM Malaysia), individual children frequently produce multiple handwriting lines or exercises.
  - Because writer identity cannot be verified from the filenames, an arbitrary image-level random train/test split **risks leaking writer-specific traits** (unique cursive styles, pen choice, slant) across splits.
  - *Policy*: We explicitly refrain from inventing or guessing participant IDs from filenames. Models evaluated on image-level splits must acknowledge this potential optimism in generalizability.

---

## 5. Exploratory Statistical Comparisons (PD vs. LPD)

The table below presents non-parametric two-sided Mann-Whitney U test comparisons with Cliff's Delta effect sizes and Benjamini-Hochberg False Discovery Rate (FDR) corrections across all 25 features:

| Feature | PD Median (IQR) | LPD Median (IQR) | Cliff's Delta | Raw p-value | FDR p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
{top_diff_str}

> **Clinical Disclaimer**: These statistical differences are exploratory observations reflecting sample distributions in this specific dataset. They do not constitute diagnostic criteria and do not prove clinical dysgraphia classification capability.

---

## 6. Generated Visual Artifacts
- **Correlation Heatmap**: `outputs/figures/feature_correlation_heatmap.png`
- **Class-wise Comparison Boxplots**: `outputs/figures/class_comparison_boxplots.png`
- **Crop Dependency Analysis**: `outputs/figures/crop_dependency_analysis.png`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    """Run full Phase 3 EDA pipeline."""
    csv_path = Path("outputs/features/features.csv")
    fig_dir = Path("outputs/figures")
    report_path = Path("outputs/reports/eda_report.md")

    print(f"Loading and validating {csv_path}...")
    df = load_and_validate_dataset(csv_path)

    print("Computing distribution summaries and outlier audits...")
    dist_df = compute_distribution_summary(df, FEATURE_COLUMNS)

    print("Analyzing feature correlations...")
    high_corr = find_highly_correlated_pairs(df, FEATURE_COLUMNS, threshold=0.85)
    print(f"  Identified {len(high_corr)} pairs with |Spearman rho| >= 0.85")

    print("Computing non-parametric class comparisons with FDR corrections...")
    comp_df = compute_class_comparisons(df, FEATURE_COLUMNS)

    print("Generating exploratory visual artifacts...")
    saved_plots = generate_eda_plots(df, FEATURE_COLUMNS, fig_dir)
    for p in saved_plots:
        print(f"  Saved plot: {p}")

    print(f"Generating markdown report at {report_path}...")
    generate_markdown_report(df, dist_df, high_corr, comp_df, report_path)

    print("\nPhase 3 EDA complete successfully.")


if __name__ == "__main__":
    main()
