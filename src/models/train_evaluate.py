"""Machine learning evaluation pipeline for GraphiSense handwriting classification.

Implements reproducible stratified cross-validation for Logistic Regression and
Random Forest classifiers across normalized (scale-invariant) and full feature subsets.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Reduced static feature subset (formerly Feature Set A)
# Resolution Dependency Note:
# While spatial ratio features (ink_density, centroid_x_norm, centroid_y_norm, active_row_ratio,
# component_aspect_ratio_mean, component_aspect_ratio_std) are dimensionless and scale-invariant,
# stroke width measurements (stroke_width_mean, stroke_width_median, stroke_width_std) are
# expressed in pixel units (2 * R_inscribed). Consequently, stroke thickness metrics depend on
# the scanner/camera capture resolution (DPI) of the document.
REDUCED_STATIC_FEATURES = [
    "ink_density",
    "centroid_x_norm",
    "centroid_y_norm",
    "active_row_ratio",
    "component_aspect_ratio_mean",
    "component_aspect_ratio_std",
    "stroke_width_mean",
    "stroke_width_median",
    "stroke_width_std",
]

# Backward-compatible alias for existing test suites
FEATURE_SET_A = REDUCED_STATIC_FEATURES

# Feature Set B: All 25 static features (includes crop dimensions and raw component distributions)
FEATURE_SET_B = [
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


def build_pipeline(model_type: str, random_state: int = 42) -> Pipeline:
    """Build a scikit-learn Pipeline ensuring all scalers are fitted only on training folds.

    Parameters
    ----------
    model_type : str
        'logistic_regression' or 'random_forest'.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.

    Raises
    ------
    ValueError
        If model_type is unsupported.
    """
    model_lower = model_type.lower()
    if model_lower == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ])
    elif model_lower == "random_forest":
        return Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=4,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ])
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'logistic_regression' or 'random_forest'."
        )


def evaluate_model_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute stratified K-fold cross-validation with zero data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary labels (1 = PD, 0 = LPD).
    model_type : str
        'logistic_regression' or 'random_forest'.
    n_splits : int, default=5
        Number of stratified folds.
    random_state : int, default=42
        Random seed for fold partitioning.

    Returns
    -------
    Dict[str, Any]
        Dictionary of fold-level and aggregated metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: List[Dict[str, float]] = []
    total_samples = len(y)
    oof_predictions = np.zeros(total_samples, dtype=int)
    oof_probabilities = np.zeros(total_samples, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build fresh pipeline for each fold
        pipeline = build_pipeline(model_type, random_state=random_state + fold_idx)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        oof_predictions[val_idx] = y_pred
        oof_probabilities[val_idx] = y_prob

        # Fold evaluation
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec_pd = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec_lpd = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall_pd": float(rec_pd),
            "recall_lpd": float(rec_lpd),
            "f1_score": float(f1),
            "roc_auc": float(auc),
        })

    # Out-of-fold cumulative metrics
    cum_cm = confusion_matrix(y, oof_predictions, labels=[0, 1])

    # Summary statistics across folds
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_pd",
        "recall_lpd",
        "f1_score",
        "roc_auc",
    ]
    mean_metrics = {f"{m}_mean": float(np.mean([f[m] for f in fold_metrics])) for m in metric_names}
    std_metrics = {f"{m}_std": float(np.std([f[m] for f in fold_metrics])) for m in metric_names}

    return {
        "model_type": model_type,
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "confusion_matrix": cum_cm,
    }


def run_all_evaluations(
    csv_path: Path = Path("outputs/features/features.csv"),
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run cross-validation for all 4 configurations: (LR, RF) x (Set A, Set B)."""
    df = pd.read_csv(csv_path)

    # Encode labels: PD -> 1, LPD -> 0
    y = np.where(df["class_label"] == "PD", 1, 0)

    configs = [
        ("Logistic Regression (Set A - Normalized)", "logistic_regression", FEATURE_SET_A),
        ("Logistic Regression (Set B - Full 25)", "logistic_regression", FEATURE_SET_B),
        ("Random Forest (Set A - Normalized)", "random_forest", FEATURE_SET_A),
        ("Random Forest (Set B - Full 25)", "random_forest", FEATURE_SET_B),
    ]

    all_results = {}
    summary_rows = []

    for name, model_type, feature_cols in configs:
        X = df[feature_cols]
        res = evaluate_model_cv(X, y, model_type, n_splits=5, random_state=random_state)
        all_results[name] = res

        means = res["mean_metrics"]
        stds = res["std_metrics"]

        summary_rows.append({
            "configuration": name,
            "model": model_type,
            "feature_set": "Set A (Normalized 9)" if len(feature_cols) == 9 else "Set B (Full 25)",
            "num_features": len(feature_cols),
            "accuracy_mean": round(means["accuracy_mean"], 4),
            "accuracy_std": round(stds["accuracy_std"], 4),
            "balanced_accuracy_mean": round(means["balanced_accuracy_mean"], 4),
            "balanced_accuracy_std": round(stds["balanced_accuracy_std"], 4),
            "precision_mean": round(means["precision_mean"], 4),
            "precision_std": round(stds["precision_std"], 4),
            "recall_pd_mean": round(means["recall_pd_mean"], 4),
            "recall_pd_std": round(stds["recall_pd_std"], 4),
            "recall_lpd_mean": round(means["recall_lpd_mean"], 4),
            "recall_lpd_std": round(stds["recall_lpd_std"], 4),
            "f1_score_mean": round(means["f1_score_mean"], 4),
            "f1_score_std": round(stds["f1_score_std"], 4),
            "roc_auc_mean": round(means["roc_auc_mean"], 4),
            "roc_auc_std": round(stds["roc_auc_std"], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    return {
        "summary_df": summary_df,
        "all_results": all_results,
        "y_true": y,
    }


def generate_evaluation_plots(
    all_results: Dict[str, Any],
    y_true: np.ndarray,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Generate and save publication-quality confusion matrices, ROC curves, and bar plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # 1. Confusion Matrices Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (config_name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        # cm structure: [[TN, FP], [FN, TP]]
        # Row: True (LPD=0, PD=1), Col: Predicted (LPD=0, PD=1)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred LPD (0)", "Pred PD (1)"], fontsize=10)
        ax.set_yticklabels(["True LPD (0)", "True PD (1)"], fontsize=10)
        ax.set_title(f"{config_name}\nCumulative OOF Confusion Matrix", fontsize=11, fontweight="bold")

        # Text annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=12)

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    saved_plots.append(cm_path)

    # 2. Cross-Validated Out-of-Fold ROC Curves
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (config_name, res) in enumerate(all_results.items()):
        oof_probs = res["oof_probabilities"]
        fpr, tpr, _ = roc_curve(y_true, oof_probs)
        auc_val = res["mean_metrics"]["roc_auc_mean"]
        auc_std = res["std_metrics"]["roc_auc_std"]
        ax.plot(
            fpr,
            tpr,
            label=f"{config_name} (AUC = {auc_val:.3f} ± {auc_std:.3f})",
            color=colors[idx],
            lw=2,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / PD Recall)", fontsize=11)
    ax.set_title("5-Fold Cross-Validated Out-of-Fold ROC Curves (N=249)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=180)
    plt.close()
    saved_plots.append(roc_path)

    # 3. Model Comparison Bar Chart
    metrics_to_plot = ["balanced_accuracy", "recall_pd", "recall_lpd", "roc_auc"]
    metric_labels = ["Bal. Accuracy", "Recall (PD)", "Recall (LPD)", "ROC-AUC"]
    configs = summary_df["configuration"].tolist()

    x = np.arange(len(metric_labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, row in summary_df.iterrows():
        means = [row[f"{m}_mean"] for m in metrics_to_plot]
        stds = [row[f"{m}_std"] for m in metrics_to_plot]
        pos = x + (idx - 1.5) * width
        ax.bar(pos, means, width, yerr=stds, label=row["configuration"], capsize=4)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model & Feature Set Performance Comparison (5-Fold Stratified CV)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    comp_path = output_dir / "model_comparison_metrics.png"
    plt.savefig(comp_path, dpi=180)
    plt.close()
    saved_plots.append(comp_path)

    return saved_plots


def generate_model_evaluation_report(
    summary_df: pd.DataFrame,
    all_results: Dict[str, Any],
    report_path: Path,
) -> None:
    """Generate a rigorous markdown report documenting baseline evaluation results and caveats."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Build summary table rows
    table_rows = []
    for _, row in summary_df.iterrows():
        table_rows.append(
            f"| **{row['configuration']}** | "
            f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f} | "
            f"{row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} | "
            f"{row['recall_pd_mean']:.3f} ± {row['recall_pd_std']:.3f} | "
            f"{row['recall_lpd_mean']:.3f} ± {row['recall_lpd_std']:.3f} | "
            f"{row['f1_score_mean']:.3f} ± {row['f1_score_std']:.3f} | "
            f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f} |"
        )
    table_str = "\n".join(table_rows)

    content = f"""# GraphiSense — Phase 4: Machine-Learning Baseline and Evaluation Report

## 1. Experimental Setup & Protocol

- **Dataset**: Mendeley Potential Dysgraphia Dataset (N=249 total: 114 PD, 135 LPD).
- **Evaluation Strategy**: 5-Fold Stratified Cross-Validation (`random_state=42`).
- **Data Leakage Prevention**: All feature scaling (`StandardScaler`) is enclosed within scikit-learn `Pipeline` objects, ensuring scalers are fit strictly on training folds and applied out-of-fold to validation folds.
- **Evaluated Models**:
  1. **Logistic Regression**: Linear regularized baseline (`C=1.0`, `solver='lbfgs'`, with `StandardScaler`).
  2. **Random Forest**: Non-linear ensemble baseline (`n_estimators=100`, `max_depth=5`, `min_samples_split=4`).
- **Evaluated Feature Sets**:
  - **Feature Set A (Normalized Baseline, 9 features)**: Emphasizes scale-invariant and crop-independent features (`ink_density`, `centroid_x_norm`, `centroid_y_norm`, `active_row_ratio`, `component_aspect_ratio_mean`, `component_aspect_ratio_std`, `stroke_width_mean`, `stroke_width_median`, `stroke_width_std`).
  - **Feature Set B (Full Static Set, 25 features)**: Diagnostic set containing all static features, including raw crop dimensions (`image_width`, `image_height`), raw pixel counts (`foreground_pixels`), and component size distributions.

---

## 2. Cross-Validation Results

The table below presents mean and standard deviation across 5 stratified folds for all 4 configurations:

| Configuration | Accuracy | Balanced Accuracy | Recall (PD) | Recall (LPD) | F1-Score (PD) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## 3. Comparison of Models and Feature Sets

1. **Feature Set A vs. Feature Set B**:
   - Feature Set A relies exclusively on normalized, scale-invariant stroke and density descriptors, preventing the model from exploiting annotator crop boundaries (`image_width`, `image_height`).
   - Feature Set B includes raw crop dimensions and component counts. Comparing both sets diagnostically reveals the degree to which raw dimensions influence separation.
2. **Logistic Regression vs. Random Forest**:
   - Random Forest captures non-linear interactions between stroke width variability and ink density.
   - Logistic Regression provides a transparent linear baseline with well-calibrated probabilities.
3. **Per-Class Sensitivity**:
   - Screening applications prioritize **Recall (PD)** to avoid missing potential dysgraphia indicators, balanced against **Recall (LPD)** to minimize unnecessary referrals.

---

## 4. Methodological Limitations & Integrity Caveats

1. **Absence of Writer Identifiers (Potential Leakage Risk)**:
   - As documented in Phase 3, the Mendeley release enumerates files sequentially (`PD (1)` ... `PD (114)`) without participant IDs or student session metadata.
   - Because writer identity cannot be established without fabrication, stratified cross-validation was partitioned at the **image level**.
   - *Critical Consequence*: If multiple cropped lines originated from the same child, images from the same writer may reside in both training and test folds. This means reported performance estimates may reflect **writer-specific stylistic familiarity** and could be optimistic relative to true writer-independent generalization.
2. **Static 2D Handwriting Boundary**:
   - Features represent 2D geometric and morphological traces. They do **not** capture kinematic velocity, pen pressure profiles, or cognitive hesitation pauses.
3. **Clinical Disclaimer**:
   - This evaluation distinguishes between the binary labels `PD` and `LPD` defined by the original annotators of this specific dataset release. It is **not** a clinical diagnostic system and must not be interpreted as validated medical diagnostic accuracy.

---

## 5. Generated Artifacts
- **Metrics Table**: `outputs/reports/model_evaluation_metrics.csv`
- **Confusion Matrices Grid**: `outputs/figures/confusion_matrices.png`
- **ROC Curves**: `outputs/figures/roc_curves.png`
- **Model Comparison Bar Plot**: `outputs/figures/model_comparison_metrics.png`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def train_and_save_final_model(
    model_type: str = "random_forest",
    feature_set_name: str = "reduced_static_features",
    csv_path: Union[str, Path] = Path("outputs/features/features.csv"),
    output_dir: Union[str, Path] = Path("outputs/models"),
    random_state: int = 42,
) -> Tuple[Path, Path]:
    """Train the selected baseline pipeline on the full available dataset and save artifacts.

    Parameters
    ----------
    model_type : str, default='random_forest'
        'random_forest' or 'logistic_regression'.
    feature_set_name : str, default='reduced_static_features'
        'reduced_static_features' or 'full_static_features'.
    csv_path : Union[str, Path], default='outputs/features/features.csv'
        Path to features CSV.
    output_dir : Union[str, Path], default='outputs/models'
        Target directory to store model and metadata.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[Path, Path]
        Paths to (saved_pipeline_file, saved_metadata_file).
    """
    df = pd.read_csv(Path(csv_path))

    if feature_set_name == "reduced_static_features":
        feature_cols = list(REDUCED_STATIC_FEATURES)
    elif feature_set_name == "full_static_features":
        feature_cols = list(FEATURE_SET_B)
    else:
        raise ValueError(f"Unsupported feature_set_name: '{feature_set_name}'")

    X = df[feature_cols]
    y = np.where(df["class_label"] == "PD", 1, 0)

    # Build and fit pipeline strictly on the full dataset
    pipeline = build_pipeline(model_type, random_state=random_state)
    pipeline.fit(X, y)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model_pipeline.joblib"
    metadata_path = out_dir / "model_metadata.json"

    # Save model artifact using joblib
    joblib.dump(pipeline, model_path)

    # Extract global model interpretability metrics
    clf = pipeline.named_steps["clf"]
    global_importance: Dict[str, float] = {}

    if hasattr(clf, "feature_importances_"):
        for feat, imp in zip(feature_cols, clf.feature_importances_):
            global_importance[feat] = round(float(imp), 4)
    elif hasattr(clf, "coef_"):
        for feat, coef in zip(feature_cols, clf.coef_[0]):
            global_importance[feat] = round(float(coef), 4)

    metadata = {
        "model_type": model_type,
        "feature_set_name": feature_set_name,
        "feature_names": feature_cols,
        "num_features": len(feature_cols),
        "class_labels": ["LPD", "PD"],
        "class_mapping": {"0": "LPD", "1": "PD"},
        "n_training_samples": int(len(y)),
        "training_distribution": {
            "PD": int((y == 1).sum()),
            "LPD": int((y == 0).sum()),
        },
        "random_state": random_state,
        "package_versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "global_feature_importance_or_coef": global_importance,
        "interpretability_notice": (
            "Feature importance is a dataset-level summary of model decision trees, "
            "not a sample-specific local attribution like SHAP."
            if model_type == "random_forest"
            else "Coefficients reflect standardized linear log-odds effects."
        ),
        "resolution_dependency": (
            "Stroke width features are in pixel units and assume consistent scan/crop resolution."
        ),
        "disclaimer": (
            "Experimental academic screening baseline; not a clinical medical diagnosis. "
            "Trained on the Mendeley Potential Dysgraphia Dataset."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path


def main() -> None:
    """Run Phase 4 model evaluation pipeline and save all deliverables."""
    csv_path = Path("outputs/features/features.csv")
    fig_dir = Path("outputs/figures")
    report_csv = Path("outputs/reports/model_evaluation_metrics.csv")
    report_md = Path("outputs/reports/model_evaluation_report.md")

    print("Running GraphiSense Phase 4 evaluation pipeline...")
    results_bundle = run_all_evaluations(csv_path, random_state=42)

    summary_df = results_bundle["summary_df"]
    all_results = results_bundle["all_results"]
    y_true = results_bundle["y_true"]

    # Save metrics CSV
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(report_csv, index=False)
    print(f"Saved evaluation metrics to {report_csv}")

    # Generate and save plots
    saved_plots = generate_evaluation_plots(all_results, y_true, summary_df, fig_dir)
    for p in saved_plots:
        print(f"Saved figure: {p}")

    # Generate markdown report
    generate_model_evaluation_report(summary_df, all_results, report_md)
    print(f"Saved evaluation report to {report_md}")

    print("\nSummary Results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
