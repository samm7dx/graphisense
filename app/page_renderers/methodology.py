"""Methodology page for GraphiSense."""

import streamlit as st


def render_methodology_page():
    """Render the Methodology page."""
    st.markdown("## Methodology")
    st.markdown("""
    GraphiSense implements a dual-analysis pipeline for handwriting analysis:
    """)

    st.markdown("### 📷 Image Input & Validation")
    st.markdown("""
    - **Supported formats**: JPG, JPEG, PNG
    - **Size limit**: 10 MB maximum
    - **Validation**: File existence, format, size, and decodability checks
    - **Color space handling**: Automatic conversion from BGR/RGB/RGBA to grayscale
    """)

    st.markdown("### 🔄 OpenCV Preprocessing Pipeline")
    st.markdown("""
    The preprocessing pipeline consists of 6 stages:

    1. **Original Image** — Input image as loaded (RGB)
    2. **Grayscale** — Single-channel luminance via OpenCV BGR→GRAY (ITU-R BT.601 weights)
    3. **Binary Mask (Otsu)** — Foreground/background separation via Otsu thresholding with polarity detection
    4. **Distance Transform** — Euclidean distance from foreground pixels to nearest background
    5. **Connected Components** — 8-connected component analysis with noise filtering (min area=5px²)
    6. **Row Projection** — Horizontal ink distribution (sum of foreground pixels per row)

    **Polarity handling**: Auto-detects dark/light background via border sampling (5% margins). Default assumes dataset convention (dark background, light ink).
    """)

    st.markdown("### 🔍 Static Feature Extraction (Random Forest)")
    st.markdown("""
    Extracts **25 static geometric, topological, and stroke distribution features**:

    - **Image Geometry**: Width, height, aspect ratio
    - **Ink Density**: Proportion of stroke pixels to total image area
    - **Connected Components**: Count, width/height/area statistics (mean, std)
    - **Spatial Distribution**: Normalized centroids, row projections, active row ratio
    - **Stroke Width Proxy**: Medial-axis Euclidean distance transform (2× inscribed radius)

    The Random Forest uses a **reduced subset of 9 features** (ink_density, centroid_x_norm, centroid_y_norm, active_row_ratio, component_aspect_ratio_mean, component_aspect_ratio_std, stroke_width_mean, stroke_width_median, stroke_width_std) for inference.
    """)

    st.markdown("### 🧠 CNN/ResNet-18 Pipeline (Advanced Analysis)")
    st.markdown("""
    - **Architecture**: ResNet-18 with pretrained ImageNet weights
    - **Input**: 224×224 RGB, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    - **Head**: Linear(512→2) for binary classification (PD/LPD)
    - **Training**: AdamW optimizer, cross-entropy loss, ReduceLROnPlateau scheduler
    - **Early stopping**: Patience=10, min_delta=1e-4
    - **Checkpointing**: Best model by validation loss
    """)

    st.markdown("### 📊 Evaluation Methodology")
    st.markdown("""
    **Random Forest**: Stratified 5-fold cross-validation (N=249). Metrics: accuracy, balanced accuracy, precision, recall (per class), F1, ROC-AUC.

    **CNN**: Held-out test split (15%). Metrics: accuracy, balanced accuracy, per-class precision/recall/F1, ROC-AUC, confusion matrix.

    ⚠️ **Important**: RF uses 5-fold CV; CNN uses held-out test split. Direct comparison requires identical evaluation protocol.

    **Dataset limitations**: No participant/writer IDs available. Image-level splitting may yield optimistic estimates if multiple samples from same writer appear in both train and test sets.
    """)


def render_about_dysgraphia_page():
    """Render the About Dysgraphia page."""
    st.markdown("## About Dysgraphia")
    st.markdown("""
    **Dysgraphia** is a specific learning disability that affects written expression. It can manifest as difficulties with:

    - **Handwriting legibility** - illegible or inconsistent letter formation
    - **Writing fluency** - slow, laborious writing process
    - **Spelling** - frequent spelling errors despite adequate reading ability
    - **Written expression** - difficulty organizing thoughts on paper
    - **Fine motor coordination** - awkward pencil grip, excessive pressure, or unusual writing posture

    ### Prevalence
    Dysgraphia affects approximately **5-20%** of school-aged children, with varying degrees of severity. It often co-occurs with other learning differences such as dyslexia, ADHD, or developmental coordination disorder.

    ### Impact
    Children with dysgraphia may experience:
    - Academic frustration and underachievement
    - Reduced self-esteem and confidence
    - Avoidance of writing tasks
    - Fatigue during writing activities

    ### Early Identification
    Early screening and intervention can significantly improve outcomes. GraphiSense provides a **computer vision-based screening support tool** that analyzes static handwriting images to identify potential dysgraphia indicators. This is intended as a **screening aid**, not a diagnostic tool.

    ### Important Disclaimer
    > **GraphiSense is an academic research prototype for screening support only. It does not provide medical, clinical, psychiatric, or educational diagnoses. Results should be interpreted by qualified professionals in conjunction with comprehensive assessment.**
    """)


def render_repository_info_page():
    """Render the Repository Info page."""
    st.markdown("## Repository Information")

    st.markdown("""
    ### Project Purpose
    GraphiSense is an academic research prototype investigating whether static handwriting morphology and geometric stroke features can provide screening support for potential dysgraphia indicators.
    """)

    st.markdown("### Repository Structure")
    st.code("""
graphisense/
├── app/
│   ├── main.py                 # Streamlit web application
│   └── pipeline_handler.py     # Decoupled pipeline handler
├── data/
│   ├── raw/                    # Raw image storage (dataset excluded via .gitignore)
│   │   ├── README.md
│   │   └── handwriting/        # Dataset directory
│   └── processed/              # Intermediate preprocessed data
├── notebooks/
│   ├── Stepwise-Guided-Notebook.ipynb
│   └── inference_demo.ipynb
├── outputs/
│   ├── features/               # Extracted feature dataset (features.csv)
│   ├── figures/                # Diagnostic plots, ROC curves, confusion matrices
│   ├── models/                 # Saved pipelines
│   │   ├── model_pipeline.joblib       # Random Forest pipeline
│   │   ├── model_metadata.json         # Random Forest metadata
│   │   └── cnn/                        # CNN checkpoints
│   │       └── checkpoints/
│   │           ├── best_model.pt
│   │       └── final_model.pt
│   └── reports/                # Statistical and model evaluation markdown reports
├── src/
│   ├── features/               # Static feature extraction
│   ├── models/                 # ML models (RF + CNN)
│   ├── preprocessing/          # OpenCV preprocessing
│   └── visualization/          # Plotting utilities
├── tests/                      # Unit and integration tests
├── requirements.txt
└── README.md
    """)

    st.markdown("### Key Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.code("outputs/models/model_pipeline.joblib")
        st.code("outputs/models/model_metadata.json")
    with col2:
        st.markdown("**CNN/ResNet-18**")
        st.code("outputs/models/cnn/checkpoints/best_model.pt")
        st.code("outputs/reports/cnn_evaluation/evaluation_results.json")

    st.markdown("### How to Run")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in data/raw/
#    data/raw/
#    ├── Potential Dysgraphia/
#    └── Low Potential Dysgraphia/

# 3. Generate features & train RF
python -m src.features.build_feature_dataset
python -m src.features.analyze_features
python -m src.models.train_evaluate

# 4. Train CNN (optional)
python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn --epochs 50

# 5. Evaluate CNN
python -m src.models.cnn_evaluate --checkpoint outputs/models/cnn/checkpoints/best_model.pt

# 6. Launch Streamlit app
streamlit run app/main.py
    """)

    st.markdown("### Known Limitations")
    st.markdown("""
    - **No participant IDs**: Dataset lacks writer identifiers; image-level splits may overestimate generalization
    - **Static images only**: Cannot measure dynamic properties (pressure, velocity, acceleration)
    - **CNN evaluation**: Small held-out test set (N=38); metrics have wide confidence intervals
    - **Research prototype**: Not validated for clinical use; screening support only
    """)


def render_comparison_page():
    """Render the RF vs CNN Comparison page."""
    st.markdown("## RF vs CNN Comparison")
    st.warning("""
    ⚠️ **Evaluation Protocol Difference**

    - **Random Forest**: 5-Fold Cross-Validation (N=249) — metrics are cross-validation estimates with standard deviation
    - **CNN/ResNet-18**: Held-out test split (15% = 38 images) — metrics are single test-set estimates

    These evaluation protocols are **not directly comparable**. A fair comparison would require both models to be evaluated on the same held-out test set using the same protocol.
    """)

    st.markdown("### Key Dataset Information")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Total Images", "249")
    with col_d2:
        st.metric("CNN Training", "174")
    with col_d3:
        st.metric("CNN Validation", "37")
    with col_d4:
        st.metric("CNN Test", "38")

    st.caption("CNN uses pretrained ImageNet weights (ResNet-18). Best checkpoint at Epoch 13. "
               "RF uses 9 handcrafted features on all 249 images via 5-fold CV.")

    st.divider()

    from app.services import load_rf_metrics_for_comparison, load_cnn_metrics_for_comparison, load_rf_confusion_matrix
    import numpy as np
    from app.components.charts import (
        plot_confusion_matrix,
        plot_protocol_aware_metric_comparison,
        plot_rf_cnn_confusion_matrices,
        plot_rf_cnn_per_class_comparison,
    )

    # Load metrics
    rf_metrics = load_rf_metrics_for_comparison()
    cnn_metrics = load_cnn_metrics_for_comparison()

    if rf_metrics is not None and cnn_metrics is not None:
        # Main metrics comparison with protocol-aware charts
        st.markdown("### Performance Comparison (Protocol-Aware)")
        st.caption("Values shown as percentages. RF includes ±1 std dev from 5-fold CV. CNN is single held-out test (N=38).")

        # Accuracy comparison
        st.plotly_chart(
            plot_protocol_aware_metric_comparison(
                rf_metrics, cnn_metrics,
                metric_key="accuracy_mean",
                metric_name="Accuracy",
                rf_std_key="accuracy_std",
            ),
            width="stretch",
            key="comparison_accuracy",
        )

        # Balanced Accuracy comparison
        st.plotly_chart(
            plot_protocol_aware_metric_comparison(
                rf_metrics, cnn_metrics,
                metric_key="balanced_accuracy_mean",
                metric_name="Balanced Accuracy",
                rf_std_key="balanced_accuracy_std",
            ),
            width="stretch",
            key="comparison_balanced_accuracy",
        )

        # F1-Score comparison
        st.plotly_chart(
            plot_protocol_aware_metric_comparison(
                rf_metrics, cnn_metrics,
                metric_key="f1_score_mean",
                metric_name="F1-Score (PD)",
                rf_std_key="f1_score_std",
            ),
            width="stretch",
            key="comparison_f1",
        )

        # ROC-AUC comparison
        st.plotly_chart(
            plot_protocol_aware_metric_comparison(
                rf_metrics, cnn_metrics,
                metric_key="roc_auc_mean",
                metric_name="ROC-AUC",
                rf_std_key="roc_auc_std",
            ),
            width="stretch",
            key="comparison_roc_auc",
        )

        st.divider()

        # Confusion matrices side by side
        st.markdown("### Confusion Matrix Comparison")
        rf_cm = load_rf_confusion_matrix()
        cnn_cm = cnn_metrics.get("test_confusion_matrix", [])

        if rf_cm is not None or cnn_cm:
            st.plotly_chart(
                plot_rf_cnn_confusion_matrices(rf_cm, cnn_cm),
                width="stretch",
                key="comparison_confusion_matrices",
            )
        else:
            st.info("Confusion matrices not available.")

        st.divider()

        # Per-class metrics comparison
        st.markdown("### Per-Class Metrics Comparison")
        st.plotly_chart(
            plot_rf_cnn_per_class_comparison(rf_metrics, cnn_metrics),
            width="stretch",
            key="comparison_per_class",
        )

        st.divider()

        # Summary table with exact values
        st.markdown("### Numerical Summary")
        test_metrics = cnn_metrics.get("test_metrics", {})

        summary_data = {
            "Metric": ["Accuracy", "Balanced Accuracy", "Precision (PD)", "Recall (PD)", "F1 (PD)", "ROC-AUC"],
            "RF Set A (5-Fold CV)": [
                f"{rf_metrics.get('accuracy_mean', 0)*100:.1f}% ± {rf_metrics.get('accuracy_std', 0)*100:.1f}%",
                f"{rf_metrics.get('balanced_accuracy_mean', 0)*100:.1f}% ± {rf_metrics.get('balanced_accuracy_std', 0)*100:.1f}%",
                f"{rf_metrics.get('precision_pd_mean', 0)*100:.1f}% ± {rf_metrics.get('precision_std', 0)*100:.1f}%",
                f"{rf_metrics.get('recall_pd_mean', 0)*100:.1f}% ± {rf_metrics.get('recall_pd_std', 0)*100:.1f}%",
                f"{rf_metrics.get('f1_pd_mean', 0)*100:.1f}% ± {rf_metrics.get('f1_score_std', 0)*100:.1f}%",
                f"{rf_metrics.get('roc_auc_mean', 0)*100:.1f}% ± {rf_metrics.get('roc_auc_std', 0)*100:.1f}%",
            ],
            "CNN (Held-Out Test)": [
                f"{test_metrics.get('accuracy', 0)*100:.1f}%",
                f"{test_metrics.get('balanced_accuracy', 0)*100:.1f}%",
                f"{test_metrics.get('precision_pd', 0)*100:.1f}%",
                f"{test_metrics.get('recall_pd', 0)*100:.1f}%",
                f"{test_metrics.get('f1_pd', 0)*100:.1f}%",
                f"{test_metrics.get('roc_auc', 0)*100:.1f}%",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

        # Protocol reminder
        st.info("""
        **Protocol Reminder**: RF uses stratified 5-fold cross-validation on all 249 images.
        CNN uses a single held-out test set of 38 images (15% split) with pretrained ImageNet weights.
        These are descriptive results from different evaluation protocols — not a controlled head-to-head comparison.
        """)

        # CNN-specific details
        st.markdown("### CNN Training Details")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Best Epoch", "13")
            st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0)*100:.1f}%")
        with col_c2:
            st.metric("Test Balanced Acc", f"{test_metrics.get('balanced_accuracy', 0)*100:.1f}%")
            st.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0)*100:.1f}%")
        with col_c3:
            st.metric("Test F1 (PD)", f"{test_metrics.get('f1_pd', 0)*100:.1f}%")
            st.metric("Test F1 (LPD)", f"{test_metrics.get('f1_lpd', 0)*100:.1f}%")

        # RF feature importance note
        st.markdown("### RF Model Details (Set A - 9 Features)")
        st.caption("""
        The RF model uses 9 handcrafted features: ink_density, centroid_x_norm, centroid_y_norm,
        active_row_ratio, component_aspect_ratio_mean, component_aspect_ratio_std,
        stroke_width_mean, stroke_width_median, stroke_width_std.
        """)

    else:
        st.info("Comparison requires both models to be trained and evaluated.")
        if rf_metrics is None:
            st.warning("Random Forest metrics not found.")
        if cnn_metrics is None:
            st.warning("CNN evaluation results not found.")