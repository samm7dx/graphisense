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

    - **Random Forest**: 5-Fold Cross-Validation (N=249) — metrics are cross-validation estimates
    - **CNN/ResNet-18**: Held-out test split (15% of data) — metrics are single test-set estimates

    These evaluation protocols are **not directly comparable**. A fair comparison would require both models to be evaluated on the same held-out test set using the same protocol.
    """)

    st.divider()

    from app.services import load_rf_metrics_for_comparison, load_cnn_metrics_for_comparison, load_rf_confusion_matrix
    import numpy as np
    from app.components.charts import plot_confusion_matrix

    # Load metrics
    rf_metrics = load_rf_metrics_for_comparison()
    cnn_metrics = load_cnn_metrics_for_comparison()

    if rf_metrics is not None and cnn_metrics is not None:
        # Metrics comparison table
        st.markdown("### Performance Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Random Forest (5-Fold CV)")
            st.metric("Accuracy", f"{rf_metrics.get('accuracy_mean', 0):.3f} ± {rf_metrics.get('accuracy_std', 0):.3f}")
            st.metric("Balanced Accuracy", f"{rf_metrics.get('balanced_accuracy_mean', 0):.3f} ± {rf_metrics.get('balanced_accuracy_std', 0):.3f}")
            st.metric("F1 (PD)", f"{rf_metrics.get('f1_score_mean', 0):.3f} ± {rf_metrics.get('f1_score_std', 0):.3f}")
            st.metric("ROC-AUC", f"{rf_metrics.get('roc_auc_mean', 0):.3f} ± {rf_metrics.get('roc_auc_std', 0):.3f}")

        with col2:
            st.markdown("##### CNN/ResNet-18 (Held-Out Test)")
            test_metrics = cnn_metrics.get("test_metrics", {})
            st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
            st.metric("Balanced Accuracy", f"{test_metrics.get('balanced_accuracy', 0):.3f}")
            st.metric("F1 (PD)", f"{test_metrics.get('f1_pd', 0):.3f}")
            st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")

        # Confusion matrices
        st.divider()
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown("##### RF Confusion Matrix (CV)")
            rf_cm = load_rf_confusion_matrix()
            if rf_cm is not None:
                fig = plot_confusion_matrix(np.array(rf_cm), labels=["LPD", "PD"])
                st.plotly_chart(fig, width="stretch", key="rf_cm_comparison")
            else:
                st.info("RF confusion matrix not available. Run full evaluation to generate.")

        with col_cm2:
            st.markdown("##### CNN Confusion Matrix (Test)")
            cnn_cm = cnn_metrics.get("test_confusion_matrix", {})
            if cnn_cm:
                cm = np.array(cnn_cm)
                fig = plot_confusion_matrix(cm, labels=["LPD", "PD"])
                st.plotly_chart(fig, width="stretch", key="cnn_cm_comparison")
            else:
                st.info("CNN confusion matrix not available.")

        # Per-class metrics
        st.divider()
        st.markdown("##### Per-Class Metrics Comparison")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Random Forest (CV)**")
            if rf_metrics:
                for cls in ["PD", "LPD"]:
                    p = rf_metrics.get(f"precision_{cls.lower()}_mean", 0)
                    r = rf_metrics.get(f"recall_{cls.lower()}_mean", 0)
                    f = rf_metrics.get(f"f1_{cls.lower()}_mean", 0)
                    st.metric(f"{cls} F1", f"{f:.3f}", delta=f"P={p:.3f}, R={r:.3f}")

        with col_p2:
            st.markdown("**CNN (Test)**")
            if cnn_metrics.get("test_metrics"):
                tm = cnn_metrics["test_metrics"]
                for cls in ["PD", "LPD"]:
                    p = tm.get(f"precision_{cls.lower()}", 0)
                    r = tm.get(f"recall_{cls.lower()}", 0)
                    f = tm.get(f"f1_{cls.lower()}", 0)
                    st.metric(f"{cls} F1", f"{f:.3f}", delta=f"P={p:.3f}, R={r:.3f}")
    else:
        st.info("Comparison requires both models to be trained and evaluated.")
        if rf_metrics is None:
            st.warning("Random Forest metrics not found.")
        if cnn_metrics is None:
            st.warning("CNN evaluation results not found.")