"""Model details component for GraphiSense tabs."""

import streamlit as st
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

from app.services import (
    load_cnn_metadata,
    load_evaluation_metrics,
    load_features_dataset,
    load_model_metadata,
)
from app.components.charts import (
    plot_feature_importance,
    plot_feature_distributions,
    plot_stroke_width_distribution,
    plot_component_size_distribution,
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
CNN_CHECKPOINT_PATH = ROOT / "outputs" / "models" / "cnn" / "checkpoints" / "best_model.pt"


def render_rf_feature_analysis_tab(features_used: dict, all_features: dict, meta: dict):
    """Render the Random Forest Feature Analysis tab."""
    st.markdown("##### Features Used by Model")

    # Feature descriptions
    feature_descriptions = {
        "ink_density": ("Proportion of stroke pixels to total image area", "Ratio [0-1]"),
        "centroid_x_norm": ("Normalized horizontal center of stroke mass", "Ratio [0-1]"),
        "centroid_y_norm": ("Normalized vertical center of stroke mass", "Ratio [0-1]"),
        "active_row_ratio": ("Proportion of image rows containing handwriting strokes", "Ratio [0-1]"),
        "component_aspect_ratio_mean": ("Mean width-to-height ratio of connected components", "Ratio"),
        "component_aspect_ratio_std": ("Std. dev. of component aspect ratios", "Ratio"),
        "stroke_width_mean": ("Mean stroke thickness via medial-axis distance transform", "Pixels (px)"),
        "stroke_width_median": ("Median stroke thickness via medial-axis distance transform", "Pixels (px)"),
        "stroke_width_std": ("Std. dev. of stroke thickness across strokes", "Pixels (px)"),
    }

    # Features used table
    feature_rows = []
    for name, val in features_used.items():
        desc, unit = feature_descriptions.get(name, ("Static spatial feature", "-"))
        formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
        feature_rows.append({
            "Feature": name,
            "Measured Value": formatted_val,
            "Unit": unit,
            "Description": desc,
        })

    st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

    # Feature importance chart
    if meta.get("global_feature_importance_or_coef"):
        st.markdown("##### Feature Importance (Global)")
        st.plotly_chart(
            plot_feature_importance(meta["global_feature_importance_or_coef"]),
            width="stretch",
            key="rf_feature_importance",
        )
        st.caption("Mean decrease in impurity from the fitted Random Forest. Not sample-specific attribution (e.g., SHAP).")

    # Feature distributions
    st.markdown("##### Feature Distributions by Class")
    features_df = load_features_dataset()
    if features_df is not None:
        feat_names = list(features_used.keys())
        st.plotly_chart(
            plot_feature_distributions(features_df, feat_names, features_used),
            width="stretch",
            key="rf_feature_distributions",
        )
        st.caption("Violin plots show full distribution; box plots show median/IQR. Diamond = current sample.")
    else:
        st.info("Feature dataset not found. Run `python -m src.features.build_feature_dataset` to enable distribution charts.")

    # All features expander
    with st.expander("View all 25 extracted static features"):
        all_rows = [{"Feature": k, "Measured Value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in all_features.items()]
        st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)


def render_cnn_model_architecture_tab():
    """Render the CNN Model Architecture tab."""
    st.markdown("##### CNN Model Architecture")

    cnn_meta = load_cnn_metadata()
    if cnn_meta:
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Architecture", cnn_meta.get("model_type", "ResNet-18"))
        with col_a2:
            st.metric("Input Size", f"{cnn_meta.get('image_size', 224)}×{cnn_meta.get('image_size', 224)}")
        with col_a3:
            st.metric("Dropout", f"{cnn_meta.get('dropout', 0.3):.1f}")

        st.markdown("---")
        st.markdown("**Architecture Details**")
        st.markdown(f"""
        - **Base Model**: ResNet-18 (pretrained on ImageNet: {cnn_meta.get('pretrained', True)})
        - **Input Resolution**: {cnn_meta.get('image_size', 224)}×{cnn_meta.get('image_size', 224)} RGB
        - **Backbone Frozen**: {cnn_meta.get('freeze_backbone', False)}
        - **Dropout Rate**: {cnn_meta.get('dropout', 0.3)}
        - **Final Layer**: Linear(512 → 2) for binary classification (PD/LPD)
        - **Training Epochs**: {cnn_meta.get('training_epochs', 0)}
        - **Best Validation Loss**: {cnn_meta.get('val_loss', 'N/A'):.4f}
        """)

        # Validation metrics if available
        val_metrics = cnn_meta.get("val_metrics", {})
        if val_metrics:
            st.markdown("##### Validation Metrics (Best Epoch)")
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            with col_v1:
                st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
            with col_v2:
                st.metric("Balanced Acc", f"{val_metrics.get('balanced_accuracy', 0):.3f}")
            with col_v3:
                st.metric("F1 (PD)", f"{val_metrics.get('f1_pd', 0):.3f}")
            with col_v4:
                st.metric("ROC-AUC", f"{val_metrics.get('roc_auc', 0):.3f}")
    else:
        st.info("CNN metadata not available.")

    st.markdown("---")
    st.caption("CNN models learn features directly from pixels. Unlike the Random Forest baseline, "
               "there are no handcrafted features to display. The model learns hierarchical "
               "visual representations through convolutional layers.")


def render_cnn_training_history_tab():
    """Render the CNN Training History tab."""
    st.markdown("##### Training History")
    cnn_meta = load_cnn_metadata()
    if cnn_meta:
        training_epochs = cnn_meta.get("training_epochs", 0)
        st.metric("Epochs Completed", training_epochs)

        # Try to load training history
        try:
            import torch
            checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            history = checkpoint.get("training_history", {})

            if history.get("train_loss"):
                import plotly.graph_objects as go
                epochs = list(range(1, len(history["train_loss"]) + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="#0284C7")))
                fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="#D97706")))
                fig.update_layout(
                    title="Training & Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=350,
                    margin=dict(l=60, r=40, t=60, b=40),
                    plot_bgcolor="white",
                )
                fig.update_xaxes(gridcolor="#E2E8F0")
                fig.update_yaxes(gridcolor="#E2E8F0")
                st.plotly_chart(fig, width="stretch", key="cnn_training_loss")

            if history.get("train_acc"):
                fig = go.Figure()
                epochs = list(range(1, len(history["train_acc"]) + 1))
                fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc", line=dict(color="#16A34A")))
                fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc", line=dict(color="#F59E0B")))
                fig.update_layout(
                    title="Training & Validation Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    height=350,
                    margin=dict(l=60, r=40, t=60, b=40),
                    plot_bgcolor="white",
                )
                fig.update_xaxes(gridcolor="#E2E8F0")
                fig.update_yaxes(gridcolor="#E2E8F0")
                st.plotly_chart(fig, width="stretch", key="cnn_training_acc")
        except Exception:
            st.info("Training history not available in checkpoint.")
    else:
        st.info("CNN metadata not available.")


def render_cnn_evaluation_tab():
    """Render the CNN Evaluation tab."""
    st.markdown("##### Held-Out Evaluation Results")
    cnn_meta = load_cnn_metadata()
    if cnn_meta:
        val_metrics = cnn_meta.get("val_metrics", {})
        if val_metrics:
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                st.metric("Val Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
            with col_e2:
                st.metric("Val Balanced Acc", f"{val_metrics.get('balanced_accuracy', 0):.3f}")
            with col_e3:
                st.metric("Val F1 (PD)", f"{val_metrics.get('f1_pd', 0):.3f}")
            with col_e4:
                st.metric("Val ROC-AUC", f"{val_metrics.get('roc_auc', 0):.3f}")

            st.markdown("---")

            # Per-class metrics
            st.markdown("##### Per-Class Metrics (Validation)")
            per_class_data = [
                ("LPD", "Precision", val_metrics.get("precision_lpd", 0)),
                ("LPD", "Recall", val_metrics.get("recall_lpd", 0)),
                ("LPD", "F1", val_metrics.get("f1_lpd", 0)),
                ("PD", "Precision", val_metrics.get("precision_pd", 0)),
                ("PD", "Recall", val_metrics.get("recall_pd", 0)),
                ("PD", "F1", val_metrics.get("f1_pd", 0)),
            ]

            for cls_name, metric_name, value in per_class_data:
                st.metric(f"{cls_name} {metric_name}", f"{value:.3f}")

            # Confusion matrix if available
            if "confusion_matrix" in val_metrics:
                st.markdown("---")
                st.markdown("##### Validation Confusion Matrix")
                import plotly.graph_objects as go
                import numpy as np
                cm = np.array(val_metrics["confusion_matrix"])
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Pred LPD", "Pred PD"],
                    y=["True LPD", "True PD"],
                    colorscale="Blues",
                    showscale=True,
                    text=[[f"{cm[i, j]}" for j in range(2)] for i in range(2)],
                    texttemplate="%{text}",
                    textfont={"size": 16},
                ))
                fig.update_layout(height=300, margin=dict(l=80, r=40, t=40, b=60))
                st.plotly_chart(fig, width="stretch", key="cnn_val_cm")
        else:
            st.info("Validation metrics not available in checkpoint.")
    else:
        st.info("CNN metadata not available.")


def render_image_processing_tab(result: dict, is_cnn: bool):
    """Render the Image Processing tab for both RF and CNN."""
    st.markdown("##### Preprocessing Pipeline")

    col_proc1, col_proc2 = st.columns(2)

    with col_proc1:
        st.markdown("**Original (RGB)**")
        st.image(result["original_rgb"], width="stretch")
        st.caption(f"{result['original_rgb'].shape[1]}×{result['original_rgb'].shape[0]} px")

    with col_proc2:
        st.markdown("**Grayscale**")
        st.image(result["grayscale"], width="stretch", clamp=True)
        st.caption(f"{result['grayscale'].shape[1]}×{result['grayscale'].shape[0]} px")

    st.markdown("---")
    st.markdown("**Binary Mask (Otsu Thresholding)**")
    st.image(result["binary_mask"], width="stretch", clamp=True)
    fg_pct = (result["binary_mask"] == 255).mean() * 100
    st.caption(f"Foreground: {fg_pct:.1f}%  |  Convention: Handwriting=255 (white), Background=0 (black)")

    if not is_cnn:
        st.markdown("---")
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            st.markdown("**Stroke Width Distribution**")
            st.plotly_chart(
                plot_stroke_width_distribution(result["binary_mask"]),
                width="stretch",
                key="rf_stroke_width_dist",
            )

        with col_dist2:
            st.markdown("**Component Area Distribution**")
            st.plotly_chart(
                plot_component_size_distribution(result["binary_mask"]),
                width="stretch",
                key="rf_component_area_dist",
            )

    if is_cnn:
        st.markdown("---")
        st.markdown("**CNN Input Tensor (Resized & Normalized)**")
        # Create a temporary file to use the existing preprocess_image method
        import tempfile
        from pathlib import Path
        from src.models.cnn_inference import CNNInference
        from PIL import Image
        
        try:
            engine = CNNInference(CNN_CHECKPOINT_PATH)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_path = Path(tf.name)
                Image.fromarray(result["original_rgb"]).save(temp_path)
            try:
                input_tensor = engine.preprocess_image(temp_path)
                tensor_vis = input_tensor[0, 0].cpu().numpy()
                tensor_vis = (tensor_vis - tensor_vis.min()) / (tensor_vis.max() - tensor_vis.min() + 1e-8)
                tensor_vis = (tensor_vis * 255).astype(np.uint8)
                tensor_rgb = cv2.cvtColor(tensor_vis, cv2.COLOR_GRAY2RGB)
                st.image(tensor_rgb, width="stretch", clamp=True)
                st.caption(f"CNN Input Tensor: {input_tensor.shape} (normalized)")
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as e:
            st.warning(f"Could not display CNN input tensor: {e}")

    with st.expander("Technical Details"):
        st.markdown(f"""
        - **Polarity**: {result.get('resolved_polarity', 'unknown')}
        - **Binarization**: Otsu automatic thresholding
        - **Connected Components**: 8-connectivity, min area = 5 px²
        - **Stroke Width**: 2 × inscribed radius (distance transform ridges)
        - **Line Segmentation**: Not performed (dataset contains single-line crops)
        """)


def render_rf_model_details_tab(meta: dict):
    """Render the Random Forest Model Details tab."""
    st.markdown("##### Classifier Architecture & Metadata")

    # Metric cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{meta.get('model_type', 'Random Forest').replace('_', ' ').title()}</div>
            <div class="metric-label">Model Type</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{meta.get('feature_set_name', 'Unknown').replace('_', ' ').title()}</div>
            <div class="metric-label">Feature Set</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{meta.get('n_training_samples', '?')}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{meta.get('num_features', '?')}</div>
            <div class="metric-label">Features Used</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Confusion matrix (RF)
    st.markdown("##### Out-of-Fold Confusion Matrix")
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    cm_path = ROOT / "outputs" / "figures" / "confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), width="stretch")
        st.caption("Cumulative out-of-fold confusion matrices for all 4 model/feature-set configurations.")
    else:
        st.info("Confusion matrix figure not found. Run full evaluation to generate.")

    st.markdown("---")

    # Model comparison chart
    st.markdown("##### Model Comparison (5-Fold CV)")
    eval_metrics = load_evaluation_metrics()
    if eval_metrics is not None:
        st.plotly_chart(
            plot_metrics_comparison(eval_metrics),
            width="stretch",
            key="rf_model_comparison",
        )
        st.caption("Mean ± std across 5 stratified folds. Feature Set A = 9 normalized features; Set B = 25 full features.")
    else:
        st.info("Evaluation metrics not found. Run `python -m src.models.train_evaluate` to generate.")

    st.markdown("---")

    # Full metadata
    with st.expander("Full Model Metadata (JSON)"):
        st.json(meta)


def render_limitations_tab(is_cnn: bool):
    """Render the Limitations tab."""
    st.markdown("##### Academic Screening-Support Scope & Integrity Safeguards")
    base_content = """
    - **Non-Diagnostic Notice**: GraphiSense is an academic research prototype. Outputs are statistical model estimates on a benchmark dataset and do not constitute clinical, psychiatric, or educational diagnoses.

    - **Research Label Nature**: Dataset labels (`PD` and `LPD`) reflect study research categories from the dataset source, not independently verified clinical medical evaluations.

    - **Static Image Limitations**: Static 2D handwriting images cannot measure dynamic physical properties such as actual pen-tip pressure, writing velocity, acceleration, or cognitive in-air hesitation pauses. All features (stroke thickness, ink density) are purely 2D morphological proxies.

    - **Image-Level Validation Caveat**: The underlying dataset does not provide participant or writer identifiers. Cross-validation was partitioned at the **image level**. If multiple line crops originate from the same writer, samples from the same writer may occur across training and validation folds, which can yield optimistic performance estimates compared to true writer-independent evaluation. No writer IDs are invented or assumed.

    - **Resolution Dependency**: Stroke thickness measurements are expressed in pixels and depend on image scan/capture resolution. Features are not DPI-normalized.

    - **Screening vs. Diagnosis**: This tool provides screening *support* only. A clinical diagnosis of dysgraphia requires multidisciplinary assessment including dynamic writing analysis, cognitive evaluation, and educational context.
    """

    if is_cnn:
        cnn_content = base_content + """
    - **CNN-Specific Limitations**: The ResNet-18 model learns visual patterns directly from pixels without explicit feature engineering. The decision process is less interpretable than the Random Forest baseline. No Grad-CAM or attention visualization is currently implemented.
        """
        st.markdown(cnn_content)
    else:
        st.markdown(base_content)