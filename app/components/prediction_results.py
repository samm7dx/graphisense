"""Prediction results display component for GraphiSense."""

import streamlit as st
import cv2
import numpy as np


def render_prediction_results(result: dict):
    """Render the full prediction results display."""
    label = result["predicted_label"]
    probs = result["class_probabilities"]
    rgb_img = result["original_rgb"]
    binary_mask = result["binary_mask"]
    is_light = result["is_light_bg"]
    features_used = result.get("features_used", {})
    all_features = result.get("all_extracted_features", {})
    meta = result.get("model_metadata", {})

    # Prediction card
    if label == "PD":
        st.markdown(
            f"""
            <div class="pred-card pred-card-pd">
                <strong>Model Prediction:</strong> Potential Dysgraphia (PD)<br>
                <strong>Estimated Probability:</strong> PD: <strong>{probs['PD']:.1%}</strong> &nbsp;|&nbsp; LPD: <strong>{probs['LPD']:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="pred-card pred-card-lpd">
                <strong>Model Prediction:</strong> Low Potential Dysgraphia (LPD)<br>
                <strong>Estimated Probability:</strong> LPD: <strong>{probs['LPD']:.1%}</strong> &nbsp;|&nbsp; PD: <strong>{probs['PD']:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive probability gauges
    from app.components.charts import plot_probability_gauge
    st.plotly_chart(
        plot_probability_gauge(probs["PD"], probs["LPD"]),
        width="stretch",
        key="probability_gauge_main",
        config={"displayModeBar": False},
    )

    # Side-by-side images
    c_orig, c_proc = st.columns(2)
    with c_orig:
        st.markdown('<div class="image-caption">Original Image</div>', unsafe_allow_html=True)
        st.image(rgb_img, width="stretch")
    with c_proc:
        fg_pct = (binary_mask == 255).mean() * 100
        mode_tag = "Light background" if is_light else "Dark background"
        st.markdown(f'<div class="image-caption">Binary Mask (Otsu, foreground {fg_pct:.1f}%, {mode_tag})</div>', unsafe_allow_html=True)
        st.image(binary_mask, clamp=True, width="stretch")

    # Distance transform
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_INFERNO)
    dist_rgb = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB)

    st.markdown('<div class="image-caption">Medial Axis Distance Transform (brighter = thicker stroke center)</div>', unsafe_allow_html=True)
    st.image(dist_rgb, width="stretch")

    return features_used, all_features, meta


def render_preprocessing_preview_while_waiting(preprocessed: dict):
    """Show preprocessing preview while waiting for analysis."""
    rgb_img = preprocessed["rgb"]
    binary_mask = preprocessed["mask"]
    is_light = preprocessed["is_light"]

    c_orig, c_proc = st.columns(2)
    with c_orig:
        st.markdown('<div class="image-caption">Original Image</div>', unsafe_allow_html=True)
        st.image(rgb_img, width="stretch")
    with c_proc:
        fg_pct = (binary_mask == 255).mean() * 100
        mode_tag = "Light background" if is_light else "Dark background"
        st.markdown(f'<div class="image-caption">Binary Mask (Otsu, foreground {fg_pct:.1f}%, {mode_tag})</div>', unsafe_allow_html=True)
        st.image(binary_mask, clamp=True, width="stretch")

    st.info("Click **Analyze handwriting** to run feature extraction and model inference.")