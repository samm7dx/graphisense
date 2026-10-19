"""GraphiSense Streamlit Web Application.

A compact, visualization-first interface for handwriting image preprocessing,
feature extraction, and baseline dysgraphia screening-support model inference.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.pipeline_handler import load_and_preprocess_image, run_app_inference

DATASET_ROOT = ROOT / "data" / "raw" / "handwriting" / "extracted" / "DATASET DYSGRAPHIA HANDWRITING"

st.set_page_config(
    page_title="GraphiSense | Handwriting Analysis",
    page_icon="✍️",
    layout="wide",
)

# Custom minimal CSS for compact, restrained visualization-first layout
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .header-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .header-left {
        display: flex;
        align-items: baseline;
        gap: 12px;
        flex-wrap: wrap;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 400;
        margin: 0;
    }
    .prototype-tag {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #0369A1;
        background-color: #E0F2FE;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .pred-card {
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .pred-card-pd {
        background-color: #FEF3C7;
        color: #92400E;
        border-left: 4px solid #D97706;
    }
    .pred-card-lpd {
        background-color: #F0FDF4;
        color: #166534;
        border-left: 4px solid #16A34A;
    }
    button[kind="primary"] {
        background-color: #0284C7 !important;
        border-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover {
        background-color: #0369A1 !important;
        border-color: #0369A1 !important;
    }
    .empty-state {
        border: 1px dashed #CBD5E1;
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #64748B;
        font-size: 0.95rem;
    }
    .prob-container {
        margin-bottom: 12px;
        padding: 10px 14px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 4px;
    }
    .prob-track {
        height: 8px;
        width: 100%;
        background-color: #E2E8F0;
        border-radius: 4px;
        overflow: hidden;
    }
    .footer-note {
        font-size: 0.8rem;
        color: #64748B;
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Compact Header Row
st.markdown(
    """
    <div class="header-row">
        <div class="header-left">
            <span class="app-title">GraphiSense</span>
            <span class="app-subtitle">Computer vision-based handwriting analysis</span>
        </div>
        <span class="prototype-tag">Research Prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Main Workspace: Two-column layout
col_input, col_viz = st.columns([4, 6], gap="large")

image_bytes = None
image_name = "sample.png"

with col_input:
    st.markdown("##### Input Handwriting Sample")

    input_source = st.radio(
        "Input source selection",
        options=["Sample Image", "Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_source == "Sample Image":
        sample_options = {
            "PD Sample 1 (Potential Dysgraphia)": DATASET_ROOT / "Potential Dysgraphia" / "PD (1).jpg",
            "PD Sample 2 (Potential Dysgraphia)": DATASET_ROOT / "Potential Dysgraphia" / "PD (42).jpg",
            "LPD Sample 1 (Low Potential Dysgraphia)": DATASET_ROOT / "Low Potential Dysgraphia" / "LPD (1).jpg",
            "LPD Sample 2 (Low Potential Dysgraphia)": DATASET_ROOT / "Low Potential Dysgraphia" / "LPD (42).jpg",
        }
        selected_sample = st.selectbox(
            "Select verified dataset sample",
            options=list(sample_options.keys()),
            label_visibility="collapsed",
        )
        sample_path = sample_options[selected_sample]

        if sample_path.exists():
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            image_name = sample_path.name
        else:
            st.error(
                f"Sample file not found at: {sample_path}. "
                "Ensure dataset is extracted under data/raw/handwriting or switch to 'Upload Image'."
            )
    else:
        uploaded = st.file_uploader(
            "Upload handwriting crop (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            image_name = uploaded.name

    # Background polarity control (compact)
    polarity_choice = st.selectbox(
        "Polarity handling",
        options=["auto", "dark_background", "light_background"],
        format_func=lambda x: {
            "auto": "Auto-detect polarity (recommended)",
            "dark_background": "Dark background (dataset standard)",
            "light_background": "Light background (paper document)",
        }[x],
        help="The dataset images have dark background with light ink. Paper photos have light background with dark ink.",
    )

    # Pre-analysis image info & action button
    if image_bytes is not None:
        try:
            rgb_preview, gray_preview, mask_preview, is_light_detected = load_and_preprocess_image(
                image_bytes, polarity=polarity_choice
            )
            h, w = rgb_preview.shape[:2]
            mode_desc = "Light background" if is_light_detected else "Dark background"
            st.caption(f"**Loaded**: `{image_name}` &nbsp;|&nbsp; Dimensions: `{w}×{h}` px &nbsp;|&nbsp; Detected: `{mode_desc}`")

            # Check if active selection changed to reset cached inference
            if (
                st.session_state.get("analyzed_image_name") != image_name
                or st.session_state.get("analyzed_polarity") != polarity_choice
            ):
                st.session_state.pop("inference_result", None)

            if st.button("Analyze handwriting", type="primary", width="stretch"):
                with st.spinner("Extracting features and querying model..."):
                    st.session_state["inference_result"] = run_app_inference(
                        image_input=image_bytes,
                        filename=image_name,
                        polarity=polarity_choice,
                    )
                    st.session_state["analyzed_image_name"] = image_name
                    st.session_state["analyzed_polarity"] = polarity_choice

        except FileNotFoundError as e:
            st.error(f"Required artifact not found: {e}")
        except ValueError as e:
            st.error(f"Input validation error: {e}")
        except Exception as e:
            st.error(f"Failed to inspect image: {type(e).__name__} — {e}")

# Right Column: Primary Visualizations
with col_viz:
    st.markdown("##### Primary Visualizations")
    result = st.session_state.get("inference_result")

    if result is None:
        st.markdown(
            """
            <div class="empty-state">
                <p style="margin: 0; font-weight: 500;">No analysis results to display</p>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem;">
                    Select a sample image or upload a handwriting crop on the left, then click <strong>Analyze handwriting</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        label = result["predicted_label"]
        probs = result["class_probabilities"]
        rgb_img = result["original_rgb"]
        binary_mask = result["binary_mask"]
        is_light = result["is_light_bg"]

        # Clear prediction label banner
        if label == "PD":
            st.markdown(
                f"""
                <div class="pred-card pred-card-pd">
                    <strong>Model prediction:</strong> Potential Dysgraphia (PD)<br>
                    <strong>Estimated class probability:</strong> PD: <strong>{probs['PD']:.1%}</strong> &nbsp;|&nbsp; LPD: <strong>{probs['LPD']:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="pred-card pred-card-lpd">
                    <strong>Model prediction:</strong> Low Potential Dysgraphia (LPD)<br>
                    <strong>Estimated class probability:</strong> LPD: <strong>{probs['LPD']:.1%}</strong> &nbsp;|&nbsp; PD: <strong>{probs['PD']:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Compact visual comparison of class probabilities
        pd_pct = probs['PD'] * 100
        lpd_pct = probs['LPD'] * 100
        st.markdown(
            f"""
            <div class="prob-container">
                <div style="margin-bottom: 8px;">
                    <div class="prob-row">
                        <span style="font-weight: 500; color: #334155;">Potential Dysgraphia (PD)</span>
                        <span style="font-weight: 600; color: #0F172A;">{probs['PD']:.1%}</span>
                    </div>
                    <div class="prob-track">
                        <div style="width: {pd_pct:.1f}%; height: 100%; background-color: #D97706; border-radius: 4px;"></div>
                    </div>
                </div>
                <div>
                    <div class="prob-row">
                        <span style="font-weight: 500; color: #334155;">Low Potential Dysgraphia (LPD)</span>
                        <span style="font-weight: 600; color: #0F172A;">{probs['LPD']:.1%}</span>
                    </div>
                    <div class="prob-track">
                        <div style="width: {lpd_pct:.1f}%; height: 100%; background-color: #0284C7; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Side-by-side original and processed images
        c_orig, c_proc = st.columns(2)
        with c_orig:
            st.caption(f"Original handwriting ({rgb_img.shape[1]}×{rgb_img.shape[0]} px)")
            st.image(rgb_img, width="stretch")
        with c_proc:
            fg_pct = (binary_mask == 255).mean() * 100
            mode_tag = "Light background" if is_light else "Dark background"
            st.caption(f"Binarized mask (Otsu, foreground {fg_pct:.1f}%, {mode_tag})")
            st.image(binary_mask, clamp=True, width="stretch")


# 3. Progressive Disclosure: Secondary Detail Sections
if result is not None:
    st.divider()
    tabs = st.tabs(["Feature Analysis", "Image Processing", "Model Details", "About & Limitations"])

    # Tab 1: Feature Analysis
    with tabs[0]:
        st.markdown("###### Extracted Features Used by Model")
        feats_used = result["features_used"]
        all_feats = result["all_extracted_features"]

        feature_descriptions = {
            "ink_density": ("Proportion of stroke pixels to total image area", "Ratio [0-1]"),
            "centroid_x_norm": ("Normalized horizontal center of stroke mass", "Ratio [0-1]"),
            "centroid_y_norm": ("Normalized vertical center of stroke mass", "Ratio [0-1]"),
            "active_row_ratio": ("Proportion of image rows containing handwriting strokes", "Ratio [0-1]"),
            "component_aspect_ratio_mean": ("Mean width-to-height ratio of connected components", "Ratio"),
            "component_aspect_ratio_std": ("Standard deviation of component aspect ratios", "Ratio"),
            "stroke_width_mean": ("Mean stroke thickness via medial-axis distance transform", "Pixels (px)"),
            "stroke_width_median": ("Median stroke thickness via medial-axis distance transform", "Pixels (px)"),
            "stroke_width_std": ("Standard deviation of stroke thickness across strokes", "Pixels (px)"),
        }

        feature_rows = []
        for name, val in feats_used.items():
            desc, unit = feature_descriptions.get(name, ("Static spatial feature", "-"))
            formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
            feature_rows.append({"Feature": name, "Measured Value": formatted_val, "Unit": unit, "Description": desc})

        st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

        with st.expander("View all 25 extracted static features"):
            all_rows = [{"Feature": k, "Measured Value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in all_feats.items()]
            st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)

    # Tab 2: Image Processing
    with tabs[1]:
        st.markdown("###### Preprocessing & Stroke Analysis")
        st.markdown(
            "The image pipeline converts inputs to grayscale, resolves polarity via border intensity sampling, "
            "and applies Otsu automatic thresholding (foreground strokes = 255 white, background = 0 black)."
        )

        # Genuine medial axis distance transform visualization
        dist_map = cv2.distanceTransform(result["binary_mask"], cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_INFERNO)
        dist_rgb = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB)

        col_dist1, col_dist2 = st.columns([1, 1])
        with col_dist1:
            mean_sw = feats_used.get("stroke_width_mean", 0.0)
            st.caption(f"Medial axis distance transform (Mean stroke thickness: {mean_sw:.2f} px)")
            st.image(dist_rgb, width="stretch")
        with col_dist2:
            st.markdown(
                """
                **Stroke Thickness Proxy Notes**:
                - The medial axis distance map computes the radius of the largest circle inscribable within each stroke.
                - Color gradient indicates stroke thickness (brighter = thicker center).
                - **Line segmentation notice**: Dataset images are cropped single handwriting lines. Multi-line segmentation is not performed on single-line inputs and is not simulated.
                """
            )

    # Tab 3: Model Details
    with tabs[2]:
        st.markdown("###### Classifier Architecture & Metadata")
        meta = result.get("model_metadata", {})
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Model Type", str(meta.get("model_type", "Random Forest")).replace("_", " ").title())
        col_m2.metric("Feature Set", "Reduced Static (9 feats)")
        col_m3.metric("Training Samples", f"{meta.get('n_training_samples', 249)} (5-Fold CV)")

        with st.expander("Full model metadata schema (JSON)"):
            st.json(meta)

    # Tab 4: About & Limitations
    with tabs[3]:
        st.markdown("###### Academic Screening-Support Scope & Integrity Safeguards")
        st.markdown(
            """
            - **Non-Diagnostic Notice**: GraphiSense is an academic research prototype. Outputs are statistical model estimates on a benchmark dataset and do not constitute clinical, psychiatric, or educational diagnoses.
            - **Research Label Nature**: Dataset labels (`PD` and `LPD`) reflect study research categories from the dataset source, not independently verified clinical medical evaluations.
            - **Static Image Limitations**: Static 2D handwriting images cannot measure dynamic physical properties such as actual pen-tip pressure, writing velocity, acceleration, or cognitive in-air hesitation pauses.
            - **Image-Level Validation**: The underlying dataset does not provide participant or writer identifiers. Cross-validation was performed at the image level, which may yield optimistic estimates if crops from the same writer appear across folds.
            - **Resolution Dependency**: Stroke thickness measurements are expressed in pixels and depend on image scan resolution.
            """
        )

# 4. Concise Persistent Footer
st.markdown(
    """
    <div class="footer-note">
        GraphiSense is an academic research prototype for potential dysgraphia screening support. It does not provide medical or clinical diagnoses.
    </div>
    """,
    unsafe_allow_html=True,
)
