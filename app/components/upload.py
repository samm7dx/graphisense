"""Upload and input component for GraphiSense."""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data" / "raw"


def render_upload_section() -> tuple:
    """Render the image upload section and return (image_bytes, image_name, polarity_choice)."""
    st.markdown("#### Input Handwriting Sample")

    # Source selection
    input_source = st.radio(
        "Input source",
        options=["Sample Image", "Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_bytes = None
    image_name = "sample.png"

    if input_source == "Sample Image":
        # Build sample list dynamically
        sample_options = {}
        if DATASET_ROOT.exists():
            for cls_folder, label in [("Potential Dysgraphia", "PD"), ("Low Potential Dysgraphia", "LPD")]:
                folder = DATASET_ROOT / cls_folder
                if folder.exists():
                    for img_file in sorted(folder.glob("*.jpg"))[:5]:  # Limit to 5 per class
                        sample_options[f"{label}: {img_file.name}"] = img_file
        else:
            st.warning("Dataset not found. Place dataset under `data/raw/` or use Upload.")

        if sample_options:
            selected = st.selectbox(
                "Select verified dataset sample",
                options=list(sample_options.keys()),
                label_visibility="collapsed",
            )
            sample_path = sample_options[selected]
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                image_name = sample_path.name
                st.session_state["image_bytes"] = image_bytes
                st.session_state["image_name"] = image_name
            else:
                st.error(f"Sample not found: {sample_path}")

    else:
        uploaded = st.file_uploader(
            "Upload handwriting crop (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            # Validate file size (max 10MB)
            if uploaded.size > 10 * 1024 * 1024:
                st.error("File too large. Maximum size is 10 MB.")
                uploaded = None
            else:
                image_bytes = uploaded.getvalue()
                image_name = uploaded.name
                st.session_state["image_bytes"] = image_bytes
                st.session_state["image_name"] = image_name

    # Polarity control
    st.markdown("---")
    polarity_choice = st.selectbox(
        "Background polarity",
        options=["auto", "dark_background", "light_background"],
        format_func=lambda x: {
            "auto": "Auto-detect (recommended)",
            "dark_background": "Dark background (dataset standard)",
            "light_background": "Light background (paper photo)",
        }[x],
        help="Dataset images have dark background with light ink. Paper photos have light background with dark ink.",
    )

    return image_bytes, image_name, polarity_choice


def render_preprocessing_preview(image_bytes, polarity_choice) -> bool:
    """Render preprocessing preview and return True if preprocessed successfully."""
    from app.pipeline_handler import load_and_preprocess_image

    if image_bytes is None:
        return False

    # Check if we need to re-preprocess
    need_preprocess = (
        st.session_state.get("analyzed_image_name") != st.session_state.get("image_name")
        or st.session_state.get("analyzed_polarity") != polarity_choice
        or "preprocessed" not in st.session_state
    )

    if need_preprocess:
        st.session_state.pop("inference_result", None)

    try:
        with st.spinner("Preprocessing..."):
            rgb_preview, gray_preview, mask_preview, is_light_detected = load_and_preprocess_image(
                image_bytes, polarity=polarity_choice
            )
        st.session_state["preprocessed"] = {
            "rgb": rgb_preview,
            "gray": gray_preview,
            "mask": mask_preview,
            "is_light": is_light_detected,
        }
        h, w = rgb_preview.shape[:2]
        mode_desc = "Light background" if is_light_detected else "Dark background"
        st.caption(f"**Loaded**: `{st.session_state.get('image_name', 'sample.png')}`  |  `{w}×{h}` px  |  Detected: `{mode_desc}`")
        return True
    except Exception as e:
        st.error(f"Preprocessing failed: {type(e).__name__} — {e}")
        st.session_state.pop("preprocessed", None)
        return False


def render_empty_state():
    """Render empty state when no image is selected."""
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">✍️</div>
            <div class="empty-state-title">No image selected</div>
            <div class="empty-state-text">
                Choose a sample image or upload your own handwriting crop to begin analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )