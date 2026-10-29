"""Modelling page - Main landing page with upload and prediction workspace."""

import streamlit as st

from app.services import (
    check_model_availability,
    check_cnn_availability,
    run_rf_inference,
    run_cnn_inference,
    load_model_metadata,
)
from app.components.upload import render_upload_section, render_preprocessing_preview, render_empty_state
from app.components.prediction_results import render_prediction_results, render_preprocessing_preview_while_waiting
from app.components.model_details import (
    render_rf_feature_analysis_tab,
    render_cnn_model_architecture_tab,
    render_cnn_training_history_tab,
    render_cnn_evaluation_tab,
    render_image_processing_tab,
    render_rf_model_details_tab,
    render_limitations_tab,
)
from app.components.navigation import render_sidebar


def render_modelling_page():
    """Render the Modelling page - the main upload and prediction workspace."""
    # Model selection
    cnn_available, cnn_msg = check_cnn_availability()
    model_available, model_msg = check_model_availability()

    model_choice = st.radio(
        "Select Model",
        ["🌲 Random Forest (Basic Analysis)", "🧠 CNN/ResNet-18 (Advanced Analysis)"],
        horizontal=True,
        key="model_choice"
    )

    is_cnn = "CNN" in model_choice

    if is_cnn and not cnn_available:
        st.error(f"⚠️ **CNN model not available.** {cnn_msg}")
        st.info("""
        To enable CNN analysis:
        1. Ensure the dataset is placed under `data/raw/` with `Potential Dysgraphia/` and `Low Potential Dysgraphia/` folders
        2. Run training: `python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn`
        3. The best checkpoint will be saved to `outputs/models/cnn/checkpoints/best_model.pt`
        """)
        st.stop()

    # Show model availability
    if is_cnn:
        if cnn_available:
            st.success(f"✅ CNN model loaded: {cnn_msg}")
        else:
            st.error("CNN model not available")
    else:
        if model_available:
            st.success(f"✅ Random Forest model loaded")
        else:
            st.warning("Random Forest model not available")

    st.divider()

    # Input section (same for both models)
    image_bytes, image_name, polarity_choice = render_upload_section()

    # Preprocessing preview
    preprocessed = None
    if image_bytes is not None:
        preprocessed = render_preprocessing_preview(image_bytes, polarity_choice)

    # Analyze button
    if image_bytes is not None and "preprocessed" in st.session_state:
        if is_cnn:
            button_label = "Analyze handwriting (CNN/ResNet)"
            can_analyze = cnn_available
            error_msg = "Cannot analyze: CNN model not available. Train a CNN model first."
        else:
            button_label = "Analyze handwriting (Random Forest)"
            can_analyze = model_available
            error_msg = "Cannot analyze: Random Forest model not available. Complete training steps in README.md first."

        if st.button(button_label, type="primary", width="stretch"):
            if not can_analyze:
                st.error(error_msg)
            else:
                st.session_state.pop("inference_result", None)
                with st.spinner(f"Running {'CNN/ResNet' if is_cnn else 'Random Forest'} inference..."):
                    try:
                        if is_cnn:
                            result = run_cnn_inference(
                                image_input=image_bytes,
                                filename=image_name,
                            )
                        else:
                            result = run_rf_inference(
                                image_input=image_bytes,
                                filename=image_name,
                                polarity=polarity_choice,
                            )
                        st.session_state["inference_result"] = result
                        st.session_state["analyzed_image_name"] = image_name
                        st.session_state["analyzed_polarity"] = polarity_choice
                        st.rerun()
                    except FileNotFoundError as e:
                        st.error(f"Required file not found: {e}")
                    except ValueError as e:
                        st.error(f"Input validation error: {e}")
                    except Exception as e:
                        st.error(f"Inference failed: {type(e).__name__} — {e}")

    elif image_bytes is None:
        render_empty_state()

    # Results area
    result = st.session_state.get("inference_result")
    preprocessed = st.session_state.get("preprocessed")

    if result is None:
        if preprocessed is not None:
            render_preprocessing_preview_while_waiting(preprocessed)
    else:
        # Full results display
        features_used, all_features, meta = render_prediction_results(result)

        st.divider()

        if is_cnn:
            # CNN Results - use expanders instead of tabs
            with st.expander("🖼️ Image Processing", expanded=True):
                render_image_processing_tab(result, is_cnn=True)

            with st.expander("🤖 Model Architecture", expanded=False):
                render_cnn_model_architecture_tab()

            with st.expander("📈 Training History", expanded=False):
                render_cnn_training_history_tab()

            with st.expander("📊 Evaluation", expanded=False):
                render_cnn_evaluation_tab()

            with st.expander("⚠️ Limitations", expanded=False):
                render_limitations_tab(is_cnn=True)
        else:
            # RF Results - use expanders instead of tabs
            with st.expander("📊 Feature Analysis", expanded=True):
                render_rf_feature_analysis_tab(features_used, all_features, meta)

            with st.expander("🖼️ Image Processing", expanded=False):
                render_image_processing_tab(result, is_cnn=False)

            with st.expander("🔬 Preprocessing Explorer", expanded=False):
                render_preprocessing_explorer_tab()

            with st.expander("🤖 Model Details", expanded=False):
                render_rf_model_details_tab(meta)

            with st.expander("⚠️ Limitations", expanded=False):
                render_limitations_tab(is_cnn=False)


def render_preprocessing_explorer_tab():
    """Render the preprocessing explorer as a standalone section (not a tab)."""
    from app.services import get_preprocessing_stages, compute_preprocessing_stages
    import cv2
    
    st.markdown("##### Interactive Preprocessing Explorer")
    st.caption("Step through the OpenCV preprocessing pipeline. Each stage shows the actual intermediate image produced by the pipeline.")

    preprocessed = st.session_state.get("preprocessed")
    result = st.session_state.get("inference_result")
    if preprocessed is not None and result is not None:
        # Initialize session state for viewer
        if "preproc_viewer_stage" not in st.session_state:
            st.session_state["preproc_viewer_stage"] = None
        if "preproc_viewer_zoom" not in st.session_state:
            st.session_state["preproc_viewer_zoom"] = 1.0

        # Get the original images
        rgb_img = preprocessed["rgb"]
        gray_img = preprocessed["gray"]
        binary_mask = preprocessed["mask"]
        is_light = preprocessed["is_light"]
        polarity = result.get("resolved_polarity", "dark_background")
        orig_h, orig_w = rgb_img.shape[:2]

        # Compute all stages if not cached
        if "preproc_stages" not in st.session_state:
            with st.spinner("Computing preprocessing stages..."):
                stages_dict = compute_preprocessing_stages(gray_img, binary_mask, polarity)
                st.session_state["preproc_stages"] = stages_dict
        else:
            stages_dict = st.session_state["preproc_stages"]

        # Combine with base images
        stage_defs = get_preprocessing_stages()
        import cv2
        all_stages = {
            "original_rgb": rgb_img,
            "grayscale": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB),
            "binary_mask": cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB),
            **stages_dict,
        }

        stage_defs = get_preprocessing_stages()

        # Thumbnail gallery - compact grid of cards
        st.markdown("**Pipeline Stages**")
        st.caption("Click **View** on any stage to open the full-size image viewer.")

        # Create grid: 3 columns for 6 stages = 2 rows
        cols_per_row = 3
        num_stages = len(stage_defs)

        for row_idx in range(0, num_stages, cols_per_row):
            row_stages = stage_defs[row_idx:row_idx + cols_per_row]
            cols = st.columns(len(row_stages), gap="medium")

            for col_idx, stage_def in enumerate(row_stages):
                with cols[col_idx]:
                    stage_key = stage_def["key"]
                    if stage_key in all_stages:
                        stage_image = all_stages[stage_key]
                        stage_h, stage_w = stage_image.shape[:2]

                        # Card container
                        with st.container(border=True):
                            # Stage number and name
                            st.markdown(
                                f"""
                                <div style="text-align: center; margin-bottom: 8px;">
                                    <span style="
                                        background: var(--color-primary-light);
                                        color: var(--color-primary);
                                        padding: 2px 8px;
                                        border-radius: 999px;
                                        font-size: 0.7rem;
                                        font-weight: 600;
                                    ">Stage {stage_def['id']}</span>
                                    <div style="font-weight: 500; font-size: 0.85rem; margin-top: 4px; color: var(--color-text);">
                                        {stage_def['name']}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Thumbnail image - fixed height container with contained image
                            thumb_h = 180
                            aspect = stage_w / stage_h if stage_h > 0 else 1.0
                            thumb_w = int(thumb_h * aspect)
                            thumb_w = min(thumb_w, 300)  # Cap width

                            # Display thumbnail centered
                            left_pad, center, right_pad = st.columns([1, 10, 1])
                            with center:
                                st.image(stage_image, width=thumb_w, clamp=True)

                            # Image dimensions
                            st.markdown(
                                f'<div style="text-align: center; font-size: 0.7rem; color: var(--color-text-muted); margin-top: 4px;">{stage_w}×{stage_h} px</div>',
                                unsafe_allow_html=True,
                            )

                            # View button
                            if st.button(
                                "View",
                                key=f"view_stage_{stage_def['id']}",
                                use_container_width=True,
                                type="secondary",
                            ):
                                st.session_state["preproc_viewer_stage"] = stage_def["id"]
                                st.session_state["preproc_viewer_zoom"] = 1.0
                                st.rerun()
                    else:
                        # Stage not available
                        with st.container(border=True):
                            st.markdown(
                                f"""
                                <div style="text-align: center; padding: 20px;">
                                    <span style="
                                        background: var(--color-warning-light);
                                        color: var(--color-warning);
                                        padding: 2px 8px;
                                        border-radius: 999px;
                                        font-size: 0.7rem;
                                        font-weight: 600;
                                    ">Stage {stage_def['id']}</span>
                                    <div style="font-weight: 500; font-size: 0.85rem; margin-top: 4px;">{stage_def['name']}</div>
                                    <div style="font-size: 0.7rem; color: var(--color-text-muted); margin-top: 8px;">Not available</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Image viewer dialog
        if st.session_state.get("preproc_viewer_stage") is not None:
            viewer_stage_id = st.session_state["preproc_viewer_stage"]
            viewer_stage_def = next((s for s in stage_defs if s["id"] == viewer_stage_id), None)

            if viewer_stage_def and viewer_stage_def["key"] in all_stages:
                viewer_image = all_stages[viewer_stage_def["key"]]
                viewer_h, viewer_w = viewer_image.shape[:2]

                @st.dialog(f"Stage {viewer_stage_id}: {viewer_stage_def['name']}", width="large")
                def show_viewer():
                    # Viewer controls
                    ctrl_cols = st.columns([1, 1, 1, 1, 2], gap="small")

                    with ctrl_cols[0]:
                        if st.button("Zoom In", use_container_width=True):
                            st.session_state["preproc_viewer_zoom"] = min(
                                st.session_state["preproc_viewer_zoom"] * 1.25, 5.0
                            )
                            st.rerun()

                    with ctrl_cols[1]:
                        if st.button("Zoom Out", use_container_width=True):
                            st.session_state["preproc_viewer_zoom"] = max(
                                st.session_state["preproc_viewer_zoom"] / 1.25, 0.25
                            )
                            st.rerun()

                    with ctrl_cols[2]:
                        if st.button("Reset Zoom", use_container_width=True):
                            st.session_state["preproc_viewer_zoom"] = 1.0
                            st.rerun()

                    with ctrl_cols[3]:
                        if st.button("Rotate 90°", use_container_width=True):
                            # Store rotation state
                            rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                            if rot_key not in st.session_state:
                                st.session_state[rot_key] = 0
                            st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
                            st.rerun()

                    # Display the image
                    zoom = st.session_state["preproc_viewer_zoom"]
                    rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                    rotation = st.session_state.get(rot_key, 0)

                    # Apply rotation if needed
                    if rotation > 0:
                        if rotation == 90:
                            disp_img = cv2.rotate(viewer_image, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation == 180:
                            disp_img = cv2.rotate(viewer_image, cv2.ROTATE_180)
                        elif rotation == 270:
                            disp_img = cv2.rotate(viewer_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            disp_img = viewer_image
                    else:
                        disp_img = viewer_image

                    # Apply zoom
                    if zoom != 1.0:
                        new_w = int(disp_img.shape[1] * zoom)
                        new_h = int(disp_img.shape[0] * zoom)
                        disp_img = cv2.resize(disp_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Image metadata
                    st.markdown(
                        f"""
                        <div style="
                            background: var(--color-bg);
                            border: 1px solid var(--color-border);
                            border-radius: var(--radius-md);
                            padding: 12px;
                            margin-bottom: 12px;
                            font-size: 0.85rem;
                        ">
                            <strong>Dimensions:</strong> {disp_img.shape[1]}×{disp_img.shape[0]} px
                            (×{zoom:.2f} zoom{f', {rotation}° rotation' if rotation else ''}) |
                            <strong>Stage:</strong> {viewer_stage_def['name']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Display image - use column to center and allow scrolling if needed
                    st.image(disp_img, clamp=True, width="stretch")

                    # Close button
                    if st.button("Close Viewer", type="primary", use_container_width=True):
                        st.session_state["preproc_viewer_stage"] = None
                        st.rerun()

                show_viewer()
            else:
                st.warning("Selected stage not available in current pipeline.")

    else:
        st.info("Run an analysis first to populate the preprocessing explorer with your image.")