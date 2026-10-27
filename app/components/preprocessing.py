"""Preprocessing explorer component for GraphiSense."""

import streamlit as st
import cv2
import numpy as np

from app.services import get_preprocessing_stages, compute_preprocessing_stages


def render_preprocessing_page():
    """Render the standalone Preprocessing page."""
    st.markdown("## Preprocessing Pipeline")
    st.markdown("""
    The preprocessing pipeline consists of 6 stages. Each stage transforms the image for the next step in the pipeline.
    """)

    # Show current preprocessed images if available
    preprocessed = st.session_state.get("preprocessed")
    if preprocessed is not None:
        st.divider()
        st.markdown("### Current Image Preprocessing Stages")

        stage_defs = get_preprocessing_stages()

        # Get the preprocessed images
        rgb_img = preprocessed["rgb"]
        gray_img = preprocessed["gray"]
        binary_mask = preprocessed["mask"]
        is_light = preprocessed["is_light"]
        polarity = st.session_state.get("analyzed_polarity", "auto")

        # Compute stages if not cached
        if "preproc_stages" not in st.session_state:
            with st.spinner("Computing preprocessing stages..."):
                stages_dict = compute_preprocessing_stages(preprocessed["gray"], preprocessed["mask"], polarity)
                st.session_state["preproc_stages"] = stages_dict
        else:
            stages_dict = st.session_state["preproc_stages"]

        all_stages = {
            "original_rgb": preprocessed["rgb"],
            "grayscale": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB),
            "binary_mask": cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB),
            **stages_dict,
        }

        # Create tabs for the 6 stages
        stage_names = [s["name"] for s in stage_defs]
        tabs = st.tabs(stage_names)

        for idx, (tab, stage_def) in enumerate(zip(tabs, stage_defs)):
            with tab:
                stage_key = stage_def["key"]
                if stage_key in all_stages:
                    stage_image = all_stages[stage_key]
                    stage_h, stage_w = stage_image.shape[:2]

                    st.markdown(f"#### Stage {stage_def['id']}: {stage_def['name']}")
                    st.markdown(f"*{stage_def['description']}*")
                    st.caption(f"Dimensions: {stage_w}×{stage_h} px")

                    st.image(stage_image, width="stretch", clamp=True)

                    # View button for full-size dialog
                    if st.button(
                        "View Full Size",
                        key=f"view_stage_full_{stage_def['id']}",
                        type="secondary",
                    ):
                        st.session_state["preproc_viewer_stage"] = stage_def["id"]
                        st.session_state["preproc_viewer_zoom"] = 1.0
                        st.rerun()
                else:
                    st.warning(f"Stage {stage_def['id']} not available")

        # Full-size image viewer dialog
        if st.session_state.get("preproc_viewer_stage") is not None:
            viewer_stage_id = st.session_state["preproc_viewer_stage"]
            viewer_stage_def = next((s for s in stage_defs if s["id"] == viewer_stage_id), None)

            if viewer_stage_def and viewer_stage_def["key"] in all_stages:
                viewer_image = all_stages[viewer_stage_def["key"]]

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
                            rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                            if rot_key not in st.session_state:
                                st.session_state[rot_key] = 0
                            st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
                            st.rerun()

                    # Display the image
                    zoom = st.session_state["preproc_viewer_zoom"]
                    rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                    rotation = st.session_state.get(rot_key, 0)

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

                    if zoom != 1.0:
                        new_w = int(disp_img.shape[1] * zoom)
                        new_h = int(disp_img.shape[0] * zoom)
                        disp_img = cv2.resize(disp_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

                    st.image(disp_img, clamp=True, width="stretch")

                    if st.button("Close Viewer", type="primary", use_container_width=True):
                        st.session_state["preproc_viewer_stage"] = None
                        st.rerun()

                show_viewer()
    else:
        st.info("Run an analysis first to populate the preprocessing explorer with your image.")
        st.caption("Navigate to the **Modelling** section, upload an image, and run analysis to see the preprocessing stages.")

        # Still show the pipeline description
        st.markdown("---")
        st.markdown("### Pipeline Stage Descriptions")
        stage_defs = get_preprocessing_stages()
        for stage in stage_defs:
            with st.expander(f"Stage {stage['id']}: {stage['name']}"):
                st.markdown(stage['description'])