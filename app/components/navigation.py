"""Navigation component for GraphiSense sidebar."""

import streamlit as st


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected section."""
    with st.sidebar:
        # Main Navigation - 6 sections
        st.markdown("### Navigation")

        nav_options = [
            "🏠 Modelling",  # Default landing page - upload & predict
            "🔬 Methodology",
            "⚖️ RF vs CNN Comparison",
            "⚙️ Preprocessing",
            "📁 Repository Info",
            "ℹ️ About Dysgraphia",
        ]

        selected_nav = st.radio(
            "Select section",
            options=nav_options,
            index=0,
            label_visibility="collapsed"
        )

    return selected_nav