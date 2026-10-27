"""RF vs CNN Comparison page for GraphiSense."""

import streamlit as st

from app.page_renderers.methodology import render_comparison_page


def render_comparison_page_wrapper():
    """Wrapper for comparison page."""
    render_comparison_page()