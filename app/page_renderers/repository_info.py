"""Repository Info page for GraphiSense."""

import streamlit as st

from app.page_renderers.methodology import render_repository_info_page


def render_repository_info_page_wrapper():
    """Wrapper for repository info page."""
    render_repository_info_page()