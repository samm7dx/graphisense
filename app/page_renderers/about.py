"""About Dysgraphia page for GraphiSense."""

import streamlit as st

from app.page_renderers.methodology import render_about_dysgraphia_page


def render_about_page_wrapper():
    """Wrapper for about page."""
    render_about_dysgraphia_page()