"""Preprocessing page for GraphiSense."""

import streamlit as st

from app.components.preprocessing import render_preprocessing_page


def render_preprocessing_page_wrapper():
    """Wrapper for preprocessing page."""
    render_preprocessing_page()