"""GraphiSense Streamlit Web Application.

A professional, visualization-first interface for handwriting image preprocessing,
feature extraction, and dysgraphia screening-support model inference.

Two analysis modes:
- Basic Analysis: Random Forest on 9 static handwriting features
- Advanced Analysis: CNN/ResNet (coming soon)
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Import page renderers
from app.page_renderers import (
    render_modelling_page,
    render_methodology_page,
    render_about_page_wrapper,
    render_comparison_page_wrapper,
    render_preprocessing_page_wrapper,
    render_repository_info_page_wrapper,
)

# Import navigation component
from app.components.navigation import render_sidebar


# =============================================================================
# Page Configuration & Styling
# =============================================================================
st.set_page_config(
    page_title="GraphiSense | Handwriting Analysis",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS with design system
st.markdown(
    """
    <style>
    /* Design tokens */
    :root {
        --color-primary: #0284C7;
        --color-primary-hover: #0369A1;
        --color-primary-light: #E0F2FE;
        --color-success: #16A34A;
        --color-success-light: #F0FDF4;
        --color-warning: #D97706;
        --color-warning-light: #FEF3C7;
        --color-danger: #DC2626;
        --color-danger-light: #FEF2F2;
        --color-text: #0F172A;
        --color-text-muted: #64748B;
        --color-border: #E2E8F0;
        --color-bg: #F8FAFC;
        --color-surface: #FFFFFF;
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
    }

    /* Global container */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header bar */
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 16px 0;
        border-bottom: 1px solid var(--color-border);
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .header-left {
        display: flex;
        align-items: baseline;
        gap: 12px;
        flex-wrap: wrap;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--color-text);
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 1rem;
        color: var(--color-text-muted);
        font-weight: 400;
        margin: 0;
    }
    .prototype-badge {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #0369A1;
        background-color: var(--color-primary-light);
        padding: 4px 10px;
        border-radius: 999px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .mode-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .mode-badge-basic { background: var(--color-success-light); color: var(--color-success); }
    .mode-badge-advanced { background: var(--color-warning-light); color: var(--color-warning); }

    /* Cards */
    .card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 20px;
        box-shadow: var(--shadow-sm);
    }
    .card-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-text);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction cards */
    .pred-card {
        padding: 16px 20px;
        border-radius: var(--radius-md);
        margin-bottom: 16px;
        font-size: 1rem;
        line-height: 1.5;
        border-left: 4px solid;
    }
    .pred-card-pd {
        background-color: var(--color-warning-light);
        color: #92400E;
        border-left-color: var(--color-warning);
    }
    .pred-card-lpd {
        background-color: var(--color-success-light);
        color: #166534;
        border-left-color: var(--color-success);
    }

    /* Probability bars */
    .prob-container {
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 16px;
        margin-bottom: 16px;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.875rem;
        margin-bottom: 8px;
    }
    .prob-bar-bg {
        flex: 1;
        height: 8px;
        background: var(--color-border);
        border-radius: 4px;
        overflow: hidden;
        margin: 0 12px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .prob-bar-pd { background: linear-gradient(90deg, #FCD34D, #F59E0B); }
    .prob-bar-lpd { background: linear-gradient(90deg, #86EFAC, #16A34A); }

    /* Metric cards */
    .metric-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .empty-state-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--color-text);
        margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 0.875rem;
        color: var(--color-text-muted);
        line-height: 1.5;
    }

    /* Progress steps */
    .progress-step {
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 12px 8px;
        text-align: center;
        min-height: 72px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    .progress-step.active {
        background: var(--color-primary-light);
        border-color: var(--color-primary);
    }
    .progress-step.complete {
        background: var(--color-success-light);
        border-color: var(--color-success);
        color: var(--color-success);
    }
    .progress-step-icon {
        width: 24px; height: 24px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 600;
    }
    .progress-step.active .progress-step-icon { background: var(--color-primary); color: white; }
    .progress-step.complete .progress-step-icon { background: var(--color-success); color: white; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 10px 20px;
        font-weight: 500;
        color: var(--color-text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--color-primary);
        border-color: var(--color-primary);
        color: white !important;
    }

    /* Sidebar */
    .sidebar-section {
        padding: 16px 0;
        border-bottom: 1px solid var(--color-border);
    }
    .sidebar-section:last-child { border-bottom: none; }

    /* Footer */
    .footer-note {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-align: center;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid var(--color-border);
    }

    /* Image caption */
    .image-caption {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-align: center;
        margin-top: 8px;
        padding: 4px;
    }

    /* Expander */
    .stExpander { border: 1px solid var(--color-border); border-radius: var(--radius-md); }
    .stExpander summary { font-weight: 500; }

    /* Dataframe */
    .stDataFrame { border: 1px solid var(--color-border); border-radius: var(--radius-md); }

    /* Button */
    .stButton > button[kind="primary"] {
        background-color: var(--color-primary) !important;
        border-color: var(--color-primary) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        border-radius: var(--radius-md) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--color-primary-hover) !important;
        border-color: var(--color-primary-hover) !important;
    }

    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed var(--color-border);
        border-radius: var(--radius-md);
        padding: 16px 20px !important;
        min-height: 80px !important;
    }
    .stFileUploader > div:hover { border-color: var(--color-primary); }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        padding: 8px 0 !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] p {
        margin: 4px 0 !important;
        font-size: 0.875rem !important;
    }

    /* Radio */
    .stRadio > div { gap: 16px; }

    /* Selectbox */
    .stSelectbox > div > div { border-radius: var(--radius-md); }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-badge.status-ready {
        background: var(--color-success-light);
        color: var(--color-success);
    }
    .status-badge.status-missing {
        background: var(--color-danger-light);
        color: var(--color-danger);
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
    }
    .status-ready .status-dot { background: var(--color-success); }
    .status-missing .status-dot { background: var(--color-danger); }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""

    # Render sidebar and get selected navigation
    selected_nav = render_sidebar()

    # Header
    st.markdown(
        f"""
        <div class="header-bar">
            <div class="header-left">
                <span class="app-title">GraphiSense</span>
                <span class="app-subtitle">Computer vision-based handwriting analysis</span>
            </div>
            <span class="prototype-badge">Research Prototype</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to the appropriate page
    if selected_nav == "🏠 Modelling":
        render_modelling_page()
    elif selected_nav == "🔬 Methodology":
        render_methodology_page()
    elif selected_nav == "⚖️ RF vs CNN Comparison":
        render_comparison_page_wrapper()
    elif selected_nav == "⚙️ Preprocessing":
        render_preprocessing_page_wrapper()
    elif selected_nav == "📁 Repository Info":
        render_repository_info_page_wrapper()
    elif selected_nav == "ℹ️ About Dysgraphia":
        render_about_page_wrapper()


if __name__ == "__main__":
    main()