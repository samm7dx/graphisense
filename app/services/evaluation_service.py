"""Evaluation metrics loading service for GraphiSense."""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Optional[pd.DataFrame]:
    """Load cross-validation evaluation metrics."""
    try:
        return pd.read_csv(ROOT / "outputs" / "reports" / "model_evaluation_metrics.csv")
    except Exception:
        return None