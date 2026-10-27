"""Feature dataset loading service for GraphiSense."""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV_PATH = ROOT / "outputs" / "features" / "features.csv"


@st.cache_data(show_spinner=False)
def load_features_dataset() -> Optional[pd.DataFrame]:
    """Load the features CSV for distribution charts."""
    try:
        return pd.read_csv(FEATURES_CSV_PATH)
    except Exception:
        return None