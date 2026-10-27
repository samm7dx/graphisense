"""Model availability and metadata service for GraphiSense."""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "models" / "model_pipeline.joblib"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.json"


@st.cache_data(show_spinner=False)
def load_model_metadata() -> Optional[Dict[str, Any]]:
    """Load model metadata with caching."""
    try:
        import json
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_model_availability() -> Tuple[bool, str]:
    """Check if model artifacts exist and are loadable."""
    if not MODEL_PATH.exists():
        return False, "Model artifact not found"
    if not METADATA_PATH.exists():
        return False, "Model metadata not found"
    try:
        meta = load_model_metadata()
        if meta is None:
            return False, "Failed to load metadata"
        required = {"model_type", "feature_names", "class_mapping"}
        if not required.issubset(meta.keys()):
            return False, "Metadata schema invalid"
        return True, "Ready"
    except Exception as e:
        return False, f"Model validation failed: {e}"