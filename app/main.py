"""GraphiSense Streamlit Web Application.

A professional, visualization-first interface for handwriting image preprocessing,
feature extraction, and dysgraphia screening-support model inference.

Two analysis modes:
- Basic Analysis: Random Forest on 9 static handwriting features
- Advanced Analysis: CNN/ResNet (coming soon)
"""

from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional, List, Tuple, Union

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import torch
except ImportError:
    torch = None

from app.pipeline_handler import load_and_preprocess_image, run_app_inference

DATASET_ROOT = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "outputs" / "models" / "model_pipeline.joblib"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.json"
FEATURES_CSV_PATH = ROOT / "outputs" / "features" / "features.csv"

# CNN paths
CNN_CHECKPOINT_DIR = ROOT / "outputs" / "models" / "cnn" / "checkpoints"
CNN_CHECKPOINT_PATH = CNN_CHECKPOINT_DIR / "best_model.pt"
CNN_EVAL_DIR = ROOT / "outputs" / "reports" / "cnn_evaluation"


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
    .prob-label { font-weight: 500; color: var(--color-text); }
    .prob-value { font-weight: 600; color: var(--color-text); }
    .prob-track {
        height: 10px;
        width: 100%;
        background-color: var(--color-border);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 6px;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease-out;
    }
    .prob-fill-pd { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .prob-fill-lpd { background: linear-gradient(90deg, #34D399, #16A34A); }

    /* Metric cards */
    .metric-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--color-text);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 999px;
    }
    .status-ready { background: var(--color-success-light); color: var(--color-success); }
    .status-missing { background: var(--color-danger-light); color: var(--color-danger); }
    .status-coming { background: var(--color-warning-light); color: var(--color-warning); }

    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
    }
    .status-ready .status-dot { background: var(--color-success); }
    .status-missing .status-dot { background: var(--color-danger); }
    .status-coming .status-dot { background: var(--color-warning); }

    /* Empty state */
    .empty-state {
        border: 1px dashed var(--color-border);
        background-color: var(--color-bg);
        border-radius: var(--radius-lg);
        padding: 48px 24px;
        text-align: center;
        color: var(--color-text-muted);
    }
    .empty-state-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.5; }
    .empty-state-title { font-size: 1.125rem; font-weight: 600; color: var(--color-text); margin-bottom: 8px; }
    .empty-state-text { font-size: 0.9rem; margin-bottom: 16px; }

    /* Progress steps */
    .progress-steps {
        display: flex;
        gap: 8px;
        margin-bottom: 24px;
    }
    .progress-step {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 12px;
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        font-size: 0.75rem;
        color: var(--color-text-muted);
        position: relative;
    }
    .progress-step.active {
        background: var(--color-primary-light);
        border-color: var(--color-primary);
        color: var(--color-primary);
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
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions
# =============================================================================
def get_preprocessing_stages() -> List[Dict[str, Any]]:
    """Define the preprocessing pipeline stages (6 stages, no Medial Axis Ridges)."""
    return [
        {
            "id": 1,
            "name": "Original Image",
            "key": "original_rgb",
            "description": "Input image as loaded (RGB). No processing applied.",
            "available": True,
        },
        {
            "id": 2,
            "name": "Grayscale",
            "key": "grayscale",
            "description": "Single-channel luminance. OpenCV BGR → GRAY conversion (ITU-R BT.601 weights).",
            "available": True,
        },
        {
            "id": 3,
            "name": "Binary Mask (Otsu)",
            "key": "binary_mask",
            "description": "Foreground/background separation via Otsu thresholding with polarity detection. Handwriting=255, Background=0.",
            "available": True,
        },
        {
            "id": 4,
            "name": "Distance Transform",
            "key": "distance_transform",
            "description": "Euclidean distance from each foreground pixel to nearest background. Brighter = further from edge (stroke center).",
            "available": True,
        },
        {
            "id": 5,
            "name": "Connected Components",
            "key": "connected_components",
            "description": "8-connected component analysis with noise filtering (min area=5px²). Bounding boxes shown in color overlay.",
            "available": True,
        },
        {
            "id": 6,
            "name": "Row Projection",
            "key": "row_projection",
            "description": "Horizontal ink distribution (sum of foreground pixels per row). Reveals writing line structure and spacing.",
            "available": True,
        },
    ]


def compute_preprocessing_stages(
    gray: np.ndarray,
    binary_mask: np.ndarray,
    polarity: str,
) -> Dict[str, np.ndarray]:
    """Compute all intermediate preprocessing stage images for UI display."""
    stages = {}

    # Stage 4: Distance Transform
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_INFERNO)
    stages["distance_transform"] = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB)

    # Stage 5: Connected Components with bounding boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        cc_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        # Filter by min area
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = areas >= 5
        for i, (stat, centroid) in enumerate(zip(stats[1:], centroids[1:])):
            if not valid[i]:
                continue
            x, y, w, h, area = stat
            cx, cy = centroid
            # Draw bounding box
            cv2.rectangle(cc_vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Draw centroid
            cv2.circle(cc_vis, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        stages["connected_components"] = cc_vis
    else:
        stages["connected_components"] = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

    # Stage 6: Row Projection visualization
    row_proj = np.sum(binary_mask == 255, axis=1).astype(np.float64)
    if len(row_proj) > 0:
        max_proj = max(row_proj.max(), 1)
        proj_vis = np.zeros((binary_mask.shape[0], 200, 3), dtype=np.uint8)
        for i, val in enumerate(row_proj):
            if val > 0:
                width = int((val / max_proj) * 180)
                cv2.line(proj_vis, (10, i), (10 + width, i), (0, 255, 0), 1)
        stages["row_projection"] = proj_vis
    else:
        stages["row_projection"] = np.zeros((100, 200, 3), dtype=np.uint8)

    return stages


def resize_for_display(image: np.ndarray, display_width: int, mode: str, original_width: int, original_height: int) -> np.ndarray:
    """Resize image for display without mutating original."""
    if mode == "Native Size":
        return image
    elif mode == "Fit to Container":
        return image  # Streamlit handles this with width="stretch"
    else:
        # Custom width - calculate height to preserve aspect ratio
        aspect_ratio = original_height / original_width if original_width > 0 else 1.0
        display_height = int(display_width * aspect_ratio)
        return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> Optional[Dict[str, Any]]:
    """Load model metadata with caching."""
    try:
        import json
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_features_dataset() -> Optional[pd.DataFrame]:
    """Load the features CSV for distribution charts."""
    try:
        return pd.read_csv(FEATURES_CSV_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Optional[pd.DataFrame]:
    """Load cross-validation evaluation metrics."""
    try:
        return pd.read_csv(ROOT / "outputs" / "reports" / "model_evaluation_metrics.csv")
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


@st.cache_data(show_spinner=False)
def check_cnn_availability() -> Tuple[bool, str]:
    """Check if CNN model checkpoint is available."""
    if not CNN_CHECKPOINT_PATH.exists():
        return False, "CNN checkpoint not found"
    try:
        import torch
        checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            return False, "Invalid checkpoint format"
        if checkpoint.get("class_mapping") != {"0": "LPD", "1": "PD"}:
            return False, "Incompatible class mapping"
        return True, "CNN model ready"
    except Exception as e:
        return False, f"Checkpoint validation failed: {e}"


@st.cache_data(show_spinner=False)
def load_cnn_metadata() -> Optional[Dict[str, Any]]:
    """Load CNN model metadata if available."""
    if not CNN_CHECKPOINT_PATH.exists():
        return None
    try:
        import torch
        checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        val_metrics = checkpoint.get("val_metrics", {})
        epoch = checkpoint.get("epoch", 0)
        training_history = checkpoint.get("training_history", {})

        return {
            "model_type": "ResNet-18",
            "pretrained": config.get("pretrained", True),
            "image_size": config.get("image_size", 224),
            "dropout": config.get("dropout", 0.3),
            "freeze_backbone": config.get("freeze_backbone", False),
            "epoch": epoch,
            "val_loss": checkpoint.get("val_loss"),
            "val_metrics": val_metrics,
            "training_epochs": len(training_history.get("train_loss", [])),
            "class_mapping": {"0": "LPD", "1": "PD"},
        }
    except Exception:
        return None


def run_cnn_inference(
    image_input: Union[bytes, str, Path],
    filename: str = "sample.png",
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Execute CNN inference and package results for display.

    Parameters
    ----------
    image_input : Union[bytes, str, Path]
        Raw image bytes or filesystem path.
    filename : str
        Source filename for extension and identification.
    checkpoint_path : Optional[Union[str, Path]]
        Optional custom checkpoint path.
    device : Optional[Union[str, torch.device]]
        Optional device for inference.

    Returns
    -------
    Dict[str, Any]
        Inference result dictionary with preview arrays attached.
    """
    if checkpoint_path is None:
        checkpoint_path = CNN_CHECKPOINT_PATH

    # Create a temporary file for the CNN inference engine
    import tempfile
    if isinstance(image_input, bytes):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(
                f"Unsupported upload file extension '{suffix}'. "
                "Upload a JPG, JPEG, or PNG image."
            )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(image_input)
            temp_path = Path(tf.name)
        eval_path = temp_path
    else:
        eval_path = Path(image_input)

    try:
        # Run CNN inference
        from src.models.cnn_inference import CNNInference
        engine = CNNInference(checkpoint_path, device=device)
        result = engine.predict(eval_path)

        # Also get the preprocessed images for visualization
        from app.pipeline_handler import load_and_preprocess_image
        if isinstance(image_input, bytes):
            rgb, gray, binary_mask, is_light = load_and_preprocess_image(image_input, polarity="auto")
        else:
            rgb, gray, binary_mask, is_light = load_and_preprocess_image(eval_path, polarity="auto")

        # Package result in same format as RF inference for consistency
        result["original_rgb"] = rgb
        result["grayscale"] = gray
        result["binary_mask"] = binary_mask
        result["is_light_bg"] = is_light
        result["resolved_polarity"] = "auto"
        result["features_used"] = {}  # CNN doesn't use explicit features
        result["all_extracted_features"] = {}
        result["model_metadata"] = {
            "model_type": "ResNet-18",
            "feature_set_name": "CNN (learned features)",
            "n_training_samples": "N/A",
        }

        return result

    finally:
        if isinstance(image_input, bytes) and 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)


def render_progress_steps(current_step: int):
    """Render progress step indicator using Streamlit columns for reliability."""
    steps = [
        ("1", "Upload"),
        ("2", "Preprocess"),
        ("3", "Analyze"),
        ("4", "Results"),
    ]
    cols = st.columns(4, gap="small")
    for i, (num, label) in enumerate(steps, 1):
        with cols[i - 1]:
            if i < current_step:
                cls = "complete"
                icon = "✓"
                bg = "var(--color-success-light)"
                border = "var(--color-success)"
                color = "var(--color-success)"
            elif i == current_step:
                cls = "active"
                icon = num
                bg = "var(--color-primary-light)"
                border = "var(--color-primary)"
                color = "var(--color-primary)"
            else:
                cls = ""
                icon = num
                bg = "var(--color-bg)"
                border = "var(--color-border)"
                color = "var(--color-text-muted)"

            st.markdown(
                f"""
                <div style="
                    background: {bg};
                    border: 1px solid {border};
                    border-radius: var(--radius-md);
                    padding: 12px 8px;
                    text-align: center;
                    min-height: 72px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                ">
                    <div style="
                        width: 24px; height: 24px; border-radius: 50%;
                        background: {color}; color: white;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 0.7rem; font-weight: 600;
                    ">{icon}</div>
                    <span style="font-size: 0.7rem; font-weight: 500; color: {color};">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_status_badge(available: bool, message: str):
    """Render model availability status badge."""
    if available:
        cls = "status-ready"
        dot = ""
        text = "Model Ready"
    else:
        cls = "status-missing"
        dot = ""
        text = f"Model Unavailable: {message}"
    st.markdown(
        f'<span class="status-badge {cls}"><span class="status-dot"></span>{text}</span>',
        unsafe_allow_html=True,
    )


def plot_feature_importance(importance_dict: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    df = pd.DataFrame(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importance"]
    )
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker_color="#0284C7",
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Global Feature Importance (Random Forest)",
        xaxis_title="Mean Decrease in Impurity",
        yaxis_title="",
        height=350,
        margin=dict(l=160, r=40, t=60, b=40),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=12),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", autorange="reversed")
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ["LPD", "PD"]) -> go.Figure:
    """Create confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Blues",
        showscale=True,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Out-of-Fold Confusion Matrix (5-Fold CV)",
        height=350,
        margin=dict(l=100, r=40, t=60, b=80),
        font=dict(family="Inter, system-ui", size=12),
    )
    return fig


def plot_feature_distributions(
    features_df: pd.DataFrame,
    feature_names: List[str],
    current_features: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Create violin/box plots for feature distributions by class."""
    fig = go.Figure()

    colors = {"PD": "#F59E0B", "LPD": "#34D399"}

    for i, feat in enumerate(feature_names):
        if feat not in features_df.columns:
            continue
        for cls in ["PD", "LPD"]:
            cls_data = features_df[features_df["class_label"] == cls][feat].dropna()
            if len(cls_data) == 0:
                continue
            fig.add_trace(go.Violin(
                y=cls_data,
                name=f"{cls}",
                legendgroup=cls,
                scalegroup=feat,
                side="positive" if cls == "PD" else "negative",
                line_color=colors[cls],
                fillcolor=colors[cls],
                opacity=0.5,
                meanline_visible=True,
                box_visible=True,
                width=0.8,
                showlegend=(i == 0),
                hovertemplate=f"<b>{feat}</b><br>Class: {cls}<br>Value: %{{y:.4f}}<extra></extra>",
            ))

        # Add current sample marker if provided
        if current_features and feat in current_features:
            val = current_features[feat]
            fig.add_trace(go.Scatter(
                x=[feat], y=[val],
                mode="markers",
                marker=dict(size=12, color="#0F172A", symbol="diamond", line=dict(width=2, color="white")),
                name="Current Sample",
                showlegend=(i == 0),
                hovertemplate=f"<b>{feat}</b><br>Current: {val:.4f}<extra></extra>",
            ))

    fig.update_layout(
        title="Feature Distributions by Class (Violin + Box)",
        xaxis_title="Feature",
        yaxis_title="Value",
        violinmode="group",
        height=500,
        margin=dict(l=60, r=40, t=60, b=100),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickangle=45, gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


def plot_probability_gauge(prob_pd: float, prob_lpd: float) -> go.Figure:
    """Create dual gauge chart for class probabilities."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_pd * 100,
        domain={"x": [0, 0.45], "y": [0, 1]},
        title={"text": "Potential Dysgraphia (PD)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#D97706"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#FEF3C7"},
                {"range": [33, 66], "color": "#FDE68A"},
                {"range": [66, 100], "color": "#FCD34D"},
            ],
            "threshold": {
                "line": {"color": "#92400E", "width": 3},
                "thickness": 0.8,
                "value": prob_pd * 100,
            },
        },
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_lpd * 100,
        domain={"x": [0.55, 1], "y": [0, 1]},
        title={"text": "Low Potential Dysgraphia (LPD)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#16A34A"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#F0FDF4"},
                {"range": [33, 66], "color": "#DCFCE7"},
                {"range": [66, 100], "color": "#86EFAC"},
            ],
            "threshold": {
                "line": {"color": "#166534", "width": 3},
                "thickness": 0.8,
                "value": prob_lpd * 100,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter, system-ui", size=11),
    )
    return fig


def plot_roc_curve(eval_metrics: pd.DataFrame) -> go.Figure:
    """Create ROC curve from evaluation metrics (approximate from mean AUC)."""
    # Since we don't have raw OOF probabilities in the CSV, we'll show the summary bar chart
    # This is a placeholder - real ROC would need the saved OOF probabilities
    fig = go.Figure()
    fig.add_annotation(
        text="ROC curves require out-of-fold probabilities.<br>Run full evaluation to generate.",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#64748B"),
    )
    fig.update_layout(
        title="ROC Curve (Out-of-Fold)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        plot_bgcolor="white",
    )
    return fig


def plot_metrics_comparison(eval_metrics: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing model configurations."""
    if eval_metrics is None or len(eval_metrics) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Evaluation metrics not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    metrics = ["accuracy_mean", "balanced_accuracy_mean", "precision_mean", "recall_pd_mean", "recall_lpd_mean", "f1_score_mean", "roc_auc_mean"]
    metric_labels = ["Accuracy", "Bal. Acc.", "Precision", "Recall (PD)", "Recall (LPD)", "F1-Score", "ROC-AUC"]

    fig = go.Figure()
    colors = ["#0284C7", "#F59E0B", "#16A34A", "#8B5CF6"]

    for idx, row in eval_metrics.iterrows():
        vals = [row[m] for m in metrics]
        errs = [row[m.replace("_mean", "_std")] for m in metrics]
        fig.add_trace(go.Bar(
            name=row["configuration"],
            x=metric_labels,
            y=vals,
            error_y=dict(type="data", array=errs, visible=True, color="rgba(0,0,0,0.3)"),
            marker_color=colors[idx % len(colors)],
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Model & Feature Set Comparison (5-Fold Stratified CV)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05], gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
        height=400,
        margin=dict(l=60, r=40, t=60, b=80),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_stroke_width_distribution(binary_mask: np.ndarray) -> go.Figure:
    """Create histogram of stroke width from distance transform."""
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_dist = cv2.dilate(dist_map, kernel)
    ridge_mask = (dist_map == dilated_dist) & (dist_map > 0.0)

    if np.any(ridge_mask):
        local_radii = dist_map[ridge_mask].astype(np.float64)
        stroke_widths = 2.0 * local_radii

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=stroke_widths,
            nbinsx=30,
            marker_color="#0284C7",
            opacity=0.7,
            name="Stroke Width",
            hovertemplate="Width: %{x:.2f} px<br>Count: %{y}<extra></extra>",
        ))

        mean_sw = np.mean(stroke_widths)
        median_sw = np.median(stroke_widths)
        fig.add_vline(x=mean_sw, line_dash="dash", line_color="#D97706", annotation_text=f"Mean: {mean_sw:.2f}")
        fig.add_vline(x=median_sw, line_dash="dot", line_color="#16A34A", annotation_text=f"Median: {median_sw:.2f}")

        fig.update_layout(
            title="Stroke Width Distribution (Medial Axis Ridges)",
            xaxis_title="Stroke Width (pixels)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=60, r=40, t=60, b=50),
            plot_bgcolor="white",
            font=dict(family="Inter, system-ui", size=11),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#E2E8F0")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="No stroke ridges detected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200)
        return fig


def plot_component_size_distribution(binary_mask: np.ndarray) -> go.Figure:
    """Create histogram of connected component areas."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        areas = areas[areas >= 5]  # Filter noise

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=areas,
            nbinsx=30,
            marker_color="#8B5CF6",
            opacity=0.7,
            name="Component Area",
            hovertemplate="Area: %{x:.0f} px²<br>Count: %{y}<extra></extra>",
        ))

        fig.update_layout(
            title="Connected Component Area Distribution",
            xaxis_title="Area (pixels²)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=60, r=40, t=60, b=50),
            plot_bgcolor="white",
            font=dict(family="Inter, system-ui", size=11),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#E2E8F0")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="No components found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200)
        return fig


# =============================================================================
# Sidebar: Model Status & Info
# =============================================================================
with st.sidebar:
    st.markdown("### GraphiSense")
    st.caption("Computer vision-based handwriting analysis for potential dysgraphia screening support.")

    # Model status
    model_available, model_msg = check_model_availability()
    render_model_status_badge(model_available, model_msg)

    st.markdown("---")

    # Metadata display
    meta = load_model_metadata()
    if meta:
        st.markdown("**Model Details**")
        st.markdown(f"- **Type**: {meta.get('model_type', 'Unknown').replace('_', ' ').title()}")
        st.markdown(f"- **Feature Set**: {meta.get('feature_set_name', 'Unknown').replace('_', ' ').title()}")
        st.markdown(f"- **Features**: {meta.get('num_features', '?')}")
        st.markdown(f"- **Training Samples**: {meta.get('n_training_samples', '?')}")
        dist = meta.get("training_distribution", {})
        st.markdown(f"- **Class Balance**: PD={dist.get('PD', '?')}, LPD={dist.get('LPD', '?')}")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Mendeley Potential Dysgraphia Handwriting Dataset (Ramlan et al., 2023)")
    st.caption("249 images • 114 PD • 135 LPD")

    st.markdown("---")
    st.markdown(
        '<div class="footer-note">GraphiSense is an academic research prototype. '
        'It does not provide medical or clinical diagnoses.</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Header
# =============================================================================
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


# =============================================================================
# Mode Selection
# =============================================================================
st.markdown("### Select Analysis Mode")

# Check CNN availability
cnn_available, cnn_msg = check_cnn_availability()

col_mode1, col_mode2 = st.columns(2, gap="large")

with col_mode1:
    basic_selected = st.button(
        "✅  Basic Analysis — Random Forest",
        use_container_width=True,
        type="primary" if st.session_state.get("analysis_mode", "basic") == "basic" else "secondary",
        help="Analyze 9 static handwriting features using a trained Random Forest classifier (Stratified 5-Fold CV, N=249)",
    )

with col_mode2:
    advanced_disabled = not cnn_available
    advanced_selected = st.button(
        "🔬  Advanced Analysis — CNN / ResNet",
        use_container_width=True,
        disabled=advanced_disabled,
        type="primary" if st.session_state.get("analysis_mode") == "advanced" else "secondary",
        help="Deep learning analysis using ResNet-18 transfer learning" if cnn_available else "Train a CNN model first (see README for instructions)",
    )

if basic_selected:
    st.session_state["analysis_mode"] = "basic"
if advanced_selected:
    st.session_state["analysis_mode"] = "advanced"

mode = st.session_state.get("analysis_mode", "basic")

# Mode badges
if mode == "basic":
    st.markdown('<span class="mode-badge mode-badge-basic">Active: Basic Analysis (Random Forest)</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="mode-badge mode-badge-advanced">Active: Advanced Analysis (CNN/ResNet)</span>', unsafe_allow_html=True)

if mode == "advanced":
    if not cnn_available:
        st.error(f"⚠️ **CNN model not available.** {cnn_msg}")
        st.info("""
        To enable Advanced Analysis:
        1. Ensure the dataset is placed under `data/raw/` with `Potential Dysgraphia/` and `Low Potential Dysgraphia/` folders
        2. Run training: `python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn`
        3. The best checkpoint will be saved to `outputs/models/cnn/best_model.pt`
        """)
        st.stop()
    else:
        st.success(f"✅ CNN model loaded: {cnn_msg}")

st.divider()


# =============================================================================
# Progress Indicator
# =============================================================================
step = 1
if st.session_state.get("image_bytes") is not None:
    step = 2
if st.session_state.get("preprocessed") is not None:
    step = 3
if st.session_state.get("inference_result") is not None:
    step = 4
render_progress_steps(step)


# =============================================================================
# Main Layout: Input + Visualizations
# =============================================================================
col_input, col_viz = st.columns([1, 1.3], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: Input & Controls
# -----------------------------------------------------------------------------
with col_input:
    st.markdown("#### Input Handwriting Sample")

    # Source selection
    input_source = st.radio(
        "Input source",
        options=["Sample Image", "Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_bytes = None
    image_name = "sample.png"
    sample_path = None

    if input_source == "Sample Image":
        # Build sample list dynamically
        sample_options = {}
        if DATASET_ROOT.exists():
            for cls_folder, label in [("Potential Dysgraphia", "PD"), ("Low Potential Dysgraphia", "LPD")]:
                folder = DATASET_ROOT / cls_folder
                if folder.exists():
                    for img_file in sorted(folder.glob("*.jpg"))[:5]:  # Limit to 5 per class
                        sample_options[f"{label}: {img_file.name}"] = img_file
        else:
            st.warning("Dataset not found. Place dataset under `data/raw/` or use Upload.")

        if sample_options:
            selected = st.selectbox(
                "Select verified dataset sample",
                options=list(sample_options.keys()),
                label_visibility="collapsed",
            )
            sample_path = sample_options[selected]
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                image_name = sample_path.name
                st.session_state["image_bytes"] = image_bytes
                st.session_state["image_name"] = image_name
            else:
                st.error(f"Sample not found: {sample_path}")

    else:
        uploaded = st.file_uploader(
            "Upload handwriting crop (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            # Validate file size (max 10MB)
            if uploaded.size > 10 * 1024 * 1024:
                st.error("File too large. Maximum size is 10 MB.")
                uploaded = None
            else:
                image_bytes = uploaded.getvalue()
                image_name = uploaded.name
                st.session_state["image_bytes"] = image_bytes
                st.session_state["image_name"] = image_name

    # Polarity control
    st.markdown("---")
    polarity_choice = st.selectbox(
        "Background polarity",
        options=["auto", "dark_background", "light_background"],
        format_func=lambda x: {
            "auto": "Auto-detect (recommended)",
            "dark_background": "Dark background (dataset standard)",
            "light_background": "Light background (paper photo)",
        }[x],
        help="Dataset images have dark background with light ink. Paper photos have light background with dark ink.",
    )

    # Preprocessing preview
    if image_bytes is not None:
        # Check if we need to re-preprocess
        need_preprocess = (
            st.session_state.get("analyzed_image_name") != image_name
            or st.session_state.get("analyzed_polarity") != polarity_choice
            or "preprocessed" not in st.session_state
        )

        if need_preprocess:
            st.session_state.pop("inference_result", None)

        try:
            with st.spinner("Preprocessing..."):
                rgb_preview, gray_preview, mask_preview, is_light_detected = load_and_preprocess_image(
                    image_bytes, polarity=polarity_choice
                )
            st.session_state["preprocessed"] = {
                "rgb": rgb_preview,
                "gray": gray_preview,
                "mask": mask_preview,
                "is_light": is_light_detected,
            }
            h, w = rgb_preview.shape[:2]
            mode_desc = "Light background" if is_light_detected else "Dark background"
            st.caption(f"**Loaded**: `{image_name}`  |  `{w}×{h}` px  |  Detected: `{mode_desc}`")
        except Exception as e:
            st.error(f"Preprocessing failed: {type(e).__name__} — {e}")
            st.session_state.pop("preprocessed", None)

    # Analyze button
    if image_bytes is not None and "preprocessed" in st.session_state:
        if mode == "basic":
            # Basic Analysis - Random Forest
            button_label = "Analyze handwriting (Random Forest)"
            can_analyze = model_available
            error_msg = "Cannot analyze: Random Forest model not available. Complete training steps in README.md first."
        else:
            # Advanced Analysis - CNN
            button_label = "Analyze handwriting (CNN/ResNet)"
            can_analyze = cnn_available
            error_msg = "Cannot analyze: CNN model not available. Train a CNN model first."

        if st.button(button_label, type="primary", width="stretch"):
            if not can_analyze:
                st.error(error_msg)
            else:
                st.session_state.pop("inference_result", None)
                with st.spinner(f"Running {mode} inference..."):
                    try:
                        if mode == "basic":
                            # Random Forest inference (existing)
                            result = run_app_inference(
                                image_input=image_bytes,
                                filename=image_name,
                                polarity=polarity_choice,
                            )
                        else:
                            # CNN inference
                            result = run_cnn_inference(
                                image_input=image_bytes,
                                filename=image_name,
                            )
                        st.session_state["inference_result"] = result
                        st.session_state["analyzed_image_name"] = image_name
                        st.session_state["analyzed_polarity"] = polarity_choice
                        st.rerun()
                    except FileNotFoundError as e:
                        st.error(f"Required file not found: {e}")
                    except ValueError as e:
                        st.error(f"Input validation error: {e}")
                    except Exception as e:
                        st.error(f"Inference failed: {type(e).__name__} — {e}")

    elif image_bytes is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">✍️</div>
                <div class="empty-state-title">No image selected</div>
                <div class="empty-state-text">
                    Choose a sample image or upload your own handwriting crop to begin analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# RIGHT COLUMN: Primary Visualizations
# -----------------------------------------------------------------------------
with col_viz:
    st.markdown("#### Primary Visualizations")

    result = st.session_state.get("inference_result")
    preprocessed = st.session_state.get("preprocessed")

    if result is None:
        if preprocessed is not None:
            # Show preprocessing preview while waiting for analysis
            rgb_img = preprocessed["rgb"]
            binary_mask = preprocessed["mask"]
            is_light = preprocessed["is_light"]

            c_orig, c_proc = st.columns(2)
            with c_orig:
                st.markdown('<div class="image-caption">Original Image</div>', unsafe_allow_html=True)
                st.image(rgb_img, width="stretch")
            with c_proc:
                fg_pct = (binary_mask == 255).mean() * 100
                mode_tag = "Light background" if is_light else "Dark background"
                st.markdown(f'<div class="image-caption">Binary Mask (Otsu, foreground {fg_pct:.1f}%, {mode_tag})</div>', unsafe_allow_html=True)
                st.image(binary_mask, clamp=True, width="stretch")

            st.info("Click **Analyze handwriting** to run feature extraction and model inference.")
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-title">No analysis results</div>
                    <div class="empty-state-text">
                        Select an image and click <strong>Analyze handwriting</strong> to see results.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Full results display
        label = result["predicted_label"]
        probs = result["class_probabilities"]
        rgb_img = result["original_rgb"]
        binary_mask = result["binary_mask"]
        is_light = result["is_light_bg"]
        features_used = result["features_used"]
        all_features = result["all_extracted_features"]
        meta = result.get("model_metadata", {})

        # Prediction card
        if label == "PD":
            st.markdown(
                f"""
                <div class="pred-card pred-card-pd">
                    <strong>Model Prediction:</strong> Potential Dysgraphia (PD)<br>
                    <strong>Estimated Probability:</strong> PD: <strong>{probs['PD']:.1%}</strong> &nbsp;|&nbsp; LPD: <strong>{probs['LPD']:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="pred-card pred-card-lpd">
                    <strong>Model Prediction:</strong> Low Potential Dysgraphia (LPD)<br>
                    <strong>Estimated Probability:</strong> LPD: <strong>{probs['LPD']:.1%}</strong> &nbsp;|&nbsp; PD: <strong>{probs['PD']:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive probability gauges
        st.plotly_chart(
            plot_probability_gauge(probs["PD"], probs["LPD"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Side-by-side images
        c_orig, c_proc = st.columns(2)
        with c_orig:
            st.markdown('<div class="image-caption">Original Image</div>', unsafe_allow_html=True)
            st.image(rgb_img, width="stretch")
        with c_proc:
            fg_pct = (binary_mask == 255).mean() * 100
            mode_tag = "Light background" if is_light else "Dark background"
            st.markdown(f'<div class="image-caption">Binary Mask (Otsu, foreground {fg_pct:.1f}%, {mode_tag})</div>', unsafe_allow_html=True)
            st.image(binary_mask, clamp=True, width="stretch")

        # Distance transform
        dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_INFERNO)
        dist_rgb = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB)

        st.markdown('<div class="image-caption">Medial Axis Distance Transform (brighter = thicker stroke center)</div>', unsafe_allow_html=True)
        st.image(dist_rgb, width="stretch")


# =============================================================================
# Detail Tabs
# =============================================================================
if result is not None:
    st.divider()

    # Determine tabs based on mode
    is_cnn = mode == "advanced"

    if is_cnn:
        tabs = st.tabs([
            "🖼️ Image Processing",
            "🤖 Model Details",
            "📈 Training History",
            "📊 Evaluation",
            "⚠️ Limitations",
        ])
    else:
        tabs = st.tabs([
            "📊 Feature Analysis",
            "🖼️ Image Processing",
            "🔬 Preprocessing Explorer",
            "🤖 Model Details",
            "⚠️ Limitations",
        ])

    # -------------------------------------------------------------------------
    # Tab 1: Feature Analysis (Basic) / Model Architecture (Advanced)
    # -------------------------------------------------------------------------
    with tabs[0]:
        if is_cnn:
            # CNN Model Architecture tab
            st.markdown("##### CNN Model Architecture")
            
            cnn_meta = load_cnn_metadata()
            if cnn_meta:
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("Architecture", cnn_meta.get("model_type", "ResNet-18"))
                with col_a2:
                    st.metric("Input Size", f"{cnn_meta.get('image_size', 224)}×{cnn_meta.get('image_size', 224)}")
                with col_a3:
                    st.metric("Dropout", f"{cnn_meta.get('dropout', 0.3):.1f}")
                
                st.markdown("---")
                st.markdown("**Architecture Details**")
                st.markdown(f"""
                - **Base Model**: ResNet-18 (pretrained on ImageNet: {cnn_meta.get('pretrained', True)})
                - **Input Resolution**: {cnn_meta.get('image_size', 224)}×{cnn_meta.get('image_size', 224)} RGB
                - **Backbone Frozen**: {cnn_meta.get('freeze_backbone', False)}
                - **Dropout Rate**: {cnn_meta.get('dropout', 0.3)}
                - **Final Layer**: Linear(512 → 2) for binary classification (PD/LPD)
                - **Training Epochs**: {cnn_meta.get('training_epochs', 0)}
                - **Best Validation Loss**: {cnn_meta.get('val_loss', 'N/A'):.4f}
                """)
                
                # Validation metrics if available
                val_metrics = cnn_meta.get("val_metrics", {})
                if val_metrics:
                    st.markdown("##### Validation Metrics (Best Epoch)")
                    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                    with col_v1:
                        st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                    with col_v2:
                        st.metric("Balanced Acc", f"{val_metrics.get('balanced_accuracy', 0):.3f}")
                    with col_v3:
                        st.metric("F1 (PD)", f"{val_metrics.get('f1_pd', 0):.3f}")
                    with col_v4:
                        st.metric("ROC-AUC", f"{val_metrics.get('roc_auc', 0):.3f}")
            else:
                st.info("CNN metadata not available.")
            
            st.markdown("---")
            st.caption("CNN models learn features directly from pixels. Unlike the Random Forest baseline, "
                       "there are no handcrafted features to display. The model learns hierarchical "
                       "visual representations through convolutional layers.")
        else:
            # Basic Analysis - Random Forest Feature Analysis
            st.markdown("##### Features Used by Model")

            # Feature descriptions
            feature_descriptions = {
                "ink_density": ("Proportion of stroke pixels to total image area", "Ratio [0-1]"),
                "centroid_x_norm": ("Normalized horizontal center of stroke mass", "Ratio [0-1]"),
                "centroid_y_norm": ("Normalized vertical center of stroke mass", "Ratio [0-1]"),
                "active_row_ratio": ("Proportion of image rows containing handwriting strokes", "Ratio [0-1]"),
                "component_aspect_ratio_mean": ("Mean width-to-height ratio of connected components", "Ratio"),
                "component_aspect_ratio_std": ("Std. dev. of component aspect ratios", "Ratio"),
                "stroke_width_mean": ("Mean stroke thickness via medial-axis distance transform", "Pixels (px)"),
                "stroke_width_median": ("Median stroke thickness via medial-axis distance transform", "Pixels (px)"),
                "stroke_width_std": ("Std. dev. of stroke thickness across strokes", "Pixels (px)"),
            }

            # Features used table
            feature_rows = []
            for name, val in features_used.items():
                desc, unit = feature_descriptions.get(name, ("Static spatial feature", "-"))
                formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
                feature_rows.append({
                    "Feature": name,
                    "Measured Value": formatted_val,
                    "Unit": unit,
                    "Description": desc,
                })

            st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

            # Feature importance chart
            if meta.get("global_feature_importance_or_coef"):
                st.markdown("##### Feature Importance (Global)")
                st.plotly_chart(
                    plot_feature_importance(meta["global_feature_importance_or_coef"]),
                    use_container_width=True,
                )
                st.caption("Mean decrease in impurity from the fitted Random Forest. Not sample-specific attribution (e.g., SHAP).")

            # Feature distributions
            st.markdown("##### Feature Distributions by Class")
            features_df = load_features_dataset()
            if features_df is not None:
                feat_names = list(features_used.keys())
                st.plotly_chart(
                    plot_feature_distributions(features_df, feat_names, features_used),
                    use_container_width=True,
                )
                st.caption("Violin plots show full distribution; box plots show median/IQR. Diamond = current sample.")
            else:
                st.info("Feature dataset not found. Run `python -m src.features.build_feature_dataset` to enable distribution charts.")

            # All features expander
            with st.expander("View all 25 extracted static features"):
                all_rows = [{"Feature": k, "Measured Value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in all_features.items()]
                st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # Tab 2: Image Processing (Basic) / Tab 1: Image Processing (Advanced)
    # -------------------------------------------------------------------------
    img_proc_tab_idx = 1 if not is_cnn else 0
    with tabs[img_proc_tab_idx]:
        st.markdown("##### Preprocessing Pipeline")

        col_proc1, col_proc2 = st.columns(2)

        with col_proc1:
            st.markdown("**Original (RGB)**")
            st.image(result["original_rgb"], width="stretch")
            st.caption(f"{result['original_rgb'].shape[1]}×{result['original_rgb'].shape[0]} px")

        with col_proc2:
            st.markdown("**Grayscale**")
            st.image(result["grayscale"], width="stretch", clamp=True)
            st.caption(f"{result['grayscale'].shape[1]}×{result['grayscale'].shape[0]} px")

        st.markdown("---")
        st.markdown("**Binary Mask (Otsu Thresholding)**")
        st.image(result["binary_mask"], width="stretch", clamp=True)
        fg_pct = (result["binary_mask"] == 255).mean() * 100
        st.caption(f"Foreground: {fg_pct:.1f}%  |  Convention: Handwriting=255 (white), Background=0 (black)")

        if not is_cnn:
            st.markdown("---")
            col_dist1, col_dist2 = st.columns(2)

            with col_dist1:
                st.markdown("**Stroke Width Distribution**")
                st.plotly_chart(
                    plot_stroke_width_distribution(result["binary_mask"]),
                    use_container_width=True,
                )

            with col_dist2:
                st.markdown("**Component Area Distribution**")
                st.plotly_chart(
                    plot_component_size_distribution(result["binary_mask"]),
                    use_container_width=True,
                )

        with st.expander("Technical Details"):
            st.markdown(f"""
            - **Polarity**: {result.get('resolved_polarity', 'unknown')}
            - **Binarization**: Otsu automatic thresholding
            - **Connected Components**: 8-connectivity, min area = 5 px²
            - **Stroke Width**: 2 × inscribed radius (distance transform ridges)
            - **Line Segmentation**: Not performed (dataset contains single-line crops)
            """)

    # -------------------------------------------------------------------------
    # Tab 3: Preprocessing Explorer (Basic only)
    # -------------------------------------------------------------------------
    if not is_cnn:
        preproc_explorer_tab_idx = 2
        with tabs[preproc_explorer_tab_idx]:
            st.markdown("##### Interactive Preprocessing Explorer")
            st.caption("Step through the OpenCV preprocessing pipeline. Each stage shows the actual intermediate image produced by the pipeline.")

            # Get or compute preprocessing stages
            preprocessed = st.session_state.get("preprocessed")
            if preprocessed is not None:
                # Initialize session state for viewer
                if "preproc_viewer_stage" not in st.session_state:
                    st.session_state["preproc_viewer_stage"] = None
                if "preproc_viewer_zoom" not in st.session_state:
                    st.session_state["preproc_viewer_zoom"] = 1.0

                # Get the original images
                rgb_img = preprocessed["rgb"]
                gray_img = preprocessed["gray"]
                binary_mask = preprocessed["mask"]
                is_light = preprocessed["is_light"]
                polarity = result.get("resolved_polarity", "dark_background")
                orig_h, orig_w = rgb_img.shape[:2]

                # Compute all stages if not cached
                if "preproc_stages" not in st.session_state:
                    with st.spinner("Computing preprocessing stages..."):
                        stages_dict = compute_preprocessing_stages(gray_img, binary_mask, polarity)
                        st.session_state["preproc_stages"] = stages_dict
                else:
                    stages_dict = st.session_state["preproc_stages"]

                # Combine with base images
                all_stages = {
                    "original_rgb": rgb_img,
                    "grayscale": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB),
                    "binary_mask": cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB),
                    **stages_dict,
                }

                stage_defs = get_preprocessing_stages()

                # Thumbnail gallery - compact grid of cards
                st.markdown("**Pipeline Stages**")
                st.caption("Click **View** on any stage to open the full-size image viewer.")

                # Create grid: 3 columns for 6 stages = 2 rows
                cols_per_row = 3
                num_stages = len(stage_defs)

                for row_idx in range(0, num_stages, cols_per_row):
                    row_stages = stage_defs[row_idx:row_idx + cols_per_row]
                    cols = st.columns(len(row_stages), gap="medium")

                    for col_idx, stage_def in enumerate(row_stages):
                        with cols[col_idx]:
                            stage_key = stage_def["key"]
                            if stage_key in all_stages:
                                stage_image = all_stages[stage_key]
                                stage_h, stage_w = stage_image.shape[:2]

                                # Card container
                                with st.container(border=True):
                                    # Stage number and name
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; margin-bottom: 8px;">
                                            <span style="
                                                background: var(--color-primary-light);
                                                color: var(--color-primary);
                                                padding: 2px 8px;
                                                border-radius: 999px;
                                                font-size: 0.7rem;
                                                font-weight: 600;
                                            ">Stage {stage_def['id']}</span>
                                            <div style="font-weight: 500; font-size: 0.85rem; margin-top: 4px; color: var(--color-text);">
                                                {stage_def['name']}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # Thumbnail image - fixed height container with contained image
                                    thumb_h = 180
                                    aspect = stage_w / stage_h if stage_h > 0 else 1.0
                                    thumb_w = int(thumb_h * aspect)
                                    thumb_w = min(thumb_w, 300)  # Cap width

                                    # Display thumbnail centered
                                    left_pad, center, right_pad = st.columns([1, 10, 1])
                                    with center:
                                        st.image(stage_image, width=thumb_w, clamp=True)

                                    # Image dimensions
                                    st.markdown(
                                        f'<div style="text-align: center; font-size: 0.7rem; color: var(--color-text-muted); margin-top: 4px;">{stage_w}×{stage_h} px</div>',
                                        unsafe_allow_html=True,
                                    )

                                    # View button
                                    if st.button(
                                        "View",
                                        key=f"view_stage_{stage_def['id']}",
                                        use_container_width=True,
                                        type="secondary",
                                    ):
                                        st.session_state["preproc_viewer_stage"] = stage_def["id"]
                                        st.session_state["preproc_viewer_zoom"] = 1.0
                                        st.rerun()
                            else:
                                # Stage not available
                                with st.container(border=True):
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 20px;">
                                            <span style="
                                                background: var(--color-warning-light);
                                                color: var(--color-warning);
                                                padding: 2px 8px;
                                                border-radius: 999px;
                                                font-size: 0.7rem;
                                                font-weight: 600;
                                            ">Stage {stage_def['id']}</span>
                                            <div style="font-weight: 500; font-size: 0.85rem; margin-top: 4px;">{stage_def['name']}</div>
                                            <div style="font-size: 0.7rem; color: var(--color-text-muted); margin-top: 8px;">Not available</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                # Image viewer dialog
                if st.session_state.get("preproc_viewer_stage") is not None:
                    viewer_stage_id = st.session_state["preproc_viewer_stage"]
                    viewer_stage_def = next((s for s in stage_defs if s["id"] == viewer_stage_id), None)

                    if viewer_stage_def and viewer_stage_def["key"] in all_stages:
                        viewer_image = all_stages[viewer_stage_def["key"]]
                        viewer_h, viewer_w = viewer_image.shape[:2]

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
                                    # Store rotation state
                                    rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                                    if rot_key not in st.session_state:
                                        st.session_state[rot_key] = 0
                                    st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
                                    st.rerun()

                            # Display the image
                            zoom = st.session_state["preproc_viewer_zoom"]
                            rot_key = f"preproc_viewer_rot_{viewer_stage_id}"
                            rotation = st.session_state.get(rot_key, 0)

                            # Apply rotation if needed
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

                            # Apply zoom
                            if zoom != 1.0:
                                new_w = int(disp_img.shape[1] * zoom)
                                new_h = int(disp_img.shape[0] * zoom)
                                disp_img = cv2.resize(disp_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                            # Image metadata
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

                            # Display image - use column to center and allow scrolling if needed
                            st.image(disp_img, clamp=True, width="stretch")

                            # Close button
                            if st.button("Close Viewer", type="primary", use_container_width=True):
                                st.session_state["preproc_viewer_stage"] = None
                                st.rerun()

                        show_viewer()
                    else:
                        st.warning("Selected stage not available in current pipeline.")

                else:
                    st.info("Run an analysis first to populate the preprocessing explorer with your image.")

    # -------------------------------------------------------------------------
    # Tab 4: Model Details
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("##### Classifier Architecture & Metadata")

        # Metric cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{meta.get('model_type', 'Random Forest').replace('_', ' ').title()}</div>
                <div class="metric-label">Model Type</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{meta.get('feature_set_name', 'Unknown').replace('_', ' ').title()}</div>
                <div class="metric-label">Feature Set</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{meta.get('n_training_samples', '?')}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{meta.get('num_features', '?')}</div>
                <div class="metric-label">Features Used</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if not is_cnn:
            # Confusion matrix (RF)
            st.markdown("##### Out-of-Fold Confusion Matrix")
            cm_path = ROOT / "outputs" / "figures" / "confusion_matrices.png"
            if cm_path.exists():
                st.image(str(cm_path), width="stretch")
                st.caption("Cumulative out-of-fold confusion matrices for all 4 model/feature-set configurations.")
            else:
                st.info("Confusion matrix figure not found. Run full evaluation to generate.")

            st.markdown("---")

            # Model comparison chart
            st.markdown("##### Model Comparison (5-Fold CV)")
            eval_metrics = load_evaluation_metrics()
            if eval_metrics is not None:
                st.plotly_chart(
                    plot_metrics_comparison(eval_metrics),
                    use_container_width=True,
                )
                st.caption("Mean ± std across 5 stratified folds. Feature Set A = 9 normalized features; Set B = 25 full features.")
            else:
                st.info("Evaluation metrics not found. Run `python -m src.models.train_evaluate` to generate.")

            st.markdown("---")

        # Full metadata
        with st.expander("Full Model Metadata (JSON)"):
            st.json(meta)

    # -------------------------------------------------------------------------
    # CNN-specific tabs
    # -------------------------------------------------------------------------
    if is_cnn:
        # Training History tab
        training_history_tab_idx = 2
        with tabs[training_history_tab_idx]:
            st.markdown("##### Training History")
            cnn_meta = load_cnn_metadata()
            if cnn_meta:
                training_epochs = cnn_meta.get("training_epochs", 0)
                st.metric("Epochs Completed", training_epochs)

                # Try to load training history
                try:
                    import torch
                    checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
                    history = checkpoint.get("training_history", {})

                    if history.get("train_loss"):
                        import plotly.graph_objects as go
                        epochs = list(range(1, len(history["train_loss"]) + 1))

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="#0284C7")))
                        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="#D97706")))
                        fig.update_layout(
                            title="Training & Validation Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=350,
                            margin=dict(l=60, r=40, t=60, b=40),
                            plot_bgcolor="white",
                        )
                        fig.update_xaxes(gridcolor="#E2E8F0")
                        fig.update_yaxes(gridcolor="#E2E8F0")
                        st.plotly_chart(fig, use_container_width=True)

                    if history.get("train_acc"):
                        fig = go.Figure()
                        epochs = list(range(1, len(history["train_acc"]) + 1))
                        fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc", line=dict(color="#16A34A")))
                        fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc", line=dict(color="#F59E0B")))
                        fig.update_layout(
                            title="Training & Validation Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            height=350,
                            margin=dict(l=60, r=40, t=60, b=40),
                            plot_bgcolor="white",
                        )
                        fig.update_xaxes(gridcolor="#E2E8F0")
                        fig.update_yaxes(gridcolor="#E2E8F0")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Training history not available in checkpoint.")
            else:
                st.info("CNN metadata not available.")

        # Evaluation tab
        eval_tab_idx = 3
        with tabs[eval_tab_idx]:
            st.markdown("##### Held-Out Evaluation Results")
            cnn_meta = load_cnn_metadata()
            if cnn_meta:
                val_metrics = cnn_meta.get("val_metrics", {})
                if val_metrics:
                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                    with col_e1:
                        st.metric("Val Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                    with col_e2:
                        st.metric("Val Balanced Acc", f"{val_metrics.get('balanced_accuracy', 0):.3f}")
                    with col_e3:
                        st.metric("Val F1 (PD)", f"{val_metrics.get('f1_pd', 0):.3f}")
                    with col_e4:
                        st.metric("Val ROC-AUC", f"{val_metrics.get('roc_auc', 0):.3f}")

                    st.markdown("---")

                    # Per-class metrics
                    st.markdown("##### Per-Class Metrics (Validation)")
                    per_class_data = [
                        ("LPD", "Precision", val_metrics.get("precision_lpd", 0)),
                        ("LPD", "Recall", val_metrics.get("recall_lpd", 0)),
                        ("LPD", "F1", val_metrics.get("f1_lpd", 0)),
                        ("PD", "Precision", val_metrics.get("precision_pd", 0)),
                        ("PD", "Recall", val_metrics.get("recall_pd", 0)),
                        ("PD", "F1", val_metrics.get("f1_pd", 0)),
                    ]

                    for cls_name, metric_name, value in per_class_data:
                        st.metric(f"{cls_name} {metric_name}", f"{value:.3f}")

                    # Confusion matrix if available
                    if "confusion_matrix" in val_metrics:
                        st.markdown("---")
                        st.markdown("##### Validation Confusion Matrix")
                        import plotly.graph_objects as go
                        import numpy as np
                        cm = np.array(val_metrics["confusion_matrix"])
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=["Pred LPD", "Pred PD"],
                            y=["True LPD", "True PD"],
                            colorscale="Blues",
                            showscale=True,
                            text=[[f"{cm[i, j]}" for j in range(2)] for i in range(2)],
                            texttemplate="%{text}",
                            textfont={"size": 16},
                        ))
                        fig.update_layout(height=300, margin=dict(l=80, r=40, t=40, b=60))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Validation metrics not available in checkpoint.")
            else:
                st.info("CNN metadata not available.")

        # Limitations tab for CNN
        limits_tab_idx = 4
        with tabs[limits_tab_idx]:
            st.markdown("##### Academic Screening-Support Scope & Integrity Safeguards")
            st.markdown(
                """
                - **Non-Diagnostic Notice**: GraphiSense is an academic research prototype. Outputs are statistical model estimates on a benchmark dataset and do not constitute clinical, psychiatric, or educational diagnoses.

                - **Research Label Nature**: Dataset labels (`PD` and `LPD`) reflect study research categories from the dataset source, not independently verified clinical medical evaluations.

                - **Static Image Limitations**: Static 2D handwriting images cannot measure dynamic physical properties such as actual pen-tip pressure, writing velocity, acceleration, or cognitive in-air hesitation pauses. All features (stroke thickness, ink density) are purely 2D morphological proxies.

                - **Image-Level Validation Caveat**: The underlying dataset does not provide participant or writer identifiers. Cross-validation was partitioned at the **image level**. If multiple line crops originate from the same writer, samples from the same writer may occur across training and validation folds, which can yield optimistic performance estimates compared to true writer-independent evaluation. No writer IDs are invented or assumed.

                - **Resolution Dependency**: Stroke thickness measurements are expressed in pixels and depend on image scan/capture resolution. Features are not DPI-normalized.

                - **CNN-Specific Limitations**: The ResNet-18 model learns visual patterns directly from pixels without explicit feature engineering. The decision process is less interpretable than the Random Forest baseline. No Grad-CAM or attention visualization is currently implemented.

                - **Screening vs. Diagnosis**: This tool provides screening *support* only. A clinical diagnosis of dysgraphia requires multidisciplinary assessment including dynamic writing analysis, cognitive evaluation, and educational context.
                """
            )
    else:
        # Limitations tab for RF (last tab)
        limits_tab_idx = 3 if not is_cnn else 4
        with tabs[limits_tab_idx]:
            st.markdown("##### Academic Screening-Support Scope & Integrity Safeguards")
            st.markdown(
                """
                - **Non-Diagnostic Notice**: GraphiSense is an academic research prototype. Outputs are statistical model estimates on a benchmark dataset and do not constitute clinical, psychiatric, or educational diagnoses.

                - **Research Label Nature**: Dataset labels (`PD` and `LPD`) reflect study research categories from the dataset source, not independently verified clinical medical evaluations.

                - **Static Image Limitations**: Static 2D handwriting images cannot measure dynamic physical properties such as actual pen-tip pressure, writing velocity, acceleration, or cognitive in-air hesitation pauses. All features (stroke thickness, ink density) are purely 2D morphological proxies.

                - **Image-Level Validation Caveat**: The underlying dataset does not provide participant or writer identifiers. Cross-validation was partitioned at the **image level**. If multiple line crops originate from the same writer, samples from the same writer may occur across training and validation folds, which can yield optimistic performance estimates compared to true writer-independent evaluation. No writer IDs are invented or assumed.

                - **Resolution Dependency**: Stroke thickness measurements are expressed in pixels and depend on image scan/capture resolution. Features are not DPI-normalized.

                - **Screening vs. Diagnosis**: This tool provides screening *support* only. A clinical diagnosis of dysgraphia requires multidisciplinary assessment including dynamic writing analysis, cognitive evaluation, and educational context.
                """
            )


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    """
    <div class="footer-note">
        GraphiSense is an academic research prototype for potential dysgraphia screening support.
        It does not provide medical or clinical diagnoses. Dataset: Mendeley Data (DOI: 10.17632/39hr8dx76p.1).
    </div>
    """,
    unsafe_allow_html=True,
)
