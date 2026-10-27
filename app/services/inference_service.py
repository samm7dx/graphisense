"""Inference service for model prediction orchestration."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CNN_CHECKPOINT_DIR = ROOT / "outputs" / "models" / "cnn" / "checkpoints"
CNN_CHECKPOINT_PATH = CNN_CHECKPOINT_DIR / "best_model.pt"


def run_rf_inference(
    image_input: Union[bytes, str, Path],
    filename: str,
    polarity: str,
) -> Dict[str, Any]:
    """Run Random Forest inference via pipeline handler."""
    from app.pipeline_handler import run_app_inference
    return run_app_inference(
        image_input=image_input,
        filename=filename,
        polarity=polarity,
    )


def run_cnn_inference(
    image_input: Union[bytes, str, Path],
    filename: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run CNN inference and package results for display."""
    if checkpoint_path is None:
        checkpoint_path = CNN_CHECKPOINT_PATH

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
        from src.models.cnn_inference import CNNInference
        from app.pipeline_handler import load_and_preprocess_image

        engine = CNNInference(checkpoint_path)
        result = engine.predict(eval_path)

        # Get preprocessed images for visualization
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


@st.cache_data(show_spinner=False)
def check_cnn_availability() -> tuple[bool, str]:
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