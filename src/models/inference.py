"""Inference pipeline for single-image handwriting dysgraphia screening in GraphiSense.

Loads a previously trained model pipeline artifact and metadata to make deterministic
predictions without refitting or modifying the model.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.image_loader import load_image_with_grayscale
from src.features.handwriting_features import extract_handwriting_features

DEFAULT_MODEL_PATH = _REPO_ROOT / "outputs" / "models" / "model_pipeline.joblib"
DEFAULT_METADATA_PATH = _REPO_ROOT / "outputs" / "models" / "model_metadata.json"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_saved_model(
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a previously trained scikit-learn pipeline and its associated metadata.

    Parameters
    ----------
    model_path : Optional[Union[str, Path]], optional
        Path to the serialized .joblib pipeline file.
        Defaults to `outputs/models/model_pipeline.joblib`.
    metadata_path : Optional[Union[str, Path]], optional
        Path to the model metadata JSON file.
        Defaults to `outputs/models/model_metadata.json`.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (fitted_pipeline, metadata_dictionary)

    Raises
    ------
    FileNotFoundError
        If the model artifact or metadata file does not exist.
    ValueError
        If the metadata file has an invalid or corrupted schema.
    """
    m_path = Path(model_path).resolve() if model_path else DEFAULT_MODEL_PATH.resolve()
    meta_path = (
        Path(metadata_path).resolve()
        if metadata_path
        else DEFAULT_METADATA_PATH.resolve()
    )

    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {m_path}. "
            "Ensure train_and_save_final_model() has been executed."
        )

    pipeline = joblib.load(m_path)
    if not hasattr(pipeline, "predict"):
        raise ValueError(f"Loaded model artifact at {m_path} does not implement required 'predict' method.")

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata file not found at: {meta_path}. "
            "Ensure train_and_save_final_model() has been executed."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse metadata JSON at {meta_path}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata at {meta_path} must contain a JSON object.")

    required_keys = {"model_type", "feature_names", "class_mapping"}
    missing = required_keys - set(metadata.keys())
    if missing:
        raise ValueError(f"Metadata file is missing required fields: {missing}")

    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError(
            "Metadata 'feature_names' must be a non-empty list of unique, non-empty strings."
        )

    class_mapping = metadata.get("class_mapping")
    if (
        not isinstance(class_mapping, dict)
        or class_mapping.get("0") != "LPD"
        or class_mapping.get("1") != "PD"
    ):
        raise ValueError(
            "Metadata 'class_mapping' must map '0' to 'LPD' and '1' to 'PD'."
        )

    model_feature_count = getattr(pipeline, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(feature_names):
        raise ValueError(
            "Model feature count does not match metadata: "
            f"{model_feature_count} model inputs vs. {len(feature_names)} feature names."
        )

    return pipeline, metadata


def predict_handwriting(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    binarize_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end inference on a single handwriting image.

    This function:
    1. Loads the image from disk with validation.
    2. Converts to grayscale and generates a polarity-aware binary mask.
    3. Extracts all static handwriting features.
    4. Selects and orders the exact feature vector expected by the saved model.
    5. Evaluates the fitted pipeline (never retraining or fitting on input).
    6. Returns predicted label, model probability estimates, and feature values.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to the handwriting image file (.jpg, .png, etc.).
    model_path : Optional[Union[str, Path]], optional
        Path to model pipeline joblib file. Defaults to outputs/models/model_pipeline.joblib.
    metadata_path : Optional[Union[str, Path]], optional
        Path to model metadata JSON file. Defaults to outputs/models/model_metadata.json.
    binarize_kwargs : Optional[Dict[str, Any]], optional
        Optional thresholding parameters passed to binarize_handwriting.
        Defaults to `{"method": "otsu", "polarity": "dark_background"}`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'image_path': Path to the input image.
        - 'predicted_label': 'PD' or 'LPD'.
        - 'predicted_class_index': 1 (PD) or 0 (LPD).
        - 'class_probabilities': {'LPD': float, 'PD': float}.
        - 'features_used': Dict of numerical features passed to the model.
        - 'all_extracted_features': Dict of all 25 static features.
        - 'model_metadata': Metadata dictionary from training.
        - 'disclaimer': Explicit non-clinical research disclaimer.

    Raises
    ------
    FileNotFoundError
        If image file or model artifacts are missing.
    ValueError
        If image path is a directory or feature schema is incompatible.
    IOError
        If image decoding fails.
    """
    file_path = Path(image_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Handwriting image not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image file extension '{file_path.suffix}'. "
            f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {file_path}")

    # Validate the image before loading model artifacts so invalid uploads get a useful error.
    original_img, gray_img = load_image_with_grayscale(file_path)

    # 1. Load trained model pipeline and schema metadata
    pipeline, metadata = load_saved_model(
        model_path=model_path, metadata_path=metadata_path
    )

    # 3. Extract static handwriting features
    default_bin_kwargs = {"method": "otsu", "polarity": "dark_background"}
    active_bin_kwargs = binarize_kwargs or default_bin_kwargs

    all_features = extract_handwriting_features(
        gray_img,
        is_binary=False,
        min_component_area=5,
        binarize_kwargs=active_bin_kwargs,
    )

    # 4. Filter and order features strictly according to model metadata schema
    expected_features = metadata["feature_names"]
    missing_feats = [f for f in expected_features if f not in all_features]
    if missing_feats:
        raise ValueError(
            f"Extracted feature set is missing expected model columns: {missing_feats}"
        )

    features_ordered = {f: all_features[f] for f in expected_features}
    X_sample = pd.DataFrame([features_ordered])

    # 5. Predict using the frozen pipeline (no refitting!)
    pred_idx = int(pipeline.predict(X_sample)[0])

    if hasattr(pipeline, "predict_proba") and hasattr(pipeline, "classes_"):
        probs = pipeline.predict_proba(X_sample)[0]
        prob_dict = {}
        for idx, cls in enumerate(pipeline.classes_):
            label = metadata["class_mapping"].get(str(cls), str(cls))
            prob_dict[label] = round(float(probs[idx]), 4)
    elif hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_sample)[0]
        prob_dict = {
            "LPD": round(float(probs[0]), 4),
            "PD": round(float(probs[1]), 4),
        }
    else:
        prob_dict = {
            "LPD": 1.0 if pred_idx == 0 else 0.0,
            "PD": 1.0 if pred_idx == 1 else 0.0,
        }

    # Ensure standard screening labels are present in probability dictionary
    for expected_label in ["LPD", "PD"]:
        if expected_label not in prob_dict:
            prob_dict[expected_label] = 0.0

    predicted_label = metadata["class_mapping"].get(str(pred_idx), "UNKNOWN")

    return {
        "image_path": str(file_path),
        "predicted_label": predicted_label,
        "predicted_class_index": pred_idx,
        "class_probabilities": prob_dict,
        "features_used": features_ordered,
        "all_extracted_features": all_features,
        "model_metadata": {
            "model_type": metadata.get("model_type"),
            "feature_set_name": metadata.get("feature_set_name"),
            "n_training_samples": metadata.get("n_training_samples"),
            "resolution_dependency": metadata.get("resolution_dependency"),
            "interpretability_notice": metadata.get("interpretability_notice"),
        },
        "disclaimer": (
            "GraphiSense is an academic research and screening-support baseline. "
            "Predicted probabilities are statistical model estimates, not clinical "
            "risk scores or a medical diagnosis of dysgraphia."
        ),
    }
