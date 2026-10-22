"""Reusable application pipeline handler for GraphiSense Streamlit interface."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.preprocessing.image_loader import load_image, to_grayscale
from src.preprocessing.binarization import binarize_handwriting, detect_background_is_light
from src.models.inference import predict_handwriting


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to an OpenCV BGR array.

    Parameters
    ----------
    image_bytes : bytes
        Raw image binary data.

    Returns
    -------
    np.ndarray
        Decoded BGR uint8 image array.

    Raises
    ------
    ValueError
        If the image data cannot be decoded by OpenCV.
    """
    if not image_bytes:
        raise ValueError("Image bytes are empty")

    image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        raise ValueError("Failed to decode image bytes into a valid image array")

    return image_array


def load_and_preprocess_image(
    image_input: Union[bytes, str, Path],
    polarity: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load an image and generate its grayscale and binary mask representations.

    Reuses existing preprocessing modules without duplicating logic.

    Parameters
    ----------
    image_input : Union[bytes, str, Path]
        Raw image bytes or filesystem path.
    polarity : str, default='auto'
        'dark_background', 'light_background', or 'auto'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, bool]
        (original_rgb, grayscale, binary_mask, is_light_bg)
    """
    if isinstance(image_input, bytes):
        bgr = decode_image_bytes(image_input)
    else:
        path = Path(image_input).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")
        bgr = load_image(path, as_gray=False)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = to_grayscale(bgr)

    # Determine polarity
    if polarity == "auto":
        is_light = detect_background_is_light(gray)
        resolved_polarity = "light_background" if is_light else "dark_background"
    else:
        resolved_polarity = polarity
        is_light = resolved_polarity == "light_background"

    binary_mask = binarize_handwriting(gray, method="otsu", polarity=resolved_polarity)

    return rgb, gray, binary_mask, is_light


def run_app_inference(
    image_input: Union[bytes, str, Path],
    filename: str = "sample.png",
    polarity: str = "auto",
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end inference and package results for display.

    Parameters
    ----------
    image_input : Union[bytes, str, Path]
        Raw image bytes or filesystem path.
    filename : str
        Source filename for extension and identification.
    polarity : str, default='auto'
        Polarity setting ('auto', 'dark_background', or 'light_background').
    model_path : Optional[Union[str, Path]]
        Optional custom model artifact path.
    metadata_path : Optional[Union[str, Path]]
        Optional custom metadata path.

    Returns
    -------
    Dict[str, Any]
        Inference result dictionary with preview arrays attached.
    """
    if isinstance(image_input, bytes):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(
                f"Unsupported upload file extension '{suffix}'. "
                "Upload a JPG, JPEG, or PNG image."
            )

    rgb, gray, binary_mask, is_light = load_and_preprocess_image(image_input, polarity=polarity)

    # Temporary file handling for predict_handwriting interface if bytes provided
    temp_path = None
    try:
        if isinstance(image_input, bytes):
            suffix = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(image_input)
                temp_path = Path(tf.name)
            eval_path = temp_path
        else:
            eval_path = Path(image_input)

        resolved_pol = "light_background" if is_light else "dark_background"
        result = predict_handwriting(
            image_path=eval_path,
            model_path=model_path,
            metadata_path=metadata_path,
            binarize_kwargs={"method": "otsu", "polarity": resolved_pol},
        )

        result["original_rgb"] = rgb
        result["grayscale"] = gray
        result["binary_mask"] = binary_mask
        result["is_light_bg"] = is_light
        result["resolved_polarity"] = resolved_pol

        return result

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
