"""Binarization module for handwriting image processing in GraphiSense.

Produces binary masks following the strict convention:
- Handwriting foreground = 255 (white)
- Background = 0 (black)
"""

from typing import Optional
import cv2
import numpy as np


def detect_background_is_light(image: np.ndarray, border_fraction: float = 0.05) -> bool:
    """Detect whether an image has a light background (paper) or dark background.

    Samples border pixels along the top, bottom, left, and right margins.
    If the mean intensity of the border is greater than 127.5, the background
    is classified as light.

    Parameters
    ----------
    image : np.ndarray
        2D single-channel uint8 image.
    border_fraction : float, default=0.05
        Fraction of height/width to consider as border margins.

    Returns
    -------
    bool
        True if the background is light (>127.5 mean intensity), False if dark.
    """
    h, w = image.shape
    b_h = max(1, int(h * border_fraction))
    b_w = max(1, int(w * border_fraction))

    top = image[:b_h, :]
    bottom = image[-b_h:, :]
    left = image[:, :b_w]
    right = image[:, -b_w:]

    border_mean = (
        top.sum() + bottom.sum() + left.sum() + right.sum()
    ) / (top.size + bottom.size + left.size + right.size)

    return bool(border_mean > 127.5)


def binarize_handwriting(
    image: np.ndarray,
    method: str = "otsu",
    threshold_value: float = 30.0,
    polarity: str = "dark_background",
) -> np.ndarray:
    """Binarize a grayscale handwriting image into a consistent binary mask.

    Output Convention:
        - Handwriting strokes (foreground) = 255
        - Background = 0

    In the Mendeley Dysgraphia Handwriting Dataset, images are formatted with
    a black background (0) and light handwriting strokes (>0). When using the
    default 'dark_background' polarity, regular thresholding (THRESH_BINARY)
    correctly yields foreground=255 and background=0 without inverse thresholding.

    Parameters
    ----------
    image : np.ndarray
        2D single-channel uint8 grayscale image.
    method : str, default='otsu'
        Thresholding algorithm:
        - 'otsu': Otsu's automatic thresholding (justified for bimodal distributions).
        - 'triangle': Triangle automatic thresholding.
        - 'fixed': User-configured cut-off threshold defined by `threshold_value`.
    threshold_value : float, default=30.0
        Cut-off threshold when method='fixed'. Ignored for 'otsu' and 'triangle'.
    polarity : str, default='dark_background'
        Handwriting/background polarity:
        - 'dark_background': Dataset convention (strokes light, background dark).
        - 'light_background': Paper convention (strokes dark, background light).
        - 'auto': Automatically determined by border intensity sampling.

    Returns
    -------
    np.ndarray
        2D uint8 binary mask containing only values 0 and 255.

    Raises
    ------
    TypeError
        If image is not a numpy ndarray or has non-uint8 dtype.
    ValueError
        If image is empty, not 2D, or if method/polarity arguments are invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Input image array is empty (size 0)")

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, but received ndim={image.ndim} with shape {image.shape}."
        )

    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected image dtype np.uint8, but received {image.dtype}."
        )

    # Validate and resolve polarity
    valid_polarities = {"dark_background", "light_background", "auto"}
    if polarity not in valid_polarities:
        raise ValueError(
            f"Invalid polarity '{polarity}'. Allowed options: {sorted(valid_polarities)}"
        )

    if polarity == "auto":
        is_light_bg = detect_background_is_light(image)
    elif polarity == "light_background":
        is_light_bg = True
    else:
        is_light_bg = False

    # For dark background (light text): text >= T -> 255, background < T -> 0 (THRESH_BINARY)
    # For light background (dark text): text <= T -> 255, background > T -> 0 (THRESH_BINARY_INV)
    thresh_flag = cv2.THRESH_BINARY_INV if is_light_bg else cv2.THRESH_BINARY

    # Apply thresholding method
    method_lower = method.lower()
    if method_lower == "otsu":
        _, binary_mask = cv2.threshold(
            image, 0, 255, thresh_flag + cv2.THRESH_OTSU
        )
    elif method_lower == "triangle":
        _, binary_mask = cv2.threshold(
            image, 0, 255, thresh_flag + cv2.THRESH_TRIANGLE
        )
    elif method_lower == "fixed":
        if threshold_value < 0 or threshold_value > 255:
            raise ValueError(
                f"threshold_value must be in [0, 255], got {threshold_value}"
            )
        _, binary_mask = cv2.threshold(
            image, float(threshold_value), 255, thresh_flag
        )
    else:
        raise ValueError(
            f"Unsupported thresholding method: '{method}'. Choose 'otsu', 'triangle', or 'fixed'."
        )

    return binary_mask
