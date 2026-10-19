"""Tests for handwriting binarization module."""

from pathlib import Path
import numpy as np
import pytest

from src.preprocessing.image_loader import load_image
from src.preprocessing.binarization import (
    binarize_handwriting,
    detect_background_is_light,
)

DATASET_ROOT = Path("data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING")


@pytest.fixture
def pd_sample_gray() -> np.ndarray:
    """Fixture returning a grayscale image from Potential Dysgraphia."""
    sample_file = list((DATASET_ROOT / "Potential Dysgraphia").glob("*.jpg"))[0]
    return load_image(sample_file, as_gray=True)


@pytest.fixture
def lpd_sample_gray() -> np.ndarray:
    """Fixture returning a grayscale image from Low Potential Dysgraphia."""
    sample_file = list((DATASET_ROOT / "Low Potential Dysgraphia").glob("*.jpg"))[0]
    return load_image(sample_file, as_gray=True)


def test_binarize_mendeley_potential_dysgraphia(pd_sample_gray: np.ndarray):
    """Test binarization on Potential Dysgraphia sample preserves expected convention."""
    mask = binarize_handwriting(pd_sample_gray, method="otsu", polarity="dark_background")

    # Output mask must only contain 0 and 255
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals, "Mask must contain both background and foreground"

    # Background must be 0: Check corners which are background in this dataset
    assert mask[0, 0] == 0
    assert mask[0, -1] == 0
    assert mask[-1, 0] == 0
    assert mask[-1, -1] == 0

    # Foreground percentage should be reasonable for a line of text (between 3% and 30%)
    fg_ratio = (mask == 255).mean()
    assert 0.03 < fg_ratio < 0.30


def test_binarize_mendeley_low_potential_dysgraphia(lpd_sample_gray: np.ndarray):
    """Test binarization on Low Potential Dysgraphia sample preserves expected convention."""
    mask = binarize_handwriting(lpd_sample_gray, method="otsu", polarity="dark_background")

    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals

    assert mask[0, 0] == 0
    assert mask[-1, -1] == 0

    fg_ratio = (mask == 255).mean()
    assert 0.03 < fg_ratio < 0.30


def test_binarize_triangle_method(pd_sample_gray: np.ndarray):
    """Test triangle thresholding produces a valid binary mask."""
    mask = binarize_handwriting(pd_sample_gray, method="triangle", polarity="dark_background")
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals


def test_binarize_fixed_method(pd_sample_gray: np.ndarray):
    """Test fixed thresholding produces a valid binary mask."""
    mask = binarize_handwriting(pd_sample_gray, method="fixed", threshold_value=25.0, polarity="dark_background")
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals


def test_detect_background_polarity():
    """Test detect_background_is_light on synthetic light and dark images."""
    # Dark background with light stroke in center
    dark_bg = np.zeros((50, 50), dtype=np.uint8)
    dark_bg[20:30, 20:30] = 200
    assert detect_background_is_light(dark_bg) is False

    # Light background with dark stroke in center
    light_bg = np.full((50, 50), 240, dtype=np.uint8)
    light_bg[20:30, 20:30] = 30
    assert detect_background_is_light(light_bg) is True


def test_binarize_synthetic_light_and_dark_bg():
    """Verify consistent convention (handwriting=255, bg=0) for both background types."""
    # 1. Dark background image (light stroke)
    dark_img = np.zeros((40, 40), dtype=np.uint8)
    dark_img[15:25, 15:25] = 220
    mask_dark = binarize_handwriting(dark_img, method="otsu", polarity="dark_background")
    assert mask_dark[0, 0] == 0
    assert mask_dark[20, 20] == 255

    # 2. Light background image (dark stroke)
    light_img = np.full((40, 40), 230, dtype=np.uint8)
    light_img[15:25, 15:25] = 20
    mask_light = binarize_handwriting(light_img, method="otsu", polarity="light_background")
    assert mask_light[0, 0] == 0
    assert mask_light[20, 20] == 255

    # 3. Auto polarity resolution
    mask_auto_dark = binarize_handwriting(dark_img, method="otsu", polarity="auto")
    assert mask_auto_dark[0, 0] == 0
    assert mask_auto_dark[20, 20] == 255

    mask_auto_light = binarize_handwriting(light_img, method="otsu", polarity="auto")
    assert mask_auto_light[0, 0] == 0
    assert mask_auto_light[20, 20] == 255


def test_binarize_invalid_inputs():
    """Test clear error handling for all invalid inputs."""
    valid_img = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        binarize_handwriting("not_an_array")  # type: ignore

    with pytest.raises(ValueError, match="empty"):
        binarize_handwriting(np.array([], dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected a 2D grayscale image"):
        binarize_handwriting(np.zeros((20, 20, 3), dtype=np.uint8))

    with pytest.raises(TypeError, match="Expected image dtype np.uint8"):
        binarize_handwriting(np.zeros((20, 20), dtype=np.float32))  # type: ignore

    with pytest.raises(ValueError, match="Invalid polarity"):
        binarize_handwriting(valid_img, polarity="unknown_polarity")

    with pytest.raises(ValueError, match="Unsupported thresholding method"):
        binarize_handwriting(valid_img, method="unsupported_method")

    with pytest.raises(ValueError, match="threshold_value must be in"):
        binarize_handwriting(valid_img, method="fixed", threshold_value=300)
