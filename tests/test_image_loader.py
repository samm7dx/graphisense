"""Tests for image loading and grayscale conversion."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from src.preprocessing.image_loader import (
    load_image,
    load_image_with_grayscale,
    to_grayscale,
)


@pytest.fixture
def sample_image_path() -> Path:
    """Fixture returning path to a sample image from the Mendeley dataset."""
    root = Path("data/raw")
    if not root.exists():
        pytest.skip("External handwriting dataset is not installed")
    pd_images = list((root / "Potential Dysgraphia").glob("*.jpg"))
    if not pd_images:
        pytest.skip("Potential Dysgraphia sample images are not available")
    return pd_images[0]


def test_load_image_success(sample_image_path: Path):
    """Test loading an image returns a valid 3-channel array."""
    img = load_image(sample_image_path)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8
    assert img.size > 0


def test_load_image_as_gray(sample_image_path: Path):
    """Test loading an image with as_gray=True returns a 2D grayscale array."""
    gray = load_image(sample_image_path, as_gray=True)
    assert isinstance(gray, np.ndarray)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray.shape == sample_image_path.name and gray.size > 0 or gray.ndim == 2


def test_load_image_with_grayscale(sample_image_path: Path):
    """Test load_image_with_grayscale returns both color and 2D grayscale."""
    original, gray = load_image_with_grayscale(sample_image_path)
    assert original.ndim == 3
    assert gray.ndim == 2
    assert original.shape[:2] == gray.shape


def test_load_image_nonexistent_raises():
    """Test loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        load_image("non_existent_image_12345.jpg")


def test_load_image_directory_raises(tmp_path: Path):
    """Test attempting to load a directory raises ValueError."""
    with pytest.raises(ValueError, match="Path is not a regular file"):
        load_image(tmp_path)


def test_load_image_corrupt_file_raises(tmp_path: Path):
    """Test attempting to decode a non-image file raises IOError."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_text("This is not a real image binary file.")
    with pytest.raises(IOError, match="failed to decode"):
        load_image(corrupt_file)


def test_to_grayscale_from_2d():
    """Test passing an already 2D array returns a copy."""
    arr = np.full((10, 10), 128, dtype=np.uint8)
    res = to_grayscale(arr)
    assert res.shape == (10, 10)
    assert np.array_equal(res, arr)


def test_to_grayscale_from_single_channel_3d():
    """Test passing a 3D (H, W, 1) array squeezes to 2D."""
    arr = np.full((10, 10, 1), 77, dtype=np.uint8)
    res = to_grayscale(arr)
    assert res.shape == (10, 10)
    assert res[0, 0] == 77


def test_to_grayscale_from_bgr():
    """Test 3-channel BGR to grayscale conversion."""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[:, :] = [255, 0, 0]  # pure blue
    res = to_grayscale(bgr)
    assert res.shape == (10, 10)
    assert res.ndim == 2


def test_to_grayscale_from_bgra():
    """Test 4-channel BGRA to grayscale conversion."""
    bgra = np.full((10, 10, 4), 200, dtype=np.uint8)
    res = to_grayscale(bgra)
    assert res.shape == (10, 10)
    assert res.ndim == 2


def test_to_grayscale_invalid_inputs():
    """Test invalid inputs to to_grayscale."""
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        to_grayscale([1, 2, 3])  # type: ignore

    with pytest.raises(ValueError, match="empty"):
        to_grayscale(np.array([], dtype=np.uint8))

    with pytest.raises(ValueError, match="Unsupported channel count"):
        to_grayscale(np.zeros((10, 10, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="Unsupported image dimension"):
        to_grayscale(np.zeros((2, 2, 2, 2), dtype=np.uint8))
