"""Tests for Streamlit application pipeline handler."""

from pathlib import Path
import numpy as np
import pytest

from app.pipeline_handler import (
    decode_image_bytes,
    load_and_preprocess_image,
    run_app_inference,
)

DATASET_ROOT = Path("data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING")


@pytest.fixture
def pd_image_bytes() -> bytes:
    """Fixture returning bytes of a valid PD sample image."""
    sample_path = next((DATASET_ROOT / "Potential Dysgraphia").glob("*.jpg"))
    return sample_path.read_bytes()


@pytest.fixture
def lpd_image_bytes() -> bytes:
    """Fixture returning bytes of a valid LPD sample image."""
    sample_path = next((DATASET_ROOT / "Low Potential Dysgraphia").glob("*.jpg"))
    return sample_path.read_bytes()


def test_app_inference_pd_sample(pd_image_bytes: bytes):
    """Verify application pipeline on representative Potential Dysgraphia sample."""
    res = run_app_inference(pd_image_bytes, filename="sample_pd.jpg", polarity="dark_background")
    assert res["predicted_label"] in ["PD", "LPD"]
    assert "class_probabilities" in res
    assert res["binary_mask"].shape[:2] == res["original_rgb"].shape[:2]
    assert set(np.unique(res["binary_mask"])).issubset({0, 255})
    assert 0.0 <= res["class_probabilities"]["PD"] <= 1.0


def test_app_inference_lpd_sample(lpd_image_bytes: bytes):
    """Verify application pipeline on representative Low Potential Dysgraphia sample."""
    res = run_app_inference(lpd_image_bytes, filename="sample_lpd.jpg", polarity="dark_background")
    assert res["predicted_label"] in ["PD", "LPD"]
    assert "class_probabilities" in res
    assert 0.0 <= res["class_probabilities"]["LPD"] <= 1.0


def test_app_corrupt_file_handling():
    """Verify app handler gracefully catches corrupt image bytes."""
    corrupt_bytes = b"This is not a real JPEG image binary sequence."
    with pytest.raises(ValueError, match="Failed to decode image bytes"):
        load_and_preprocess_image(corrupt_bytes)


def test_app_missing_model_handling(pd_image_bytes: bytes, tmp_path: Path):
    """Verify app handler raises FileNotFoundError if model artifact is missing."""
    missing_model = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        run_app_inference(pd_image_bytes, model_path=missing_model)


def test_load_and_preprocess_polarity_modes(pd_image_bytes: bytes):
    """Verify auto-detection and explicit polarity settings."""
    _, _, mask_auto, is_light_auto = load_and_preprocess_image(pd_image_bytes, polarity="auto")
    assert not is_light_auto  # Mendeley dataset is dark background

    _, _, mask_dark, is_light_dark = load_and_preprocess_image(pd_image_bytes, polarity="dark_background")
    assert not is_light_dark
    assert np.array_equal(mask_auto, mask_dark)


def test_app_empty_bytes_handling():
    """Verify app handler gracefully catches empty image bytes."""
    with pytest.raises(ValueError, match="Image bytes are empty"):
        load_and_preprocess_image(b"")


def test_app_inference_from_filepath():
    """Verify run_app_inference works when passed a Path directly."""
    sample_path = next((DATASET_ROOT / "Potential Dysgraphia").glob("*.jpg"))
    res = run_app_inference(sample_path)
    assert res["predicted_label"] in ["PD", "LPD"]
    assert "PD" in res["class_probabilities"]


def test_app_invalid_filepath_handling():
    """Verify load_and_preprocess_image raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_and_preprocess_image("non_existent_file_path.png")
