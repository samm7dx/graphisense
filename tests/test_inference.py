"""Tests for single-image inference pipeline."""

from pathlib import Path
import json
import joblib
import numpy as np
import pytest

from src.models.inference import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_saved_model,
    predict_handwriting,
)
from src.models.train_evaluate import REDUCED_STATIC_FEATURES

DATASET_ROOT = Path("data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING")


@pytest.fixture
def sample_pd_path() -> Path:
    """Fixture returning a valid PD image path."""
    return next((DATASET_ROOT / "Potential Dysgraphia").glob("*.jpg"))


@pytest.fixture
def sample_lpd_path() -> Path:
    """Fixture returning a valid LPD image path."""
    return next((DATASET_ROOT / "Low Potential Dysgraphia").glob("*.jpg"))


def test_predict_handwriting_pd_sample(sample_pd_path: Path):
    """Verify inference on a representative Potential Dysgraphia sample."""
    res = predict_handwriting(sample_pd_path)
    assert res["predicted_label"] in ["PD", "LPD"]
    assert res["predicted_class_index"] in [0, 1]
    assert "class_probabilities" in res
    assert 0.0 <= res["class_probabilities"]["PD"] <= 1.0
    assert 0.0 <= res["class_probabilities"]["LPD"] <= 1.0
    assert pytest.approx(sum(res["class_probabilities"].values()), 0.001) == 1.0
    assert "disclaimer" in res


def test_predict_handwriting_lpd_sample(sample_lpd_path: Path):
    """Verify inference on a representative Low Potential Dysgraphia sample."""
    res = predict_handwriting(sample_lpd_path)
    assert res["predicted_label"] in ["PD", "LPD"]
    assert "class_probabilities" in res
    assert pytest.approx(sum(res["class_probabilities"].values()), 0.001) == 1.0


def test_feature_column_ordering(sample_pd_path: Path):
    """Verify feature dictionary strictly matches the metadata schema ordering."""
    res = predict_handwriting(sample_pd_path)
    features_used = res["features_used"]

    _, metadata = load_saved_model()
    expected_order = metadata["feature_names"]

    assert list(features_used.keys()) == expected_order
    assert len(features_used) == len(REDUCED_STATIC_FEATURES)


def test_missing_and_corrupt_artifacts(tmp_path: Path, sample_pd_path: Path):
    """Verify informative errors on missing or incompatible model artifacts."""
    # Missing model file
    missing_model = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        predict_handwriting(sample_pd_path, model_path=missing_model)

    # Missing metadata file
    missing_meta = tmp_path / "non_existent_meta.json"
    with pytest.raises(FileNotFoundError, match="Model metadata file not found"):
        predict_handwriting(sample_pd_path, metadata_path=missing_meta)

    # Corrupted metadata JSON
    bad_meta = tmp_path / "bad_meta.json"
    bad_meta.write_text("{ corrupt json ...")
    with pytest.raises(ValueError, match="Failed to parse metadata JSON"):
        load_saved_model(metadata_path=bad_meta)

    # Incompatible metadata schema
    empty_meta = tmp_path / "empty_meta.json"
    empty_meta.write_text(json.dumps({"some_key": "val"}))
    with pytest.raises(ValueError, match="missing required fields"):
        load_saved_model(metadata_path=empty_meta)

    # Corrupt model without predict method
    non_model_file = tmp_path / "invalid_model.joblib"
    joblib.dump({"not_a": "model"}, non_model_file)
    with pytest.raises(ValueError, match="does not implement required 'predict' method"):
        load_saved_model(model_path=non_model_file)


def test_unreadable_and_invalid_images(tmp_path: Path):
    """Verify errors on unreadable image paths and invalid formats."""
    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Handwriting image not found"):
        predict_handwriting("missing_image_file.jpg")

    # Directory instead of file
    with pytest.raises(ValueError, match="Path is not a regular file"):
        predict_handwriting(tmp_path)

    # Unsupported extension
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("plain text")
    with pytest.raises(ValueError, match="Unsupported image file extension"):
        predict_handwriting(txt_file)

    # Empty 0-byte image file
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="Image file is empty"):
        predict_handwriting(empty_file)

    # Corrupt / non-image file with valid extension
    corrupt_file = tmp_path / "fake_img.jpg"
    corrupt_file.write_text("not an image binary")
    with pytest.raises(IOError, match="failed to decode"):
        predict_handwriting(corrupt_file)


def test_deterministic_inference(sample_pd_path: Path):
    """Verify repeated inference on the same input yields identical predictions."""
    res1 = predict_handwriting(sample_pd_path)
    res2 = predict_handwriting(sample_pd_path)

    assert res1["predicted_label"] == res2["predicted_label"]
    assert res1["predicted_class_index"] == res2["predicted_class_index"]
    assert res1["class_probabilities"] == res2["class_probabilities"]
    assert res1["features_used"] == res2["features_used"]


def test_model_serialization_and_reload(tmp_path: Path, sample_pd_path: Path):
    """Verify re-saved pipeline artifact can be loaded and produces consistent inference."""
    orig_pipeline, orig_meta = load_saved_model()

    temp_model_file = tmp_path / "temp_pipeline.joblib"
    temp_meta_file = tmp_path / "temp_metadata.json"

    joblib.dump(orig_pipeline, temp_model_file)
    with open(temp_meta_file, "w", encoding="utf-8") as f:
        json.dump(orig_meta, f)

    res_orig = predict_handwriting(sample_pd_path)
    res_temp = predict_handwriting(
        sample_pd_path, model_path=temp_model_file, metadata_path=temp_meta_file
    )

    assert res_orig["predicted_label"] == res_temp["predicted_label"]
    assert res_orig["class_probabilities"] == res_temp["class_probabilities"]
