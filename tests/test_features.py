"""Tests for static handwriting feature extraction and dataset building."""

import csv
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.preprocessing.image_loader import load_image
from src.features.handwriting_features import extract_handwriting_features
from src.features.build_feature_dataset import build_feature_table, CLASS_MAPPING
from src.models.train_evaluate import FEATURE_SET_B


DATASET_ROOT = Path("data/raw")
OUTPUT_CSV = Path("outputs/features/features.csv")


def test_synthetic_known_geometry_and_ink_density():
    """Verify feature calculation on a synthetic binary image with known geometry."""
    # 100 height, 200 width
    h, w = 100, 200
    img = np.zeros((h, w), dtype=np.uint8)

    # Add a rectangle of height 20, width 50 (area = 1000)
    top, left = 40, 75
    rect_h, rect_w = 20, 50
    img[top : top + rect_h, left : left + rect_w] = 255

    feats = extract_handwriting_features(img, is_binary=True, min_component_area=5)

    assert feats["image_width"] == 200
    assert feats["image_height"] == 100
    assert pytest.approx(feats["aspect_ratio"], 0.001) == 2.0
    assert feats["foreground_pixels"] == 1000
    assert pytest.approx(feats["ink_density"], 0.0001) == 1000 / 20000  # 0.05

    # Connected component check
    assert feats["component_count"] == 1
    assert pytest.approx(feats["component_width_mean"], 0.1) == rect_w
    assert pytest.approx(feats["component_height_mean"], 0.1) == rect_h
    assert pytest.approx(feats["component_area_mean"], 0.1) == 1000

    # Centroid check
    expected_cx = left + (rect_w - 1) / 2.0
    expected_cy = top + (rect_h - 1) / 2.0
    assert pytest.approx(feats["centroid_x"], 0.1) == expected_cx
    assert pytest.approx(feats["centroid_y"], 0.1) == expected_cy
    assert pytest.approx(feats["centroid_x_norm"], 0.01) == expected_cx / w
    assert pytest.approx(feats["centroid_y_norm"], 0.01) == expected_cy / h

    # Stroke width proxy (for rect_h=20, max inscribed radius is 10, stroke width ~20)
    assert 15.0 <= feats["stroke_width_mean"] <= 21.0


def test_empty_foreground_mask():
    """Verify empty mask does not crash and returns documented neutral feature values."""
    empty_mask = np.zeros((80, 120), dtype=np.uint8)
    feats = extract_handwriting_features(empty_mask, is_binary=True)

    assert feats["image_width"] == 120
    assert feats["image_height"] == 80
    assert feats["foreground_pixels"] == 0
    assert feats["ink_density"] == 0.0
    assert feats["component_count"] == 0
    assert feats["component_width_mean"] == 0.0
    assert feats["component_height_mean"] == 0.0
    assert feats["centroid_x"] == 0.0
    assert feats["centroid_y"] == 0.0
    assert feats["active_row_ratio"] == 0.0
    assert feats["stroke_width_mean"] == 0.0


def test_connected_component_filtering():
    """Verify that min_component_area correctly filters noise specks."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Component 1: 10x10 block = area 100
    mask[10:20, 10:20] = 255
    # Component 2: 1x2 noise = area 2
    mask[50:51, 50:52] = 255
    # Component 3: 1x3 noise = area 3
    mask[70:71, 70:73] = 255

    # With min_area=5, only the 10x10 component should be retained
    feats_filtered = extract_handwriting_features(mask, is_binary=True, min_component_area=5)
    assert feats_filtered["component_count"] == 1
    assert pytest.approx(feats_filtered["component_area_mean"], 0.1) == 100.0

    # With min_area=1, all 3 components should be counted
    feats_all = extract_handwriting_features(mask, is_binary=True, min_component_area=1)
    assert feats_all["component_count"] == 3


def test_invalid_input_errors():
    """Verify input validation and informative exceptions."""
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        extract_handwriting_features("invalid_type")  # type: ignore

    with pytest.raises(ValueError, match="empty"):
        extract_handwriting_features(np.array([]))

    with pytest.raises(ValueError, match="min_component_area must be >= 0"):
        extract_handwriting_features(np.zeros((10, 10), dtype=np.uint8), min_component_area=-1)

    # 3D array when is_binary=True
    with pytest.raises(ValueError, match="Binary mask must be 2D"):
        extract_handwriting_features(np.zeros((10, 10, 3), dtype=np.uint8), is_binary=True)

    # Binary mask containing non {0, 255} values
    invalid_mask = np.full((10, 10), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="Binary mask must contain only values in {0, 255}"):
        extract_handwriting_features(invalid_mask, is_binary=True)


def test_representative_images_from_both_classes():
    """Verify feature extraction on real dataset images from both classes."""
    if not DATASET_ROOT.exists():
        pytest.skip("External handwriting dataset is not installed")
    for folder_name, expected_label in CLASS_MAPPING.items():
        sample_path = next((DATASET_ROOT / folder_name).glob("*.jpg"), None)
        if sample_path is None:
            pytest.skip(f"Dataset images are missing for class {folder_name}")
        img = load_image(sample_path, as_gray=True)

        feats = extract_handwriting_features(
            img,
            is_binary=False,
            min_component_area=5,
            binarize_kwargs={"method": "otsu", "polarity": "dark_background"},
        )

        assert feats["image_width"] > 0
        assert feats["image_height"] > 0
        assert feats["aspect_ratio"] > 1.0
        assert 0.01 <= feats["ink_density"] <= 0.35
        assert feats["component_count"] > 0
        assert feats["component_area_mean"] > 0.0
        assert 0.0 <= feats["centroid_x_norm"] <= 1.0
        assert 0.0 <= feats["centroid_y_norm"] <= 1.0
        assert feats["stroke_width_mean"] > 0.0


def test_features_csv_integrity():
    """Verify outputs/features/features.csv row count, labels, status and no missing values."""
    if not OUTPUT_CSV.exists():
        pytest.skip("Generated feature dataset is not available")
    assert OUTPUT_CSV.exists(), "outputs/features/features.csv does not exist"

    with open(OUTPUT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Expected exact count of 249 images
    assert len(rows) == 249, f"Expected 249 rows, got {len(rows)}"

    pd_count = sum(1 for r in rows if r["class_label"] == "PD")
    lpd_count = sum(1 for r in rows if r["class_label"] == "LPD")
    assert pd_count == 114, f"Expected 114 PD rows, got {pd_count}"
    assert lpd_count == 135, f"Expected 135 LPD rows, got {lpd_count}"

    for i, row in enumerate(rows):
        assert row["status"] == "SUCCESS", f"Row {i} failed: {row['status']}"
        # Check that no feature values are empty
        for col, val in row.items():
            assert val != "", f"Missing value in row {i} column {col}"
            if col not in ["filename", "class_label", "image_path", "status"]:
                float_val = float(val)
                assert not np.isnan(float_val), f"NaN value in row {i} column {col}"


def test_build_feature_table_reports_missing_dataset(tmp_path: Path):
    """Missing external data should produce instructions instead of an empty CSV."""
    missing_dataset = tmp_path / "missing-dataset"
    with pytest.raises(FileNotFoundError, match="Download and extract the dataset"):
        build_feature_table(dataset_dir=missing_dataset, output_csv=tmp_path / "features.csv")


def test_build_feature_table_requires_both_class_folders(tmp_path: Path):
    """A partial extraction should identify the absent class directory."""
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "Potential Dysgraphia").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Low Potential Dysgraphia"):
        build_feature_table(dataset_dir=dataset_dir, output_csv=tmp_path / "features.csv")


def test_build_feature_table_writes_training_schema(tmp_path: Path):
    """Synthetic class folders should produce the feature CSV schema expected downstream."""
    dataset_dir = tmp_path / "dataset"
    for folder_name in CLASS_MAPPING:
        class_dir = dataset_dir / folder_name
        class_dir.mkdir(parents=True)
        image = np.zeros((48, 96), dtype=np.uint8)
        cv2.line(image, (8, 24), (88, 24), 255, 3)
        assert cv2.imwrite(str(class_dir / "sample.jpg"), image)

    output_csv = tmp_path / "outputs" / "features.csv"
    rows = build_feature_table(
        dataset_dir=dataset_dir,
        output_csv=output_csv,
        repo_root=tmp_path,
    )

    assert output_csv.is_file()
    assert len(rows) == 2
    assert all(row["status"] == "SUCCESS" for row in rows)
    with open(output_csv, newline="", encoding="utf-8") as feature_file:
        header = next(csv.reader(feature_file))
    assert set(FEATURE_SET_B).issubset(header)
