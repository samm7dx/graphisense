"""Tests for CNN dataset discovery and handling."""

import tempfile
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cnn_dataset import (
    discover_dataset,
    CLASS_TO_IDX,
    SUPPORTED_EXTENSIONS,
)


def create_synthetic_image(path: Path, size: tuple = (64, 64)) -> None:
    """Create a simple synthetic image for testing."""
    import cv2
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (size[0]-10, size[1]-10), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)


class TestDatasetDiscovery:
    """Tests for dataset discovery functionality."""

    def test_direct_root_discovery(self, tmp_path: Path):
        """Test discovery when class folders are directly under root."""
        # Create class folders directly under root
        for folder_name in CLASS_TO_IDX.keys():
            class_dir = tmp_path / folder_name
            class_dir.mkdir()
            create_synthetic_image(class_dir / "sample1.jpg")
            create_synthetic_image(class_dir / "sample2.jpg")

        paths, labels = discover_dataset(tmp_path)
        
        assert len(paths) == 4  # 2 images per class
        assert labels.count(0) == 2  # LPD
        assert labels.count(1) == 2  # PD
        # Check all images are from correct class folders
        for path, label in zip(paths, labels):
            expected_folder = "Low Potential Dysgraphia" if label == 0 else "Potential Dysgraphia"
            assert expected_folder in str(path)

    def test_nested_dataset_discovery(self, tmp_path: Path):
        """Test discovery when class folders are nested under a subdirectory."""
        # Create nested structure: root/dataset_name/class_folders
        dataset_dir = tmp_path / "handwriting" / "extracted" / "DATASET DYSGRAPHIA HANDWRITING"
        dataset_dir.mkdir(parents=True)
        
        for folder_name in CLASS_TO_IDX.keys():
            class_dir = dataset_dir / folder_name
            class_dir.mkdir()
            create_synthetic_image(class_dir / "sample1.jpg")
            create_synthetic_image(class_dir / "sample2.jpg")

        # Pass the parent root (tmp_path), should find nested dataset
        paths, labels = discover_dataset(tmp_path)
        
        assert len(paths) == 4
        assert labels.count(0) == 2
        assert labels.count(1) == 2

    def test_missing_class_folder_raises_error(self, tmp_path: Path):
        """Test that missing class folder raises FileNotFoundError."""
        # Create only one class folder
        class_dir = tmp_path / "Potential Dysgraphia"
        class_dir.mkdir()
        create_synthetic_image(class_dir / "sample1.jpg")

        with pytest.raises(FileNotFoundError, match="No valid dataset found"):
            discover_dataset(tmp_path)

    def test_no_supported_images_raises_error(self, tmp_path: Path):
        """Test that empty class folder raises FileNotFoundError."""
        for folder_name in CLASS_TO_IDX.keys():
            class_dir = tmp_path / folder_name
            class_dir.mkdir()
            # Create a text file instead of image
            (class_dir / "readme.txt").write_text("not an image")

        with pytest.raises(FileNotFoundError, match="No valid dataset found"):
            discover_dataset(tmp_path)

    def test_ambiguous_nested_datasets_raises_error(self, tmp_path: Path):
        """Test that multiple nested datasets raise ValueError."""
        # Create two nested datasets
        for dataset_name in ["Dataset_A", "Dataset_B"]:
            dataset_dir = tmp_path / dataset_name
            dataset_dir.mkdir()
            for folder_name in CLASS_TO_IDX.keys():
                class_dir = dataset_dir / folder_name
                class_dir.mkdir()
                create_synthetic_image(class_dir / "sample1.jpg")

        with pytest.raises(ValueError, match="Multiple candidate datasets found"):
            discover_dataset(tmp_path)

    def test_explicit_nested_path_works(self, tmp_path: Path):
        """Test that explicitly passing the nested dataset path works."""
        # Create two nested datasets
        for dataset_name in ["Dataset_A", "Dataset_B"]:
            dataset_dir = tmp_path / dataset_name
            dataset_dir.mkdir()
            for folder_name in CLASS_TO_IDX.keys():
                class_dir = dataset_dir / folder_name
                class_dir.mkdir()
                create_synthetic_image(class_dir / "sample1.jpg")

        # Explicitly pass one of the dataset paths
        explicit_path = tmp_path / "Dataset_A"
        paths, labels = discover_dataset(explicit_path)
        
        assert len(paths) == 2
        assert labels.count(0) == 1
        assert labels.count(1) == 1

    def test_nonexistent_root_raises_error(self, tmp_path: Path):
        """Test that nonexistent root raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Dataset root not found"):
            discover_dataset(nonexistent)

    def test_not_a_directory_raises_error(self, tmp_path: Path):
        """Test that file path raises NotADirectoryError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        with pytest.raises(NotADirectoryError):
            discover_dataset(file_path)

    def test_unsupported_extension_ignored(self, tmp_path: Path):
        """Test that unsupported file extensions are ignored."""
        for folder_name in CLASS_TO_IDX.keys():
            class_dir = tmp_path / folder_name
            class_dir.mkdir()
            create_synthetic_image(class_dir / "sample1.jpg")
            # Create unsupported file
            (class_dir / "readme.txt").write_text("text")

        paths, labels = discover_dataset(tmp_path)
        
        # Should only find the .jpg files
        assert len(paths) == 2
        for path in paths:
            assert path.suffix.lower() in SUPPORTED_EXTENSIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])