"""CNN Dataset module for GraphiSense.

Handles dataset discovery, label mapping, transforms, and data splits.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.image_loader import load_image

# Default dataset root - configurable
DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "raw"

# Class mapping: folder name -> class index
CLASS_TO_IDX = {
    "Potential Dysgraphia": 1,  # PD
    "Low Potential Dysgraphia": 0,  # LPD
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IDX_TO_LABEL = {1: "PD", 0: "LPD"}

# ImageNet normalization for pretrained ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_candidate_dataset_roots(root: Path) -> List[Path]:
    """Find all directories under root that contain the expected class folders.
    
    Returns a list of paths that contain both 'Potential Dysgraphia' and 
    'Low Potential Dysgraphia' subdirectories with images.
    """
    candidates = []
    for folder_name in CLASS_TO_IDX.keys():
        matches = list(root.rglob(folder_name))
        for match in matches:
            if match.is_dir():
                # Check if sibling class folder exists
                parent = match.parent
                sibling_name = [n for n in CLASS_TO_IDX.keys() if n != folder_name][0]
                sibling = parent / sibling_name
                if sibling.is_dir():
                    # Check if both have supported images
                    has_images = False
                    for cls_folder in [match, sibling]:
                        images = list(cls_folder.iterdir())
                        if any(f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file() for f in images):
                            has_images = True
                            break
                    if has_images:
                        candidates.append(parent)
    # Deduplicate
    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)
    return unique_candidates


def discover_dataset(
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
) -> Tuple[List[Path], List[int]]:
    """Discover all images and their labels in the dataset.

    Supports two dataset structures:
    1. Class folders directly under dataset_root (e.g., data/raw/Potential Dysgraphia/)
    2. Class folders nested under a subdirectory (e.g., data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING/)

    Args:
        dataset_root: Root directory to search for class subdirectories.

    Returns:
        Tuple of (image_paths, labels) where labels are class indices (0=LPD, 1=PD).

    Raises:
        FileNotFoundError: If dataset root or class directories are missing.
        ValueError: If multiple candidate datasets found and no explicit selection.
    """
    root = Path(dataset_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    # First, try direct class folders under root
    direct_found = True
    for folder_name in CLASS_TO_IDX.keys():
        folder_path = root / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            direct_found = False
            break
        # Check if it has any supported images
        images = list(folder_path.iterdir())
        if not any(f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file() for f in images):
            direct_found = False
            break

    if direct_found:
        search_root = root
    else:
        # Search for nested dataset structures
        candidates = _find_candidate_dataset_roots(root)
        
        if not candidates:
            raise FileNotFoundError(
                f"No valid dataset found under {root}. "
                f"Expected class folders: {list(CLASS_TO_IDX.keys())} "
                f"directly under root or nested within a subdirectory."
            )
        
        if len(candidates) > 1:
            # Multiple candidates - report and require explicit selection
            candidate_strs = "\n  ".join(str(c.relative_to(root)) for c in candidates)
            raise ValueError(
                f"Multiple candidate datasets found under {root}:\n  {candidate_strs}\n"
                f"Please specify the exact dataset root path (e.g., "
                f"'data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING')."
            )
        
        search_root = candidates[0]

    image_paths = []
    labels = []

    for folder_name, class_idx in CLASS_TO_IDX.items():
        folder_path = search_root / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(
                f"Class directory not found: {folder_path}. "
                f"Expected class folders: {list(CLASS_TO_IDX.keys())}"
            )
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Class path is not a directory: {folder_path}")

        folder_images = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
        ])

        if not folder_images:
            raise FileNotFoundError(
                f"No supported images found in {folder_path}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        image_paths.extend(folder_images)
        labels.extend([class_idx] * len(folder_images))

    return image_paths, labels


def get_class_distribution(labels: List[int]) -> Dict[str, int]:
    """Get class distribution as a dictionary."""
    unique, counts = np.unique(labels, return_counts=True)
    dist = {}
    for u, c in zip(unique, counts):
        dist[IDX_TO_LABEL[u]] = int(c)
    return dist


class HandwritingDataset(Dataset):
    """PyTorch Dataset for handwriting images."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the dataset.

        Args:
            image_paths: List of paths to image files.
            labels: List of class indices (0=LPD, 1=PD).
            transform: Optional torchvision transforms to apply.
        """
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image using existing loader (handles non-ASCII paths, etc.)
        try:
            bgr = load_image(img_path, as_gray=False)
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {e}")

        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        pil_image = Image.fromarray(rgb)

        if self.transform:
            tensor = self.transform(pil_image)
        else:
            # Default: just convert to tensor
            tensor = transforms.ToTensor()(pil_image)

        return tensor, label


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> transforms.Compose:
    """Get transforms for training or evaluation.

    Args:
        image_size: Target image size (square).
        is_training: Whether to include training augmentations.
        use_imagenet_norm: Whether to use ImageNet normalization.

    Returns:
        Composed transforms.
    """
    if use_imagenet_norm:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_data_splits(
    image_paths: List[Path],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    participant_ids: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/val/test splits with optional participant-level grouping.

    Args:
        image_paths: List of image paths.
        labels: List of class indices.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        participant_ids: Optional list of participant identifiers for each image.
                         If provided, splits are done at participant level.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(image_paths)
    indices = np.arange(n)

    if participant_ids is not None:
        # Participant-level split: group by participant ID
        unique_participants = np.unique(participant_ids)
        n_participants = len(unique_participants)

        np.random.seed(seed)
        np.random.shuffle(unique_participants)

        n_train = int(n_participants * train_ratio)
        n_val = int(n_participants * val_ratio)

        train_participants = set(unique_participants[:n_train])
        val_participants = set(unique_participants[n_train:n_train + n_val])
        test_participants = set(unique_participants[n_train + n_val:])

        train_indices = [i for i, pid in enumerate(participant_ids) if pid in train_participants]
        val_indices = [i for i, pid in enumerate(participant_ids) if pid in val_participants]
        test_indices = [i for i, pid in enumerate(participant_ids) if pid in test_participants]
    else:
        # Standard stratified image-level split
        from sklearn.model_selection import train_test_split

        train_indices, temp_indices = train_test_split(
            indices, train_size=train_ratio, stratify=labels, random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            stratify=[labels[i] for i in temp_indices],
            random_state=seed,
        )

    return train_indices, val_indices, test_indices


def create_dataloaders(
    dataset: Dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Args:
        dataset: The full dataset.
        train_indices: Training set indices.
        val_indices: Validation set indices.
        test_indices: Test set indices.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers (0 for Windows compatibility).
        pin_memory: Whether to pin memory for GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def save_split_metadata(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    image_paths: List[Path],
    labels: List[int],
    output_path: Union[str, Path],
    config: Dict[str, Any],
) -> None:
    """Save split metadata for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def to_list(arr):
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return list(arr)

    metadata = {
        "train_indices": to_list(train_indices),
        "val_indices": to_list(val_indices),
        "test_indices": to_list(test_indices),
        "image_paths": [str(p) for p in image_paths],
        "labels": to_list(labels),
        "class_mapping": IDX_TO_LABEL,
        "config": config,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_split_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load split metadata."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Need to import cv2 for the dataset class
import cv2