"""CNN Inference module for GraphiSense.

Checkpoint-based inference for ResNet-18 handwriting classification.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cnn_dataset import get_transforms
from src.models.cnn_model import (
    ResNet18Binary,
    create_model,
    load_checkpoint,
)


DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "models" / "cnn"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "best_model.pt"

# Class mapping must match training
CLASS_MAPPING = {"0": "LPD", "1": "PD"}
IDX_TO_LABEL = {0: "LPD", 1: "PD"}
LABEL_TO_IDX = {"LPD": 0, "PD": 1}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CNNInference:
    """CNN inference engine for handwriting classification."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = 224,
    ):
        """Initialize the inference engine.

        Args:
            checkpoint_path: Path to the model checkpoint.
            device: Device to run inference on.
            image_size: Input image size for transforms.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model, self.metadata = load_checkpoint(
            self.checkpoint_path,
            device=self.device,
            num_classes=2,
        )
        self.transform = get_transforms(image_size=image_size, is_training=False)

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed tensor ready for model input (1, C, H, W).
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {path.suffix}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is empty: {path}")

        # Load image
        pil_image = Image.open(path).convert("RGB")

        # Apply transforms
        tensor = self.transform(pil_image)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def predict(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.
            return_probabilities: Whether to include class probabilities.

        Returns:
            Dictionary with prediction results.
        """
        # Preprocess
        input_tensor = self.preprocess_image(image_path).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(probabilities))

        predicted_label = IDX_TO_LABEL[predicted_idx]
        class_probas = {
            "LPD": float(probabilities[0]),
            "PD": float(probabilities[1]),
        }

        result = {
            "image_path": str(Path(image_path).resolve()),
            "predicted_label": predicted_label,
            "predicted_class_index": predicted_idx,
            "model_type": "ResNet-18",
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "image_size": self.image_size,
            "disclaimer": (
                "GraphiSense CNN is an academic research prototype. "
                "Predicted probabilities are statistical model estimates, not clinical "
                "risk scores or a medical diagnosis of dysgraphia."
            ),
        }

        if return_probabilities:
            result["class_probabilities"] = class_probas

        return result

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        return_probabilities: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run inference on multiple images.

        Args:
            image_paths: List of image paths.
            return_probabilities: Whether to include probabilities.

        Returns:
            List of prediction dictionaries.
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": str(Path(path).resolve()),
                    "error": str(e),
                    "predicted_label": "ERROR",
                })
        return results


def predict_handwriting_cnn(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Convenience function for single-image CNN inference.

    Args:
        image_path: Path to the handwriting image.
        checkpoint_path: Path to model checkpoint (uses default if None).
        device: Device for inference.
        image_size: Input image size.

    Returns:
        Prediction dictionary.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    engine = CNNInference(checkpoint_path, device, image_size)
    return engine.predict(image_path)


def check_cnn_availability() -> Tuple[bool, str]:
    """Check if CNN model checkpoint is available.

    Returns:
        Tuple of (available, message).
    """
    if not DEFAULT_CHECKPOINT_PATH.exists():
        return False, f"Checkpoint not found: {DEFAULT_CHECKPOINT_PATH}"

    try:
        # Try to load metadata
        checkpoint = torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            return False, "Invalid checkpoint format"
        if checkpoint.get("class_mapping") != {"0": "LPD", "1": "PD"}:
            return False, "Incompatible class mapping"
        return True, "CNN model ready"
    except Exception as e:
        return False, f"Checkpoint validation failed: {e}"


def get_cnn_metadata() -> Optional[Dict[str, Any]]:
    """Get CNN model metadata if available."""
    if not DEFAULT_CHECKPOINT_PATH.exists():
        return None

    try:
        checkpoint = torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        val_metrics = checkpoint.get("val_metrics", {})
        epoch = checkpoint.get("epoch", 0)
        training_history = checkpoint.get("training_history", {})

        return {
            "model_type": "ResNet-18",
            "pretrained": config.get("pretrained", True),
            "image_size": config.get("image_size", 224),
            "dropout": config.get("dropout", 0.3),
            "freeze_backbone": config.get("freeze_backbone", False),
            "epoch": epoch,
            "val_loss": checkpoint.get("val_loss"),
            "val_metrics": val_metrics,
            "training_epochs": len(training_history.get("train_loss", [])),
            "class_mapping": CLASS_MAPPING,
        }
    except Exception:
        return None