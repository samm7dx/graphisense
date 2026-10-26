"""CNN Model module for GraphiSense.

ResNet-18 transfer learning implementation for binary handwriting classification.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class ResNet18Binary(nn.Module):
    """ResNet-18 with binary classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize the ResNet-18 binary classifier.

        Args:
            num_classes: Number of output classes (default 2 for PD/LPD).
            pretrained: Whether to use ImageNet pretrained weights.
            freeze_backbone: Whether to freeze the backbone parameters.
            dropout: Dropout rate before the final layer.
        """
        super().__init__()

        # Load ResNet-18
        if pretrained:
            # Try to use pretrained weights
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
                self.pretrained = True
            except Exception:
                # Fallback: load without weights if pretrained not available
                self.backbone = models.resnet18(weights=None)
                self.pretrained = False
        else:
            self.backbone = models.resnet18(weights=None)
            self.pretrained = False

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features

        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the final layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
) -> ResNet18Binary:
    """Factory function to create a ResNet18Binary model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Whether to freeze the backbone.
        dropout: Dropout rate.
        device: Device to place the model on.

    Returns:
        Initialized ResNet18Binary model.
    """
    model = ResNet18Binary(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    num_classes: int = 2,
) -> Tuple[ResNet18Binary, Dict[str, Any]]:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.
        num_classes: Number of classes (must match checkpoint).

    Returns:
        Tuple of (model, checkpoint_metadata).

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        ValueError: If checkpoint is invalid or incompatible.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Validate checkpoint structure
    required_keys = {"model_state_dict", "class_mapping", "config"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")

    # Verify class mapping
    class_mapping = checkpoint["class_mapping"]
    if class_mapping != {"0": "LPD", "1": "PD"}:
        raise ValueError(f"Incompatible class mapping: {class_mapping}")

    # Get config
    config = checkpoint.get("config", {})
    dropout = config.get("dropout", 0.0)
    freeze_backbone = config.get("freeze_backbone", False)

    # Create model with matching architecture
    model = create_model(
        num_classes=num_classes,
        pretrained=False,  # We'll load weights from checkpoint
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        device=device,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def save_checkpoint(
    model: ResNet18Binary,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    class_mapping: Dict[str, str],
    config: Dict[str, Any],
    output_path: Union[str, Path],
    training_history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Save a model checkpoint with metadata.

    Args:
        model: The model to save.
        optimizer: The optimizer (for resuming training).
        epoch: Current epoch number.
        train_loss: Training loss.
        val_loss: Validation loss.
        val_metrics: Validation metrics dictionary.
        class_mapping: Class index to label mapping.
        config: Training configuration.
        output_path: Where to save the checkpoint.
        training_history: Optional training history.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "class_mapping": class_mapping,
        "config": config,
        "training_history": training_history or {},
    }

    torch.save(checkpoint, path)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_model_summary(model: ResNet18Binary) -> Dict[str, Any]:
    """Get a summary of the model architecture."""
    params = count_parameters(model)
    return {
        "architecture": "ResNet-18",
        "pretrained": model.pretrained,
        "num_classes": model.backbone.fc.out_features if not isinstance(model.backbone.fc, nn.Sequential) else model.backbone.fc[-1].out_features,
        "parameters": params,
        "backbone_frozen": not any(p.requires_grad for p in model.backbone.parameters() if p is not list(model.backbone.fc.parameters())[0]),
    }