"""CNN Training module for GraphiSense.

Training, validation, and checkpointing for ResNet-18 transfer learning.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cnn_dataset import (
    HandwritingDataset,
    create_data_splits,
    create_dataloaders,
    discover_dataset,
    get_class_distribution,
    get_transforms,
    save_split_metadata,
)
from src.models.cnn_model import (
    ResNet18Binary,
    count_parameters,
    create_model,
    get_model_summary,
    save_checkpoint,
)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    """Validate the model.

    Returns:
        Tuple of (average_loss, accuracy, metrics_dict, all_predictions, all_targets).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = calculate_metrics(all_targets, all_preds)

    return avg_loss, accuracy, metrics, all_preds, all_targets


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Per-class metrics
    metrics["precision_pd"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall_pd"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["f1_pd"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    metrics["precision_lpd"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["recall_lpd"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["f1_lpd"] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    # ROC-AUC if probabilities provided
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def predict_proba(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predicted probabilities for a dataloader.

    Returns:
        Tuple of (probabilities, targets).
    """
    model.eval()
    all_probas = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)
            all_probas.extend(probas.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_probas), np.array(all_targets)


def train_model(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path],
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
    patience: int = 10,
    min_delta: float = 1e-4,
    gradient_clip: Optional[float] = 1.0,
    num_workers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Train a ResNet-18 model on the handwriting dataset.

    Args:
        dataset_root: Root directory of the dataset.
        output_dir: Directory to save checkpoints and logs.
        image_size: Input image size.
        batch_size: Training batch size.
        epochs: Maximum number of epochs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        train_ratio: Training set fraction.
        val_ratio: Validation set fraction.
        test_ratio: Test set fraction.
        seed: Random seed.
        pretrained: Use ImageNet pretrained weights.
        freeze_backbone: Freeze backbone parameters.
        dropout: Dropout rate.
        patience: Early stopping patience.
        min_delta: Minimum improvement for early stopping.
        gradient_clip: Gradient clipping value.
        num_workers: DataLoader workers.
        device: Device to train on.

    Returns:
        Dictionary with training results and paths.
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Discover dataset
    print("Discovering dataset...")
    image_paths, labels = discover_dataset(dataset_root)
    print(f"Found {len(image_paths)} images")
    print(f"Class distribution: {get_class_distribution(labels)}")

    # Create transforms
    train_transform = get_transforms(image_size=image_size, is_training=True)
    eval_transform = get_transforms(image_size=image_size, is_training=False)

    # Create datasets
    full_dataset = HandwritingDataset(image_paths, labels, transform=None)

    # Create splits
    print("Creating data splits...")
    train_indices, val_indices, test_indices = create_data_splits(
        image_paths,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Create datasets with appropriate transforms
    train_dataset = HandwritingDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=train_transform,
    )
    val_dataset = HandwritingDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=eval_transform,
    )
    test_dataset = HandwritingDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=eval_transform,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # Save split metadata
    split_config = {
        "image_size": image_size,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "dropout": dropout,
    }
    save_split_metadata(
        train_indices, val_indices, test_indices,
        image_paths, labels,
        output_dir / "split_metadata.json",
        split_config,
    )

    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        device=device,
    )

    summary = get_model_summary(model)
    print(f"Model: {summary['architecture']}")
    print(f"Parameters: {summary['parameters']['total']:,} total, {summary['parameters']['trainable']:,} trainable")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    training_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        # Validate
        val_loss, val_acc, val_metrics, _, _ = validate(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)
        training_history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                class_mapping={"0": "LPD", "1": "PD"},
                config={
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "pretrained": pretrained,
                    "freeze_backbone": freeze_backbone,
                    "dropout": dropout,
                    "seed": seed,
                },
                output_path=checkpoints_dir / "best_model.pt",
                training_history=training_history,
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (best at epoch {best_epoch})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metrics=val_metrics,
        class_mapping={"0": "LPD", "1": "PD"},
        config=split_config,
        output_path=checkpoints_dir / "final_model.pt",
        training_history=training_history,
    )

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_metrics, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )

    # Get test probabilities for ROC-AUC
    test_probas, _ = predict_proba(model, test_loader, device)
    test_metrics["roc_auc"] = test_metrics.get("roc_auc", 0.0)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test F1 (PD): {test_metrics['f1_pd']:.4f}")
    print(f"Test F1 (LPD): {test_metrics['f1_lpd']:.4f}")

    # Save evaluation results
    eval_results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_metrics": test_metrics,
        "test_predictions": test_preds.tolist(),
        "test_targets": test_targets.tolist(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "total_time_seconds": total_time,
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(eval_results), f, indent=2)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "checkpoints_dir": str(checkpoints_dir),
        "output_dir": str(output_dir),
        "eval_results": eval_results,
    }


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet-18 for GraphiSense")
    parser.add_argument("--dataset-root", type=str, default="data/raw",
                        help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="outputs/models/cnn",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="Don't use pretrained weights")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze backbone")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    results = train_model(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(f"\nTraining complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()