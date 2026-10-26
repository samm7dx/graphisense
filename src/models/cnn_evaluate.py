"""CNN Evaluation module for GraphiSense.

Held-out evaluation and metrics for trained ResNet-18 model.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cnn_dataset import (
    HandwritingDataset,
    create_data_splits,
    discover_dataset,
    get_transforms,
    load_split_metadata,
)
from src.models.cnn_inference import CNNInference
from src.models.cnn_model import load_checkpoint


DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "models" / "cnn"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "best_model.pt"


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a dataloader.

    Returns:
        Dictionary with comprehensive metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probas = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probas.extend(probabilities.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probas = np.array(all_probas)

    avg_loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0

    # Calculate metrics
    metrics = calculate_detailed_metrics(all_targets, all_preds, all_probas)

    metrics["loss"] = avg_loss
    metrics["num_samples"] = len(all_targets)

    return metrics


def calculate_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    for class_idx, class_name in [(0, "LPD"), (1, "PD")]:
        metrics[f"precision_{class_name.lower()}"] = precision_score(
            y_true, y_pred, pos_label=class_idx, zero_division=0
        )
        metrics[f"recall_{class_name.lower()}"] = recall_score(
            y_true, y_pred, pos_label=class_idx, zero_division=0
        )
        metrics[f"f1_{class_name.lower()}"] = f1_score(
            y_true, y_pred, pos_label=class_idx, zero_division=0
        )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = ["LPD", "PD"]

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=["LPD", "PD"], output_dict=True, zero_division=0
    )

    # ROC-AUC if probabilities provided
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

            # ROC curve data
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except Exception as e:
            metrics["roc_auc"] = 0.0
            metrics["roc_auc_error"] = str(e)

    return metrics


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    dataset_root: Union[str, Path],
    split_metadata_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Evaluate a saved checkpoint on the test set.

    Args:
        checkpoint_path: Path to the model checkpoint.
        dataset_root: Root directory of the dataset.
        split_metadata_path: Path to split metadata (uses default if None).
        batch_size: Batch size for evaluation.
        num_workers: DataLoader workers.
        device: Device to evaluate on.
        image_size: Input image size.

    Returns:
        Dictionary with evaluation results.
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Evaluating on device: {device}")

    # Load checkpoint
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    config = metadata.get("config", {})

    # Discover dataset
    image_paths, labels = discover_dataset(dataset_root)

    # Load or create splits
    if split_metadata_path is None:
        split_metadata_path = Path(checkpoint_path).parent.parent / "split_metadata.json"

    if Path(split_metadata_path).exists():
        print("Loading existing split...")
        split_data = load_split_metadata(split_metadata_path)
        train_indices = split_data["train_indices"]
        val_indices = split_data["val_indices"]
        test_indices = split_data["test_indices"]
    else:
        print("Creating new split (WARNING: may not match training split)...")
        train_indices, val_indices, test_indices = create_data_splits(
            image_paths, labels,
            train_ratio=config.get("train_ratio", 0.7),
            val_ratio=config.get("val_ratio", 0.15),
            test_ratio=config.get("test_ratio", 0.15),
            seed=config.get("seed", 42),
        )

    # Create test dataset
    eval_transform = get_transforms(
        image_size=config.get("image_size", image_size),
        is_training=False,
    )

    test_dataset = HandwritingDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=eval_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test set size: {len(test_dataset)}")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device)

    # Add metadata
    results = {
        "checkpoint": str(checkpoint_path),
        "model_config": config,
        "test_metrics": metrics,
        "test_set_size": len(test_dataset),
        "test_class_distribution": {
            "LPD": int(np.sum(np.array(labels)[test_indices] == 0)),
            "PD": int(np.sum(np.array(labels)[test_indices] == 1)),
        },
    }

    return results


def run_evaluation_cli(
    checkpoint_path: Optional[Union[str, Path]] = None,
    dataset_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[str] = None,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Run evaluation from CLI."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    if dataset_root is None:
        dataset_root = _REPO_ROOT / "data" / "raw"
    if output_dir is None:
        output_dir = _REPO_ROOT / "outputs" / "reports" / "cnn_evaluation"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        image_size=image_size,
    )

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

    print(f"\nEvaluation complete. Results saved to {results_path}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {results['test_metrics']['balanced_accuracy']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics'].get('roc_auc', 0.0):.4f}")

    return results


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GraphiSense CNN checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str, default="data/raw",
                        help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")

    args = parser.parse_args()

    run_evaluation_cli(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()