"""Machine learning models and evaluation pipelines for GraphiSense."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    # Random Forest modules
    "FEATURE_SET_A": "train_evaluate",
    "FEATURE_SET_B": "train_evaluate",
    "REDUCED_STATIC_FEATURES": "train_evaluate",
    "build_pipeline": "train_evaluate",
    "evaluate_model_cv": "train_evaluate",
    "run_all_evaluations": "train_evaluate",
    "train_and_save_final_model": "train_evaluate",
    "load_saved_model": "inference",
    "predict_handwriting": "inference",
    # CNN/ResNet modules
    "HandwritingDataset": "cnn_dataset",
    "discover_dataset": "cnn_dataset",
    "get_transforms": "cnn_dataset",
    "create_data_splits": "cnn_dataset",
    "create_dataloaders": "cnn_dataset",
    "ResNet18Binary": "cnn_model",
    "create_model": "cnn_model",
    "load_checkpoint": "cnn_model",
    "save_checkpoint": "cnn_model",
    "train_model": "cnn_train",
    "CNNInference": "cnn_inference",
    "predict_handwriting_cnn": "cnn_inference",
    "check_cnn_availability": "cnn_inference",
    "get_cnn_metadata": "cnn_inference",
    "evaluate_checkpoint": "cnn_evaluate",
    "run_evaluation_cli": "cnn_evaluate",
}

__all__ = [
    # Random Forest
    "FEATURE_SET_A",
    "FEATURE_SET_B",
    "REDUCED_STATIC_FEATURES",
    "build_pipeline",
    "evaluate_model_cv",
    "run_all_evaluations",
    "train_and_save_final_model",
    "load_saved_model",
    "predict_handwriting",
    # CNN/ResNet
    "HandwritingDataset",
    "discover_dataset",
    "get_transforms",
    "create_data_splits",
    "create_dataloaders",
    "ResNet18Binary",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
    "train_model",
    "CNNInference",
    "predict_handwriting_cnn",
    "check_cnn_availability",
    "get_cnn_metadata",
    "evaluate_checkpoint",
    "run_evaluation_cli",
]


def __getattr__(name: str) -> Any:
    """Import public API members lazily to keep module CLI execution clean."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
