"""Machine learning models and evaluation pipelines for GraphiSense."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "FEATURE_SET_A": "train_evaluate",
    "FEATURE_SET_B": "train_evaluate",
    "REDUCED_STATIC_FEATURES": "train_evaluate",
    "build_pipeline": "train_evaluate",
    "evaluate_model_cv": "train_evaluate",
    "run_all_evaluations": "train_evaluate",
    "train_and_save_final_model": "train_evaluate",
    "load_saved_model": "inference",
    "predict_handwriting": "inference",
}

__all__ = [
    "FEATURE_SET_A",
    "FEATURE_SET_B",
    "REDUCED_STATIC_FEATURES",
    "build_pipeline",
    "evaluate_model_cv",
    "run_all_evaluations",
    "train_and_save_final_model",
    "load_saved_model",
    "predict_handwriting",
]


def __getattr__(name: str) -> Any:
    """Import public API members lazily to keep module CLI execution clean."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
