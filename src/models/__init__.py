"""Machine learning models and evaluation pipelines for GraphiSense."""

from src.models.train_evaluate import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    REDUCED_STATIC_FEATURES,
    build_pipeline,
    evaluate_model_cv,
    run_all_evaluations,
    train_and_save_final_model,
)
from src.models.inference import (
    load_saved_model,
    predict_handwriting,
)

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
