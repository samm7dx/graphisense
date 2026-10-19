"""Interactive demonstration utility for single-image handwriting inference.

Loads an image, performs binarization, feature extraction, and model prediction,
and saves a visual diagnostic report figure.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.image_loader import load_image_with_grayscale
from src.preprocessing.binarization import binarize_handwriting
from src.models.inference import predict_handwriting


DATASET_ROOT = _REPO_ROOT / "data" / "raw" / "handwriting" / "extracted" / "DATASET DYSGRAPHIA HANDWRITING"


def run_visual_demo(
    image_path: Union[str, Path],
    save_fig_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Run end-to-end inference and produce an interpretability visualization figure.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to the handwriting image file.
    save_fig_path : Optional[Union[str, Path]], optional
        Path where the diagnostic figure will be saved.

    Returns
    -------
    dict
        Full inference result dictionary from predict_handwriting.
    """
    img_path = Path(image_path).resolve()
    print(f"\n=======================================================")
    print(f"GraphiSense Single-Image Inference Demonstration")
    print(f"Target Image: {img_path.name}")
    print(f"=======================================================")

    # 1. Run inference
    result = predict_handwriting(img_path)

    # 2. Extract visualization representations
    original_bgr, gray = load_image_with_grayscale(img_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    binary_mask = binarize_handwriting(gray, method="otsu", polarity="dark_background")

    # Medial axis distance transform visualization
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 3. Create diagnostic visual plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"1. Original Image: {img_path.name} ({original_rgb.shape[1]}x{original_rgb.shape[0]} px)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap="gray")
    fg_pct = (binary_mask == 255).mean() * 100
    axes[1].set_title(f"2. Binary Foreground Mask (Otsu Threshold, Foreground: {fg_pct:.2f}%)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    im_dist = axes[2].imshow(dist_map, cmap="inferno")
    fig.colorbar(im_dist, ax=axes[2], fraction=0.03, pad=0.02, label="Inscribed Radius (px)")
    mean_sw = result["features_used"]["stroke_width_mean"]
    axes[2].set_title(f"3. Stroke Distance Transform & Medial Axis (Mean Stroke Thickness: {mean_sw:.2f} px)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    pred_label = result["predicted_label"]
    probs = result["class_probabilities"]
    fig.suptitle(
        f"Prediction: {pred_label} (P(PD) = {probs['PD']:.1%}, P(LPD) = {probs['LPD']:.1%})\n"
        f"Disclaimer: Research Screening Baseline — Not a Medical Diagnosis",
        fontsize=13,
        fontweight="bold",
        color="#8b0000" if pred_label == "PD" else "#005500",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    if save_fig_path:
        out_p = Path(save_fig_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=180)
        print(f"Saved diagnostic visualization to: {out_p}")

    plt.close()

    # 4. Print clean console report
    print(f"\nModel Prediction:")
    print(f"  Predicted Class Label: {result['predicted_label']}")
    print(f"  Estimated Model Probabilities:")
    print(f"    - Potential Dysgraphia (PD):     {probs['PD']:.2%}")
    print(f"    - Low Potential Dysgraphia (LPD): {probs['LPD']:.2%}")

    print(f"\nExtracted Feature Vector (Reduced Static Features):")
    for feat_name, feat_val in result["features_used"].items():
        print(f"  - {feat_name:<30} = {feat_val}")

    print(f"\nNotice & Disclaimer:")
    print(f"  {result['disclaimer']}")
    print(f"=======================================================\n")

    return result


def main() -> None:
    """CLI handler for interactive single-image demonstration."""
    parser = argparse.ArgumentParser(description="GraphiSense Single-Image Inference Demonstration")
    parser.add_argument("image_path", nargs="?", help="Path to input handwriting image")
    args = parser.parse_args()

    fig_dir = _REPO_ROOT / "outputs" / "figures"

    if args.image_path:
        run_visual_demo(args.image_path, save_fig_path=fig_dir / "demo_single_prediction.png")
    else:
        # Run demo on representative samples from both classes
        pd_sample = next((DATASET_ROOT / "Potential Dysgraphia").glob("*.jpg"))
        lpd_sample = next((DATASET_ROOT / "Low Potential Dysgraphia").glob("*.jpg"))

        print("No image specified. Demonstrating on representative dataset samples:")
        print(f"1. Potential Dysgraphia Sample: {pd_sample.name}")
        run_visual_demo(pd_sample, save_fig_path=fig_dir / "demo_inference_PD_sample.png")

        print(f"2. Low Potential Dysgraphia Sample: {lpd_sample.name}")
        run_visual_demo(lpd_sample, save_fig_path=fig_dir / "demo_inference_LPD_sample.png")


if __name__ == "__main__":
    main()
