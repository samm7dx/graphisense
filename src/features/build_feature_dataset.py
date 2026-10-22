"""Script and reusable builder to extract features for the handwriting dataset.

Scans the dataset directories, extracts static handwriting features, and outputs
a tabular dataset to `outputs/features/features.csv`.
"""

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is on sys.path when script is executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.preprocessing.image_loader import load_image
from src.features.handwriting_features import extract_handwriting_features


CLASS_MAPPING = {
    "Potential Dysgraphia": "PD",
    "Low Potential Dysgraphia": "LPD",
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_feature_table(
    dataset_dir: Union[str, Path] = "data/raw",
    output_csv: Union[str, Path] = "outputs/features/features.csv",
    repo_root: Optional[Union[str, Path]] = None,
    min_component_area: int = 5,
) -> List[Dict[str, Any]]:
    """Recursively process handwriting image folders and save extracted features to CSV.

    Parameters
    ----------
    dataset_dir : Union[str, Path]
        Path to the root dataset folder containing class subdirectories.
    output_csv : Union[str, Path]
        Destination filepath for the resulting CSV file.
    repo_root : Optional[Union[str, Path]], optional
        Root path of the repository for computing relative image paths.
        Defaults to the current working directory.
    min_component_area : int, default=5
        Threshold for filtering noise components in connected component analysis.

    Returns
    -------
    List[Dict[str, Any]]
        List of row dictionaries written to CSV.
    """
    ds_path = Path(dataset_dir).resolve()
    out_path = Path(output_csv).resolve()
    root_path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()

    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset root directory not found: {ds_path}. "
            "Download and extract the dataset as described in data/raw/README.md."
        )
    if not ds_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {ds_path}")

    missing_folders = [
        folder_name
        for folder_name in CLASS_MAPPING
        if not (ds_path / folder_name).is_dir()
    ]
    if missing_folders:
        raise FileNotFoundError(
            f"Dataset class folders missing under {ds_path}: {missing_folders}. "
            "Place both 'Potential Dysgraphia' and 'Low Potential Dysgraphia' "
            "folders directly in the dataset root."
        )

    # Ensure parent output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # Iterate through class folders deterministically
    for folder_name, label in CLASS_MAPPING.items():
        folder_path = ds_path / folder_name
        image_files = sorted(
            [
                f
                for f in folder_path.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                if f.is_file()
            ],
            key=lambda p: p.name,
        )
        if not image_files:
            raise FileNotFoundError(
                f"No supported handwriting images found in {folder_path}. "
                f"Expected one of: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        for img_path in image_files:
            rel_path = img_path.relative_to(root_path).as_posix()
            row: Dict[str, Any] = {
                "filename": img_path.name,
                "class_label": label,
                "image_path": rel_path,
                "status": "SUCCESS",
            }

            try:
                # Load as grayscale
                img = load_image(img_path, as_gray=True)
                # Extract features with dark background polarity (dataset verified)
                feats = extract_handwriting_features(
                    img,
                    is_binary=False,
                    min_component_area=min_component_area,
                    binarize_kwargs={"method": "otsu", "polarity": "dark_background"},
                )
                row.update(feats)
            except Exception as exc:
                row["status"] = f"FAILED: {type(exc).__name__}: {str(exc)}"

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No images found to process in {ds_path}")

    # Determine fieldnames deterministically: metadata first, then sorted feature names
    first_success = next((r for r in rows if r["status"] == "SUCCESS"), rows[0])
    feature_keys = [
        k for k in first_success.keys() if k not in ["filename", "class_label", "image_path", "status"]
    ]
    fieldnames = ["filename", "class_label", "image_path", "status"] + feature_keys

    # Write to CSV
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    """CLI entrypoint for dataset feature table construction."""
    print("Building GraphiSense handwriting feature dataset...")
    rows = build_feature_table()
    success_count = sum(1 for r in rows if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in rows if r["status"] != "SUCCESS")
    print(f"Extraction complete: {len(rows)} images processed.")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failed_count}")
    print(f"  Output:     outputs/features/features.csv")


if __name__ == "__main__":
    main()
