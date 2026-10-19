
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "data" / "raw" / "raw_image1.jpeg"
MASK_PATH = PROJECT_DIR / "outputs" / "threshold.png"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original image and binary foreground mask
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise SystemExit(f"Could not load original image: {IMAGE_PATH}")

if mask is None:
    raise SystemExit(f"Could not load threshold mask: {MASK_PATH}")

if image.shape[:2] != mask.shape[:2]:
    raise SystemExit("Image and threshold mask dimensions do not match.")

# Ensure foreground is white (255) and background is black (0)
_, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# --------------------------------------------------
# 1. Connected component analysis
# --------------------------------------------------
number_of_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(binary, connectivity=8)
)

component_rows = []

# Label 0 is the background, so skip it
for label_id in range(1, number_of_labels):
    x = int(stats[label_id, cv2.CC_STAT_LEFT])
    y = int(stats[label_id, cv2.CC_STAT_TOP])
    width = int(stats[label_id, cv2.CC_STAT_WIDTH])
    height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
    area = int(stats[label_id, cv2.CC_STAT_AREA])

    component_rows.append({
        "component_id": label_id,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "area_pixels": area,
        "centroid_x": round(float(centroids[label_id][0]), 2),
        "centroid_y": round(float(centroids[label_id][1]), 2),
        "aspect_ratio": round(width / height, 4) if height else 0,
        "is_small_component": area < 10,
    })

components_df = pd.DataFrame(
    component_rows,
    columns=[
        "component_id", "x", "y", "width", "height",
        "area_pixels", "centroid_x", "centroid_y",
        "aspect_ratio", "is_small_component"
    ],
)

components_path = OUTPUT_DIR / "connected_components.csv"
components_df.to_csv(components_path, index=False)

# --------------------------------------------------
# 2. Contour analysis
# --------------------------------------------------
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

contour_rows = []
overlay = image.copy()

for contour_id, contour in enumerate(contours, start=1):
    x, y, width, height = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    contour_rows.append({
        "contour_id": contour_id,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "contour_area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "aspect_ratio": round(width / height, 4) if height else 0,
        "is_small_contour": area < 10,
    })

    # Draw detected boundaries on the original image
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)

contours_df = pd.DataFrame(
    contour_rows,
    columns=[
        "contour_id", "x", "y", "width", "height",
        "contour_area", "perimeter", "aspect_ratio",
        "is_small_contour"
    ],
)

contours_path = OUTPUT_DIR / "contours.csv"
contours_df.to_csv(contours_path, index=False)

overlay_path = OUTPUT_DIR / "contours_overlay.png"
if not cv2.imwrite(str(overlay_path), overlay):
    raise SystemExit("Could not save contour visualization.")

# --------------------------------------------------
# 3. Report results
# --------------------------------------------------
foreground_pixels = int(cv2.countNonZero(binary))

print("\n=== Connected Component Analysis ===")
print("Components excluding background:", len(component_rows))
print("Components with area < 10 pixels:",
      int(components_df["is_small_component"].sum()))
print("CSV saved:", components_path)

print("\n=== Contour Analysis ===")
print("External contours detected:", len(contours))
print("Contours with area < 10 pixels:",
      int(contours_df["is_small_contour"].sum()))
print("CSV saved:", contours_path)

print("\n=== Foreground Mask ===")
print("White foreground pixels:", foreground_pixels)
print("Contour overlay saved:", overlay_path)
print("\nFeature extraction completed successfully!")
