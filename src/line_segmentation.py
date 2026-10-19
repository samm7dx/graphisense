
from pathlib import Path
import csv

import cv2
import numpy as np

# 1. Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "data" / "raw" / "raw_image1.jpeg"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load and rotate image
original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if original is None:
    raise SystemExit(f"ERROR: Cannot load image: {IMAGE_PATH}")

rotated = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)

print("Original dimensions:", original.shape)
print("Rotated dimensions:", rotated.shape)

# 3. Crop the writing area
# Based on the observed positions in the rotated image.
image = rotated[350:1050, 0:1600].copy()

if image.size == 0:
    raise SystemExit("ERROR: Crop is empty. Check crop coordinates.")

# 4. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 5. Threshold: handwriting and notebook rules become white
_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

height, width = binary.shape

# 6. Detect long horizontal notebook rules
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (max(50, width // 3), 1),
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel,
)

# Remove horizontal rules while preserving handwriting
text_mask = binary.copy()

rule_mask = cv2.dilate(
    horizontal_lines,
    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    iterations=1,
)

text_mask[rule_mask > 0] = 0

# 7. Connect nearby handwriting marks
join_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (max(15, width // 80), 3),
)

joined_text = cv2.dilate(
    text_mask,
    join_kernel,
    iterations=1,
)

# 8. Horizontal projection
projection = np.count_nonzero(joined_text, axis=1)

window_size = 9
smoothed = np.convolve(
    projection,
    np.ones(window_size) / window_size,
    mode="same",
)

peak = smoothed.max()

if peak == 0:
    raise SystemExit(
        "ERROR: No foreground pixels detected. "
        "Inspect text_mask.png."
    )

row_threshold = max(2, peak * 0.12)
active_rows = smoothed > row_threshold

# 9. Group active rows into line regions
regions = []
start = None

for y, active in enumerate(active_rows):
    if active and start is None:
        start = y
    elif not active and start is not None:
        end = y - 1
        if end - start + 1 >= 8:
            regions.append((start, end))
        start = None

if start is not None:
    end = height - 1
    if end - start + 1 >= 8:
        regions.append((start, end))

# Merge regions separated by small gaps
merged_regions = []

for y1, y2 in regions:
    if merged_regions and y1 - merged_regions[-1][1] <= 8:
        merged_regions[-1] = (
            merged_regions[-1][0],
            y2,
        )
    else:
        merged_regions.append((y1, y2))
# Remove thin regions such as notebook rules and borders
merged_regions = [
    (y1, y2)
    for y1, y2 in merged_regions
    if y2 - y1 + 1 >= 50
]

# 10. Draw bounding boxes and measure line features
overlay = image.copy()
line_data = []

for line_id, (y1, y2) in enumerate(merged_regions, start=1):
    crop = text_mask[y1:y2 + 1, :]

    columns = np.where(
        np.count_nonzero(crop, axis=0) > 0
    )[0]

    if len(columns) == 0:
        continue

    x1 = int(columns[0])
    x2 = int(columns[-1])

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 200, 0),
        2,
    )

    cv2.putText(
        overlay,
        f"Line {line_id}",
        (max(0, x1), max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    line_width = x2 - x1 + 1
    line_height = y2 - y1 + 1

    line_data.append({
        "line_id": line_id,
        "x": x1,
        "y": y1,
        "width": line_width,
        "height": line_height,
        "aspect_ratio": round(
            line_width / max(1, line_height), 3
        ),
        "ink_pixels": int(np.count_nonzero(crop)),
    })

# 11. Save output images
images = {
    "rotated_image.png": rotated,
    "cropped_writing_area.png": image,
    "horizontal_rules.png": horizontal_lines,
    "text_mask.png": text_mask,
    "projection_input.png": joined_text,
    "line_segmentation.png": overlay,
}

for filename, result in images.items():
    path = OUTPUT_DIR / filename

    if not cv2.imwrite(str(path), result):
        raise SystemExit(f"ERROR: Could not save {path}")

# 12. Save line measurements to CSV
csv_path = OUTPUT_DIR / "line_features.csv"

fields = [
    "line_id",
    "x",
    "y",
    "width",
    "height",
    "aspect_ratio",
    "ink_pixels",
]

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(line_data)

# 13. Report results
print("\nLINE SEGMENTATION COMPLETE")
print("Cropped image dimensions:", image.shape)
print("Detected line regions:", len(line_data))
print("Overlay:", OUTPUT_DIR / "line_segmentation.png")
print("Features:", csv_path)
print("Inspect cropped_writing_area.png and text_mask.png.")
