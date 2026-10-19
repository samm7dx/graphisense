
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "data" / "raw" / "raw_image1.jpeg"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the handwriting image
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if image is None:
    raise SystemExit(f"Could not load image: {IMAGE_PATH}")

print("Image loaded successfully!")
print("Image dimensions (height, width, channels):", image.shape)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a binary image using Otsu thresholding
threshold_value, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

# Save the processed images
cv2.imwrite(str(OUTPUT_DIR / "grayscale.png"), gray)
cv2.imwrite(str(OUTPUT_DIR / "threshold.png"), binary)

print("Grayscale image saved.")
print("Threshold image saved.")
print("Otsu threshold value:", threshold_value)
print("Prototype completed successfully!")
