"""Preprocessing pipeline stages service for GraphiSense."""

from typing import List, Dict, Any
import numpy as np
import cv2


def get_preprocessing_stages() -> List[Dict[str, Any]]:
    """Define the preprocessing pipeline stages (6 stages, no Medial Axis Ridges)."""
    return [
        {
            "id": 1,
            "name": "Original Image",
            "key": "original_rgb",
            "description": "Input image as loaded (RGB). No processing applied.",
            "available": True,
        },
        {
            "id": 2,
            "name": "Grayscale",
            "key": "grayscale",
            "description": "Single-channel luminance. OpenCV BGR → GRAY conversion (ITU-R BT.601 weights).",
            "available": True,
        },
        {
            "id": 3,
            "name": "Binary Mask (Otsu)",
            "key": "binary_mask",
            "description": "Foreground/background separation via Otsu thresholding with polarity detection. Handwriting=255, Background=0.",
            "available": True,
        },
        {
            "id": 4,
            "name": "Distance Transform",
            "key": "distance_transform",
            "description": "Euclidean distance from each foreground pixel to nearest background. Brighter = further from edge (stroke center).",
            "available": True,
        },
        {
            "id": 5,
            "name": "Connected Components",
            "key": "connected_components",
            "description": "8-connected component analysis with noise filtering (min area=5px²). Bounding boxes shown in color overlay.",
            "available": True,
        },
        {
            "id": 6,
            "name": "Row Projection",
            "key": "row_projection",
            "description": "Horizontal ink distribution (sum of foreground pixels per row). Reveals writing line structure and spacing.",
            "available": True,
        },
    ]


def compute_preprocessing_stages(
    gray: np.ndarray,
    binary_mask: np.ndarray,
    polarity: str,
) -> Dict[str, np.ndarray]:
    """Compute all intermediate preprocessing stage images for UI display."""
    stages = {}

    # Stage 4: Distance Transform
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_INFERNO)
    stages["distance_transform"] = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB)

    # Stage 5: Connected Components with bounding boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        cc_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        # Filter by min area
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = areas >= 5
        for i, (stat, centroid) in enumerate(zip(stats[1:], centroids[1:])):
            if not valid[i]:
                continue
            x, y, w, h, area = stat
            cx, cy = centroid
            # Draw bounding box
            cv2.rectangle(cc_vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Draw centroid
            cv2.circle(cc_vis, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        stages["connected_components"] = cc_vis
    else:
        stages["connected_components"] = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

    # Stage 6: Row Projection visualization
    row_proj = np.sum(binary_mask == 255, axis=1).astype(np.float64)
    if len(row_proj) > 0:
        max_proj = max(row_proj.max(), 1)
        proj_vis = np.zeros((binary_mask.shape[0], 200, 3), dtype=np.uint8)
        for i, val in enumerate(row_proj):
            if val > 0:
                width = int((val / max_proj) * 180)
                cv2.line(proj_vis, (10, i), (10 + width, i), (0, 255, 0), 1)
        stages["row_projection"] = proj_vis
    else:
        stages["row_projection"] = np.zeros((100, 200, 3), dtype=np.uint8)

    return stages