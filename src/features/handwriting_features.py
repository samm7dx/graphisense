"""Static handwriting feature extraction module for GraphiSense.

This module extracts interpretable geometric, topological, and stroke distribution
features from handwriting images or binarized masks.

Important Scope & Methodology Notes:
1. Connected Components: Connected components do NOT necessarily correspond to
   individual letters or words. In natural and dysgraphic handwriting, overlapping,
   cursive connections, and touching strokes often merge multiple letters into a
   single connected structure, while dotting and accents form disjoint components.
2. Static vs. Dynamic Limitations: Static 2D scanned/photographed handwriting images
   CANNOT measure true dynamic physical quantities such as actual physical pen tip
   pressure, writing velocity, acceleration, or in-air pauses. All metrics extracted
   here (such as distance-transform stroke thickness and ink density) are purely
   2D morphological and spatial proxies.
"""

from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from src.preprocessing.image_loader import to_grayscale
from src.preprocessing.binarization import binarize_handwriting


def extract_handwriting_features(
    image: np.ndarray,
    is_binary: bool = False,
    min_component_area: int = 5,
    binarize_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[int, float]]:
    """Extract a documented, interpretable baseline set of static handwriting features.

    Parameters
    ----------
    image : np.ndarray
        Input image. If `is_binary=False`, this can be a BGR or grayscale image,
        which will be safely binarized using `binarize_handwriting`.
        If `is_binary=True`, this must already be a 2D binary mask with values in {0, 255}.
    is_binary : bool, default=False
        Whether the input `image` is already a pre-binarized mask (foreground=255, background=0).
    min_component_area : int, default=5
        Minimum pixel area threshold to filter out tiny isolated noise specks
        from connected component analysis. Background (label 0) is always excluded.
    binarize_kwargs : Optional[Dict[str, Any]], optional
        Optional keyword arguments passed to `binarize_handwriting` (e.g. method, polarity).
        Ignored if `is_binary=True`.

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing extracted numerical features:
        - Image geometry: image_width, image_height, aspect_ratio
        - Ink density: ink_density, foreground_pixels
        - Connected components: component_count, component_width_mean, component_width_std,
          component_height_mean, component_height_std, component_area_mean, component_area_std,
          component_aspect_ratio_mean, component_aspect_ratio_std
        - Spatial & row projection: centroid_x, centroid_y, centroid_x_norm, centroid_y_norm,
          row_projection_mean, row_projection_std, row_projection_max, active_row_ratio
        - Stroke width proxy: stroke_width_mean, stroke_width_median, stroke_width_std

    Raises
    ------
    TypeError
        If `image` is not a numpy ndarray.
    ValueError
        If `image` is empty, has invalid dimensions, or invalid binary mask values.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Input image array is empty (size 0)")

    if min_component_area < 0:
        raise ValueError(
            f"min_component_area must be >= 0, got {min_component_area}"
        )

    # Prepare binary mask
    if is_binary:
        if image.ndim != 2:
            raise ValueError(
                f"Binary mask must be 2D, got ndim={image.ndim} with shape {image.shape}"
            )
        unique_vals = np.unique(image)
        if not np.all(np.isin(unique_vals, [0, 255])):
            raise ValueError(
                f"Binary mask must contain only values in {{0, 255}}, got {unique_vals.tolist()}"
            )
        binary_mask = image.astype(np.uint8) if image.dtype != np.uint8 else image
    else:
        gray = to_grayscale(image)
        kwargs = binarize_kwargs or {}
        binary_mask = binarize_handwriting(gray, **kwargs)

    h, w = binary_mask.shape
    total_pixels = h * w
    aspect_ratio = float(w / h) if h > 0 else 0.0

    fg_mask = binary_mask == 255
    foreground_pixel_count = int(np.count_nonzero(fg_mask))
    ink_density = float(foreground_pixel_count / total_pixels) if total_pixels > 0 else 0.0

    # Base feature dictionary
    features: Dict[str, Union[int, float]] = {
        "image_width": int(w),
        "image_height": int(h),
        "aspect_ratio": round(aspect_ratio, 4),
        "ink_density": round(ink_density, 6),
        "foreground_pixels": foreground_pixel_count,
    }

    # Handle empty foreground mask gracefully without crashing
    if foreground_pixel_count == 0:
        features.update(
            {
                "component_count": 0,
                "component_width_mean": 0.0,
                "component_width_std": 0.0,
                "component_height_mean": 0.0,
                "component_height_std": 0.0,
                "component_area_mean": 0.0,
                "component_area_std": 0.0,
                "component_aspect_ratio_mean": 0.0,
                "component_aspect_ratio_std": 0.0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "centroid_x_norm": 0.0,
                "centroid_y_norm": 0.0,
                "row_projection_mean": 0.0,
                "row_projection_std": 0.0,
                "row_projection_max": 0,
                "active_row_ratio": 0.0,
                "stroke_width_mean": 0.0,
                "stroke_width_median": 0.0,
                "stroke_width_std": 0.0,
            }
        )
        return features

    # 1. Connected Component Analysis
    # Connectivity 8 captures continuous handwritten strokes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Exclude background (index 0)
    # stats columns: [CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA]
    fg_stats = stats[1:]
    areas = fg_stats[:, cv2.CC_STAT_AREA]

    # Filter out tiny isolated noise components
    valid_mask = areas >= min_component_area
    valid_stats = fg_stats[valid_mask]

    component_count = int(len(valid_stats))
    features["component_count"] = component_count

    if component_count > 0:
        comp_widths = valid_stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
        comp_heights = valid_stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        comp_areas = valid_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        comp_aspect_ratios = comp_widths / np.maximum(comp_heights, 1.0)

        features["component_width_mean"] = round(float(np.mean(comp_widths)), 4)
        features["component_width_std"] = round(float(np.std(comp_widths)), 4)
        features["component_height_mean"] = round(float(np.mean(comp_heights)), 4)
        features["component_height_std"] = round(float(np.std(comp_heights)), 4)
        features["component_area_mean"] = round(float(np.mean(comp_areas)), 4)
        features["component_area_std"] = round(float(np.std(comp_areas)), 4)
        features["component_aspect_ratio_mean"] = round(
            float(np.mean(comp_aspect_ratios)), 4
        )
        features["component_aspect_ratio_std"] = round(
            float(np.std(comp_aspect_ratios)), 4
        )
    else:
        # All foreground was smaller than noise threshold
        features["component_width_mean"] = 0.0
        features["component_width_std"] = 0.0
        features["component_height_mean"] = 0.0
        features["component_height_std"] = 0.0
        features["component_area_mean"] = 0.0
        features["component_area_std"] = 0.0
        features["component_aspect_ratio_mean"] = 0.0
        features["component_aspect_ratio_std"] = 0.0

    # 2. Horizontal Distribution & Spatial Centroid
    # Coordinates of all foreground pixels
    y_coords, x_coords = np.nonzero(fg_mask)
    centroid_x = float(np.mean(x_coords))
    centroid_y = float(np.mean(y_coords))
    centroid_x_norm = float(centroid_x / w) if w > 0 else 0.0
    centroid_y_norm = float(centroid_y / h) if h > 0 else 0.0

    features["centroid_x"] = round(centroid_x, 4)
    features["centroid_y"] = round(centroid_y, 4)
    features["centroid_x_norm"] = round(centroid_x_norm, 4)
    features["centroid_y_norm"] = round(centroid_y_norm, 4)

    # Row projection (ink distribution along the vertical axis)
    row_proj = np.sum(fg_mask, axis=1).astype(np.float64)
    features["row_projection_mean"] = round(float(np.mean(row_proj)), 4)
    features["row_projection_std"] = round(float(np.std(row_proj)), 4)
    features["row_projection_max"] = int(np.max(row_proj))
    active_rows = np.count_nonzero(row_proj > 0)
    features["active_row_ratio"] = round(float(active_rows / h), 4) if h > 0 else 0.0

    # 3. Stroke Width Proxy via Distance Transform
    # Euclidean distance from each foreground pixel to nearest background (0) pixel.
    # At stroke centerlines (medial axis ridges), dist equals the local radius R,
    # so local stroke width is approximately 2 * R.
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Find medial axis ridges via morphological dilation local-maximum test
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_dist = cv2.dilate(dist_map, kernel)
    ridge_mask = (dist_map == dilated_dist) & (dist_map > 0.0)

    if np.any(ridge_mask):
        local_radii = dist_map[ridge_mask].astype(np.float64)
        stroke_widths = 2.0 * local_radii
        features["stroke_width_mean"] = round(float(np.mean(stroke_widths)), 4)
        features["stroke_width_median"] = round(float(np.median(stroke_widths)), 4)
        features["stroke_width_std"] = round(float(np.std(stroke_widths)), 4)
    else:
        # Fallback if no clean ridges detected
        features["stroke_width_mean"] = 0.0
        features["stroke_width_median"] = 0.0
        features["stroke_width_std"] = 0.0

    return features
