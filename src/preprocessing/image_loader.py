"""Image loading and color conversion module for GraphiSense."""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Safely convert an input image to a 2D grayscale uint8 array.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array (2D grayscale, 3-channel BGR, or 4-channel BGRA).

    Returns
    -------
    np.ndarray
        2D single-channel uint8 grayscale image.

    Raises
    ------
    TypeError
        If input is not a numpy ndarray.
    ValueError
        If the input image is empty or has unsupported dimensions/channels.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Input image array is empty (size 0)")

    if image.ndim == 2:
        # Already 2D grayscale
        return image.copy()

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy()
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(
            f"Unsupported channel count: {channels}. Expected 1, 3, or 4 channels."
        )

    raise ValueError(
        f"Unsupported image dimension: {image.ndim}. Expected 2D or 3D array."
    )


def load_image(path: Union[str, Path], as_gray: bool = False) -> np.ndarray:
    """Load an image from disk using OpenCV with comprehensive error checking.

    Parameters
    ----------
    path : Union[str, Path]
        Filesystem path to the image file.
    as_gray : bool, default=False
        If True, returns a single-channel 2D grayscale image.
        If False, returns the image in original BGR color format.

    Returns
    -------
    np.ndarray
        Loaded image array (uint8).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If the path points to a directory instead of a file.
    IOError
        If OpenCV fails to decode the image file.
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    # Use cv2.imread; if it fails (e.g. non-ASCII paths on Windows), fallback to imdecode
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        try:
            image_data = np.fromfile(str(file_path), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception:
            image = None

    if image is None:
        raise IOError(f"OpenCV failed to decode image: {file_path}")

    if as_gray:
        return to_grayscale(image)

    return image


def load_image_with_grayscale(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image from disk and return both the original and grayscale versions.

    Parameters
    ----------
    path : Union[str, Path]
        Filesystem path to the image file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (original_bgr_image, grayscale_2d_image)
    """
    original = load_image(path, as_gray=False)
    gray = to_grayscale(original)
    return original, gray
