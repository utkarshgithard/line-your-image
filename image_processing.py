"""
Mathematical Image Analysis module for Line Art Reconstruction.
Handles image loading, grayscale normalization, Gaussian filtering, Sobel gradients,
structure tensor orientation fields, and edge detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path, max_dimension: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image from disk, convert to RGB, and resize to fit within max_dimension
    while strictly preserving the aspect ratio.

    Returns:
        rgb_image: uint8 numpy array with shape (H, W, 3) in RGB order.
        normalized_intensity: float32 numpy array with shape (H, W) in [0.0, 1.0],
            where 0.0 is pure black and 1.0 is pure white.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    # Use Pillow for robust format support (PNG, JPG, JPEG, WEBP, BMP, etc.)
    try:
        pil_img = Image.open(path_obj)
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}: {e}") from e

    w, h = pil_img.size
    if max_dimension > 0 and max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    rgb = np.array(pil_img, dtype=np.uint8)
    # Convert RGB to grayscale normalized in [0, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    normalized_intensity = gray.astype(np.float32) / 255.0

    return rgb, normalized_intensity


def compute_darkness(intensity: np.ndarray) -> np.ndarray:
    """
    Compute darkness field D(x, y) = 1.0 - I(x, y).
    D is clamped in range [0.0, 1.0], where 0.0 is pure white and 1.0 is pure black.
    """
    darkness = 1.0 - intensity
    return np.clip(darkness, 0.0, 1.0).astype(np.float32)


def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    Apply Gaussian smoothing to reduce sensor noise while preserving important edges.
    If sigma <= 0, returns the input image unchanged.
    """
    if sigma <= 0:
        return image.copy()
    # Kernel size chosen according to 6*sigma rule
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def compute_gradients(intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operators:
    Gx = dI / dx
    Gy = dI / dy
    Gradient magnitude: G(x, y) = sqrt(Gx^2 + Gy^2)

    Returns:
        gx: float32 array of horizontal gradient
        gy: float32 array of vertical gradient
        magnitude: float32 array of gradient magnitude, normalized to [0.0, 1.0]
    """
    # cv2.Sobel with CV_32F
    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)

    magnitude = np.sqrt(gx * gx + gy * gy)
    max_val = float(np.max(magnitude))
    if max_val > 1e-6:
        normalized_magnitude = np.clip(magnitude / max_val, 0.0, 1.0)
    else:
        normalized_magnitude = np.zeros_like(magnitude)

    return gx, gy, normalized_magnitude


def compute_structure_tensor(
    intensity: np.ndarray,
    sigma_deriv: float = 1.0,
    sigma_smooth: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D structure tensor (second-moment matrix):
    J = [ [J_xx, J_xy],
          [J_xy, J_yy] ]
    where J_xx = (dI/dx)^2, J_yy = (dI/dy)^2, J_xy = (dI/dx)(dI/dy).
    Smoothed with Gaussian kernel sigma_smooth.

    Returns:
        contour_angle: Angle in radians [-pi/2, pi/2] along the tangent of contours/isophotes.
        coherence: Anisotropy measure in [0.0, 1.0] indicating edge vs flat/isotropic regions.
    """
    # 1. Smooth image slightly before differentiation
    smoothed = apply_gaussian_blur(intensity, sigma_deriv)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)

    # 2. Tensor components
    j_xx = gx * gx
    j_yy = gy * gy
    j_xy = gx * gy

    # 3. Smooth tensor components spatially
    j_xx = apply_gaussian_blur(j_xx, sigma_smooth)
    j_yy = apply_gaussian_blur(j_yy, sigma_smooth)
    j_xy = apply_gaussian_blur(j_xy, sigma_smooth)

    # 4. Principal direction:
    # 2*theta = atan2(2*J_xy, J_xx - J_yy) gives the gradient direction (maximum change).
    # Adding pi/2 (or rotating by 90 degrees) gives the contour tangent direction (along edges).
    grad_angle = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy)
    contour_angle = grad_angle + (np.pi / 2.0)

    # Wrap angles to [-pi/2, pi/2]
    contour_angle = np.mod(contour_angle + (np.pi / 2.0), np.pi) - (np.pi / 2.0)

    # 5. Coherence / Anisotropy: ((lambda1 - lambda2) / (lambda1 + lambda2))^2
    diff = j_xx - j_yy
    trace = j_xx + j_yy
    numerator = np.sqrt(diff * diff + 4.0 * (j_xy * j_xy))
    denominator = trace + 1e-6
    coherence = np.clip(numerator / denominator, 0.0, 1.0)

    return contour_angle.astype(np.float32), coherence.astype(np.float32)


def detect_edges(intensity: np.ndarray, sensitivity: float = 0.4) -> np.ndarray:
    """
    Detect important structural boundaries using Canny edge detector
    with adaptive thresholding scaled by sensitivity.

    Returns:
        edge_map: float32 array in [0.0, 1.0] where 1.0 denotes a detected edge.
    """
    # Convert to uint8 for OpenCV Canny
    uint8_img = np.clip(intensity * 255.0, 0.0, 255.0).astype(np.uint8)

    # Median intensity for Otsu-like robust threshold calculation
    v = np.median(uint8_img)

    # Adjust thresholds based on sensitivity: higher sensitivity -> lower thresholds -> more edges
    # sensitivity 0.4 -> sensible default
    factor = np.clip(1.0 - sensitivity * 0.75, 0.1, 1.5)
    lower = int(max(0, (1.0 - 0.33 * factor) * v))
    upper = int(min(255, (1.0 + 0.33 * factor) * v))
    if lower >= upper:
        lower = max(0, upper - 30)

    edges = cv2.Canny(uint8_img, lower, upper)
    return (edges.astype(np.float32) / 255.0)


def detect_face_and_features(rgb_image: np.ndarray, config=None):
    """
    Detect whether a face exists and locate facial features (eyes, eyebrows, nose, lips, jawline).
    Wraps face_processing.detect_face_features.
    """
    from face_processing import detect_face_features

    return detect_face_features(rgb_image, config)

