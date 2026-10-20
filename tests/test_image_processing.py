"""
Tests for mathematical image processing, grayscale normalization, and feature extraction.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from image_processing import (
    apply_gaussian_blur,
    compute_darkness,
    compute_gradients,
    compute_structure_tensor,
    detect_edges,
    load_image,
)


def test_invalid_path_raises():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_path_12345.png")


def test_image_loading_and_aspect_ratio_preservation(tmp_path):
    # Create test rectangular image (400 x 200)
    test_file = tmp_path / "test_rect.png"
    img = Image.new("RGB", (400, 200), color=(128, 64, 32))
    img.save(test_file)

    # Max dimension = 200 -> should resize to (200, 100)
    rgb, norm_gray = load_image(test_file, max_dimension=200)
    h, w = norm_gray.shape
    assert w == 200
    assert h == 100
    assert norm_gray.min() >= 0.0
    assert norm_gray.max() <= 1.0


def test_grayscale_normalization_and_darkness():
    # Intensity array from pure black (0.0) to pure white (1.0)
    intensity = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)
    darkness = compute_darkness(intensity)

    assert darkness.shape == intensity.shape
    assert darkness[0, 0] == pytest.approx(1.0)  # Black pixel -> 1.0 darkness
    assert darkness[1, 1] == pytest.approx(0.0)  # White pixel -> 0.0 darkness
    assert darkness[0, 1] == pytest.approx(0.75)
    assert darkness[1, 0] == pytest.approx(0.25)


def test_sobel_gradients():
    # Vertical edge: left is black (0), right is white (1)
    intensity = np.zeros((20, 20), dtype=np.float32)
    intensity[:, 10:] = 1.0

    gx, gy, mag = compute_gradients(intensity)
    assert gx.shape == (20, 20)
    assert gy.shape == (20, 20)
    assert mag.shape == (20, 20)

    # Gradient should be horizontal (gx non-zero near column 10, gy ~ 0)
    assert np.max(np.abs(gx[:, 9:11])) > 0.0
    assert np.max(np.abs(gy)) < 1e-3
    assert mag.max() <= 1.0
    assert mag.min() >= 0.0


def test_structure_tensor():
    # Horizontal lines -> gradient is vertical, contour orientation should be horizontal (~0 rad)
    intensity = np.zeros((30, 30), dtype=np.float32)
    for r in range(0, 30, 4):
        intensity[r : r + 2, :] = 1.0

    contour_angle, coherence = compute_structure_tensor(intensity)
    assert contour_angle.shape == (30, 30)
    assert coherence.shape == (30, 30)
    assert coherence.min() >= 0.0
    assert coherence.max() <= 1.0


def test_detect_edges():
    # Sharp square in center
    img = np.ones((50, 50), dtype=np.float32)
    img[15:35, 15:35] = 0.0

    edges = detect_edges(img, sensitivity=0.5)
    assert edges.shape == (50, 50)
    assert edges.max() == 1.0
    assert edges.min() == 0.0
    # Center and outer boundary should have 0 edges
    assert edges[25, 25] == 0.0
    assert edges[0, 0] == 0.0
    # The perimeter around 15 and 35 should have detected edge pixels
    assert np.sum(edges[14:16, 15:35]) > 0
