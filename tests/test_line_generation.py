"""
Tests for line generation, adaptive sampling, spacing, orientation, and crosshatching.
"""

import math
import numpy as np
import pytest

from config import LineArtConfig, LineSegment
from line_generation import (
    compute_target_density,
    compute_target_spacing,
    create_straight_segment,
    generate_line_art,
    get_active_families,
)


def test_target_density_formula():
    darkness = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    rho_min = 0.01
    rho_max = 0.09
    gamma = 2.0

    rho = compute_target_density(darkness, rho_min=rho_min, rho_max=rho_max, gamma=gamma)

    assert rho[0] == pytest.approx(rho_min)
    assert rho[2] == pytest.approx(rho_max)
    # At D=0.5, gamma=2: 0.01 + 0.25 * (0.09 - 0.01) = 0.01 + 0.02 = 0.03
    assert rho[1] == pytest.approx(0.03)


def test_safe_spacing_calculation():
    # Test zero and near-zero density handling
    rho = np.array([0.0, 0.05, 0.1], dtype=np.float32)
    spacing = compute_target_spacing(rho, max_spacing=500.0)

    # Zero density should yield max_spacing safely without division-by-zero error
    assert spacing[0] == 500.0
    assert spacing[1] == pytest.approx(1.0 / 0.05)
    assert spacing[2] == pytest.approx(1.0 / 0.1)


def test_active_families_by_darkness():
    # Highlights -> 0 families
    assert get_active_families(0.05, max_families=4) == 0
    # Light midtone -> 1 family
    assert get_active_families(0.25, max_families=4) == 1
    # Midtone -> 2 families
    assert get_active_families(0.45, max_families=4) == 2
    # Shadow -> 3 families
    assert get_active_families(0.70, max_families=4) == 3
    # Deep shadow -> 4 families
    assert get_active_families(0.95, max_families=4) == 4

    # Respect max_families limit
    assert get_active_families(0.95, max_families=2) == 2


def test_create_straight_segment_clipping():
    width, height = 100, 100
    # Segment near edge
    seg = create_straight_segment(
        x0=5.0, y0=50.0, angle_rad=0.0, length=20.0, width_limit=width, height_limit=height
    )
    assert seg is not None
    # x1 should be clamped to 0
    assert seg.x1 >= 0.0
    assert seg.x2 <= 100.0
    assert seg.y1 == 50.0
    assert seg.y2 == 50.0


def test_random_seed_reproducibility():
    h, w = 60, 60
    darkness = np.full((h, w), 0.6, dtype=np.float32)
    contour = np.zeros((h, w), dtype=np.float32)
    coherence = np.zeros((h, w), dtype=np.float32)
    edge_map = np.zeros((h, w), dtype=np.float32)
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)

    config1 = LineArtConfig(random_seed=42, max_dimension=60, optimization_iterations=0)
    lines1 = generate_line_art(darkness, contour, coherence, edge_map, gx, gy, config1)

    config2 = LineArtConfig(random_seed=42, max_dimension=60, optimization_iterations=0)
    lines2 = generate_line_art(darkness, contour, coherence, edge_map, gx, gy, config2)

    assert len(lines1) == len(lines2)
    for l1, l2 in zip(lines1, lines2):
        assert l1.as_tuple == pytest.approx(l2.as_tuple)

    # Different seed should produce different lines
    config3 = LineArtConfig(random_seed=999, max_dimension=60, optimization_iterations=0)
    lines3 = generate_line_art(darkness, contour, coherence, edge_map, gx, gy, config3)
    assert any(
        l1.as_tuple != pytest.approx(l3.as_tuple)
        for l1, l3 in zip(lines1[:10], lines3[:10])
    )
