"""
Tests for reconstruction loss calculation and iterative refinement.
"""

import numpy as np
import pytest

from config import LineArtConfig, LineSegment
from optimization import compute_reconstruction_loss, refine_line_arrangements
from renderer import render_lines


def test_reconstruction_loss_components():
    h, w = 40, 40
    target_intensity = np.full((h, w), 0.5, dtype=np.float32)
    edge_map = np.zeros((h, w), dtype=np.float32)

    config = LineArtConfig()
    lines = [LineSegment(10, 10, 30, 30)]

    _, rendered_intensity = render_lines(lines, w, h, config)
    total_loss, components = compute_reconstruction_loss(
        rendered_intensity, target_intensity, edge_map, lines, config
    )

    assert total_loss > 0.0
    assert "intensity_loss" in components
    assert "edge_loss" in components
    assert "structural_loss" in components
    assert "regularization_loss" in components
    assert components["intensity_loss"] >= 0.0
    assert 0.0 <= components["ssim"] <= 1.0


def test_iterative_refinement_monotonically_tracks_loss():
    h, w = 50, 50
    # Gradient target
    target_intensity = np.linspace(0.1, 0.9, w, dtype=np.float32)
    target_intensity = np.tile(target_intensity, (h, 1))

    darkness = 1.0 - target_intensity
    edge_map = np.zeros((h, w), dtype=np.float32)
    contour_angle = np.zeros((h, w), dtype=np.float32)

    config = LineArtConfig(
        max_dimension=50,
        optimization_iterations=2,
        random_seed=42,
    )

    initial_lines = [
        LineSegment(x1=5, y1=10, x2=20, y2=10),
        LineSegment(x1=5, y1=30, x2=20, y2=30),
    ]

    refined_lines, history = refine_line_arrangements(
        initial_lines=initial_lines,
        target_intensity=target_intensity,
        darkness=darkness,
        edge_map=edge_map,
        contour_angle=contour_angle,
        config=config,
    )

    assert len(history) == config.optimization_iterations + 1
    # Check that best loss does not increase
    losses = [step["total_loss"] for step in history]
    assert losses[-1] <= losses[0]
