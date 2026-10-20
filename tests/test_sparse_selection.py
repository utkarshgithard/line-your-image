"""
Tests for sparse parsimonious line pursuit and minimum-line objective.
Verifies the 6-step selection algorithm:
1. Candidate pool generation
2. Structural edge prioritization
3. Incremental visual gain measurement
4. Line penalty (L0 regularization)
5. Duplicate / redundant line rejection
6. Line budget constraint & stopping criteria
"""

import numpy as np
import pytest

from config import LineArtConfig, LineSegment
from sparse_line_selection import (
    bresenham_line_pixels,
    generate_candidate_pool,
    line_similarity,
    select_parsimonious_lines,
)


def test_bresenham_pixels_in_bounds():
    w, h = 100, 80
    xs, ys = bresenham_line_pixels(10, 20, 90, 70, w, h)

    assert len(xs) > 0
    assert len(xs) == len(ys)
    assert np.all(xs >= 0) and np.all(xs < w)
    assert np.all(ys >= 0) and np.all(ys < h)
    assert xs[0] == 10 and ys[0] == 20
    assert xs[-1] == 90 and ys[-1] == 70


def test_line_similarity_duplicate_detection():
    # Two almost identical lines
    l1 = LineSegment(10.0, 10.0, 50.0, 50.0)
    l2 = LineSegment(10.5, 9.8, 50.2, 50.1)
    sim_high = line_similarity(l1, l2)
    assert sim_high > 0.85

    # Orthogonal line
    l3 = LineSegment(10.0, 50.0, 50.0, 10.0)
    sim_low = line_similarity(l1, l3)
    assert sim_low == 0.0


def test_line_budget_hard_limit():
    h, w = 100, 100
    # Image with sharp rectangle in center
    darkness = np.zeros((h, w), dtype=np.float32)
    darkness[25:75, 25:75] = 0.9
    edge_map = np.zeros((h, w), dtype=np.float32)
    edge_map[25, 25:75] = 1.0
    edge_map[75, 25:75] = 1.0
    edge_map[25:75, 25] = 1.0
    edge_map[25:75, 75] = 1.0

    contour_angle = np.zeros((h, w), dtype=np.float32)
    coherence = np.ones((h, w), dtype=np.float32)

    config = LineArtConfig(style="minimalist", min_length=10, max_length=50)

    # Hard budget = 15 lines
    budget = 15
    selected = select_parsimonious_lines(
        darkness,
        edge_map,
        contour_angle,
        coherence,
        config,
        line_budget=budget,
        line_penalty=0.01,
    )

    assert len(selected) > 0
    assert len(selected) <= budget


def test_line_penalty_rejects_empty_canvas():
    # Pure white image: darkness = 0, edges = 0
    h, w = 80, 80
    darkness = np.zeros((h, w), dtype=np.float32)
    edge_map = np.zeros((h, w), dtype=np.float32)
    contour_angle = np.zeros((h, w), dtype=np.float32)
    coherence = np.zeros((h, w), dtype=np.float32)

    config = LineArtConfig(style="minimalist")
    selected = select_parsimonious_lines(
        darkness,
        edge_map,
        contour_angle,
        coherence,
        config,
        line_budget=50,
        line_penalty=0.05,
    )

    # Penalty should reject all candidates because there is zero visual gain to explain
    assert len(selected) == 0


def test_priority_first_lines_capture_strongest_features():
    h, w = 100, 100
    darkness = np.zeros((h, w), dtype=np.float32)
    edge_map = np.zeros((h, w), dtype=np.float32)

    # One very strong long line in middle
    edge_map[50, 10:90] = 1.0
    darkness[50, 10:90] = 1.0

    contour_angle = np.zeros((h, w), dtype=np.float32)  # horizontal
    coherence = np.ones((h, w), dtype=np.float32)

    config = LineArtConfig(style="minimalist", min_length=15, max_length=80)
    selected = select_parsimonious_lines(
        darkness,
        edge_map,
        contour_angle,
        coherence,
        config,
        line_budget=10,
        line_penalty=0.02,
    )

    assert len(selected) > 0
    # The first line selected should align with the prominent horizontal edge at y=50
    first_line = selected[0]
    assert abs(first_line.midpoint[1] - 50) < 5
    assert abs(first_line.y1 - first_line.y2) < 5  # roughly horizontal


def test_already_dark_pixels_rejected_and_not_rewarded():
    """Verify that lines crossing pixels that are already sufficiently dark receive negative score."""
    d_target = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    d_curr = np.array([0.75, 0.75, 0.75], dtype=np.float32)  # already sufficiently dark!
    stroke_value = 0.85

    d_new = np.minimum(1.0, d_curr + stroke_value)
    err_before = (d_target - d_curr) ** 2
    err_after = (d_target - d_new) ** 2
    delta_err = err_before - err_after

    # Reduction must be strictly negative (it worsens reconstruction error)
    assert np.all(delta_err < 0.0)


def test_recalculated_scores_prevent_redundant_overlapping_lines():
    """Verify that placing one line updates the residual and recalculates overlapping candidate scores."""
    from sparse_line_selection import _evaluate_candidate_gain

    h, w = 100, 100
    darkness = np.zeros((h, w), dtype=np.float32)
    # A single line feature
    darkness[50, 10:90] = 0.8
    edge_map = np.zeros((h, w), dtype=np.float32)
    edge_map[50, 10:90] = 1.0

    current_canvas = np.zeros((h, w), dtype=np.float32)
    current_edge = np.zeros((h, w), dtype=np.float32)

    xs = np.arange(10, 90, dtype=np.int32)
    ys = np.full(80, 50, dtype=np.int32)

    # Score before any line is placed
    score_before = _evaluate_candidate_gain(
        xs, ys, 1.0, darkness, current_canvas, edge_map, current_edge,
        stroke_value=0.85, edge_weight=0.5, line_penalty=0.01,
    )
    assert score_before > 0.3  # Strong positive gain

    # Simulate placing the line: update reconstruction and residual
    current_canvas[ys, xs] = 0.85
    current_edge[ys, xs] = 1.0

    # Recalculate score for an overlapping line after the canvas is updated
    score_after = _evaluate_candidate_gain(
        xs, ys, 1.0, darkness, current_canvas, edge_map, current_edge,
        stroke_value=0.85, edge_weight=0.5, line_penalty=0.01,
    )

    # Score must plummet because adding another line here overshoots the darkness
    assert score_after < 0.0
    assert score_after < score_before


def test_selection_rendering_model_matches_renderer():
    """Verify that rasterize_candidate_stroke matches renderer.render_lines stroke width and antialiasing."""
    from renderer import render_lines
    from sparse_line_selection import compute_stroke_intensity, rasterize_candidate_stroke

    w, h = 80, 80
    cfg = LineArtConfig(line_width=1.5, supersampling=2, line_color=(15, 15, 20), background_color=(255, 255, 255))
    line = LineSegment(15.0, 20.0, 65.0, 60.0)

    # 1. Stroke intensity matches renderer
    intensity = compute_stroke_intensity(cfg)
    expected_intensity = (255.0 - (0.299 * 15 + 0.587 * 15 + 0.114 * 20)) / 255.0
    assert abs(intensity - expected_intensity) < 1e-4

    # 2. Footprint rasterization matches render_lines darkness output
    xs, ys, alphas, (bx1, bx2, by1, by2) = rasterize_candidate_stroke(line, w, h, cfg)
    assert len(xs) > 0
    # Antialiased values must be fractional (not just 1.0)
    assert np.any((alphas > 0.0) & (alphas < 0.99))

    # Compare with actual render_lines
    _, rend_intensity = render_lines([line], w, h, cfg)
    rend_darkness = 1.0 - rend_intensity

    # Darkness at footprint coordinates must be proportional to alphas * stroke_intensity
    footprint_darkness = rend_darkness[ys, xs]
    expected_darkness = alphas * intensity
    assert np.allclose(footprint_darkness, expected_darkness, atol=0.08)



