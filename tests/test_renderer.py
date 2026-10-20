"""
Tests for rendering lines, anti-aliased supersampling, dimensions, and canvas background.
"""

import numpy as np
import pytest

from config import LineArtConfig, LineSegment
from renderer import render_lines


def test_empty_lines_produce_white_canvas():
    config = LineArtConfig(background_color=(255, 255, 255), supersampling=1)
    rgb, gray = render_lines([], width=100, height=80, config=config)

    assert rgb.shape == (80, 100, 3)
    assert gray.shape == (80, 100)
    # Entire canvas should be pure white
    assert np.all(rgb == 255)
    assert np.all(gray == 1.0)


def test_output_dimensions_with_scale():
    config = LineArtConfig()
    lines = [LineSegment(10, 10, 50, 50)]

    # Scale 1.0
    rgb1, _ = render_lines(lines, width=120, height=80, config=config, scale=1.0)
    assert rgb1.shape == (80, 120, 3)

    # Scale 2.5
    rgb2, _ = render_lines(lines, width=120, height=80, config=config, scale=2.5)
    assert rgb2.shape == (200, 300, 3)


def test_line_drawing_creates_dark_marks_on_white_background():
    config = LineArtConfig(
        background_color=(255, 255, 255),
        line_color=(0, 0, 0),
        line_width=2.0,
        supersampling=2,
    )
    # Draw horizontal line across middle
    lines = [LineSegment(x1=10, y1=25, x2=90, y2=25)]
    rgb, gray = render_lines(lines, width=100, height=50, config=config)

    # Background at corners should be pure white
    assert tuple(rgb[0, 0]) == (255, 255, 255)
    assert tuple(rgb[49, 99]) == (255, 255, 255)

    # Points on the line should be dark
    midpoint_color = rgb[25, 50]
    assert np.mean(midpoint_color) < 50  # Near black

    # Overall image has line marks on white background
    white_pixel_ratio = np.mean(gray > 0.95)
    assert white_pixel_ratio > 0.85
    assert white_pixel_ratio < 1.0  # Not completely white
