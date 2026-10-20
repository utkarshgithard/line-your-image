"""
Tests for configuration management, parameter validation, and JSON serialization.
"""

import tempfile
from pathlib import Path
import pytest

from config import LineArtConfig, LineSegment


def test_default_config_valid():
    config = LineArtConfig()
    config.validate()  # Should not raise
    assert config.max_dimension == 800
    assert config.rho_min == 0.0
    assert config.rho_max > 0.0
    assert config.gamma > 0.0


def test_invalid_parameters_raise():
    # Negative max_dimension
    with pytest.raises(ValueError):
        LineArtConfig(max_dimension=-100).validate()

    # Negative rho_min
    with pytest.raises(ValueError):
        LineArtConfig(rho_min=-0.1).validate()

    # rho_max <= rho_min
    with pytest.raises(ValueError):
        LineArtConfig(rho_min=0.1, rho_max=0.05).validate()

    # gamma <= 0
    with pytest.raises(ValueError):
        LineArtConfig(gamma=0.0).validate()

    # min_length > max_length
    with pytest.raises(ValueError):
        LineArtConfig(min_length=50.0, max_length=10.0).validate()

    # Invalid style
    with pytest.raises(ValueError):
        LineArtConfig(style="invalid_style").validate()

    # Invalid edge sensitivity
    with pytest.raises(ValueError):
        LineArtConfig(edge_sensitivity=1.5).validate()


def test_json_roundtrip():
    original = LineArtConfig(
        max_dimension=640,
        rho_max=0.09,
        gamma=1.5,
        style="engraving",
        random_seed=123,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        original.save_json(tmp_path)
        loaded = LineArtConfig.load_json(tmp_path)
        assert loaded.max_dimension == original.max_dimension
        assert loaded.rho_max == original.rho_max
        assert loaded.gamma == original.gamma
        assert loaded.style == original.style
        assert loaded.random_seed == original.random_seed
        assert loaded.orientation_angles == original.orientation_angles
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_line_segment_geometry():
    seg = LineSegment(x1=0.0, y1=0.0, x2=3.0, y2=4.0)
    assert seg.length == pytest.approx(5.0)
    assert seg.midpoint == (1.5, 2.0)
    assert seg.as_tuple == (0.0, 0.0, 3.0, 4.0)
