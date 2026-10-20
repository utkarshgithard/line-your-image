"""
Configuration and data structures for Mathematical Line Art Reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence, Tuple


@dataclass
class LineSegment:
    """Represents a single straight line segment in 2D space."""

    x1: float
    y1: float
    x2: float
    y2: float
    family_id: int = 0
    weight: float = 1.0

    @property
    def length(self) -> float:
        """Euclidean length of the line segment."""
        import math

        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint coordinate (x, y)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def angle_rad(self) -> float:
        """Angle in radians [-pi, pi]."""
        import math

        return math.atan2(self.y2 - self.y1, self.x2 - self.x1)

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return endpoints as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class LineArtConfig:
    """
    Configuration parameters for line-art image reconstruction.
    All parameters are validated and have sensible defaults.
    """

    # Image sizing & preprocessing
    max_dimension: int = 800  # Max working dimension (preserves aspect ratio)
    gaussian_sigma: float = 1.2  # Smoothing strength for noise reduction

    # Density and spacing mapping
    rho_min: float = 0.0  # Minimum line density in bright areas (0 = empty white)
    rho_max: float = 0.08  # Maximum line density in dark areas (lines per pixel)
    gamma: float = 1.2  # Non-linear darkness response (D^gamma)
    base_spacing: float = 12.0  # Base spacing scale in pixels

    # Line geometric dimensions
    line_width: float = 0.8  # Line thickness at working resolution (pixels)
    min_length: float = 8.0  # Minimum line segment length (pixels)

    max_length: float = 28.0  # Maximum line segment length (pixels)

    # Orientation & Crosshatching
    orientation_angles: Tuple[float, ...] = (45.0, 135.0, 0.0, 90.0)  # In degrees
    num_families: int = 4  # Max active crosshatching families in darkest regions
    style: str = "adaptive"  # 'adaptive', 'engraving', 'contour', 'crosshatch', 'minimalist'

    # Edge preservation
    edge_sensitivity: float = 0.4  # Canny / gradient thresholding factor
    edge_weight: float = 0.35  # Importance of placing lines along detected edges

    # Minimal-Line Objective (Sparse Parsimonious Pursuit)
    line_budget: int = 2000  # Hard budget for maximum lines in minimalist mode
    line_penalty: float = 0.08  # L0 penalty per line preventing superfluous marks

    # Iterative Optimization
    optimization_iterations: int = 3  # Refinement passes (0 to disable)
    intensity_weight: float = 1.0  # Weight for pixel intensity matching loss
    edge_loss_weight: float = 0.4  # Weight for edge alignment loss
    structural_weight: float = 0.3  # Weight for structural similarity loss
    regularization_weight: float = 0.05  # Penalty for excessive line overlap

    # Rendering
    supersampling: int = 2  # Supersampling factor for anti-aliasing (1, 2, 3, 4)
    output_scale: float = 1.0  # Export resolution multiplier
    line_color: Tuple[int, int, int] = (15, 15, 20)  # Near-black ink RGB
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Pure white RGB

    # Randomness
    random_seed: int = 42  # Seed for reproducible sampling

    # Face-Aware Processing
    enable_face_detection: bool = True  # Automatically activate if a face is detected
    face_salience_weight: float = 3.0  # Priority multiplier for eyes, eyebrows, nose, lips
    face_model_path: str = "models/face_landmarker.task"


    def validate(self) -> None:
        """Validate parameter ranges and logical consistency."""
        if self.max_dimension <= 0:
            raise ValueError(f"max_dimension must be > 0, got {self.max_dimension}")
        if self.rho_min < 0:
            raise ValueError(f"rho_min must be >= 0, got {self.rho_min}")
        if self.rho_max <= self.rho_min:
            raise ValueError(
                f"rho_max ({self.rho_max}) must be strictly greater than rho_min ({self.rho_min})"
            )
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.base_spacing <= 0:
            raise ValueError(f"base_spacing must be > 0, got {self.base_spacing}")
        if self.line_width <= 0:
            raise ValueError(f"line_width must be > 0, got {self.line_width}")
        if self.min_length <= 0:
            raise ValueError(f"min_length must be > 0, got {self.min_length}")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) cannot be smaller than min_length ({self.min_length})"
            )
        if self.num_families < 1:
            raise ValueError(f"num_families must be >= 1, got {self.num_families}")
        if not (0.0 <= self.edge_sensitivity <= 1.0):
            raise ValueError(
                f"edge_sensitivity must be in [0, 1], got {self.edge_sensitivity}"
            )
        if self.gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {self.gaussian_sigma}")
        if self.optimization_iterations < 0:
            raise ValueError(
                f"optimization_iterations must be >= 0, got {self.optimization_iterations}"
            )
        if self.supersampling < 1:
            raise ValueError(f"supersampling must be >= 1, got {self.supersampling}")
        if self.output_scale <= 0:
            raise ValueError(f"output_scale must be > 0, got {self.output_scale}")
        if self.line_budget <= 0:
            raise ValueError(f"line_budget must be > 0, got {self.line_budget}")
        if self.line_penalty < 0:
            raise ValueError(f"line_penalty must be >= 0, got {self.line_penalty}")

        valid_styles = {"adaptive", "engraving", "contour", "crosshatch", "minimalist"}
        if self.style.lower() not in valid_styles:
            raise ValueError(
                f"style must be one of {valid_styles}, got '{self.style}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["orientation_angles"] = list(self.orientation_angles)
        d["line_color"] = list(self.line_color)
        d["background_color"] = list(self.background_color)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineArtConfig:
        """Create configuration from dictionary with type conversions."""
        data = dict(data)
        if "orientation_angles" in data:
            data["orientation_angles"] = tuple(data["orientation_angles"])
        if "line_color" in data:
            data["line_color"] = tuple(data["line_color"])
        if "background_color" in data:
            data["background_color"] = tuple(data["background_color"])
        config = cls(**data)
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> LineArtConfig:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
