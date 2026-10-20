"""
Line Generation module for Mathematical Line Art Reconstruction.
Implements:
- Mapping from image darkness to target line density and spacing
- Adaptive sampling with spatial Poisson-disc/blue-noise characteristics
- Smooth structure-tensor and engraving orientation fields
- Dynamic multi-family crosshatching based on darkness levels
- Edge-preserving line alignment using straight line segments
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from config import LineArtConfig, LineSegment


def compute_target_density(
    darkness: np.ndarray,
    rho_min: float = 0.0,
    rho_max: float = 0.08,
    gamma: float = 1.2,
) -> np.ndarray:
    """
    Compute target line density field:
    rho(x, y) = rho_min + D(x, y)^gamma * (rho_max - rho_min)

    Returns:
        rho: float32 array in [rho_min, rho_max]
    """
    d_clamped = np.clip(darkness, 0.0, 1.0)
    rho = rho_min + np.power(d_clamped, gamma) * (rho_max - rho_min)
    return rho.astype(np.float32)


def compute_target_spacing(rho: np.ndarray, max_spacing: float = 1000.0) -> np.ndarray:
    """
    Compute target spacing:
    spacing(x, y) = 1.0 / rho(x, y)
    Safely handles zero or near-zero density by returning max_spacing.
    """
    spacing = np.full_like(rho, max_spacing, dtype=np.float32)
    valid_mask = rho > 1e-5
    spacing[valid_mask] = 1.0 / rho[valid_mask]
    return spacing


class SpatialHashIndex:
    """
    2D spatial hash grid for fast O(1) neighborhood distance queries
    used in variable-radius Poisson disc sampling.
    """

    def __init__(self, cell_size: float, width: int, height: int):
        self.cell_size = max(1.0, cell_size)
        self.grid_w = int(math.ceil(width / self.cell_size))
        self.grid_h = int(math.ceil(height / self.cell_size))
        self.grid: dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}

    def _coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x / self.cell_size), int(y / self.cell_size))

    def insert(self, x: float, y: float, r: float) -> None:
        cell = self._coord_to_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append((x, y, r))

    def is_valid(self, x: float, y: float, r: float) -> bool:
        """
        Check if point (x, y) with exclusion radius r satisfies minimum distance
        to all existing points in neighboring cells.
        """
        cx, cy = self._coord_to_cell(x, y)
        # Search 3x3 or 5x5 neighborhood depending on radius vs cell size
        search_radius_cells = max(1, int(math.ceil(r / self.cell_size)))
        for nx in range(cx - search_radius_cells, cx + search_radius_cells + 1):
            for ny in range(cy - search_radius_cells, cy + search_radius_cells + 1):
                points = self.grid.get((nx, ny))
                if not points:
                    continue
                for px, py, pr in points:
                    dist_sq = (px - x) * (px - x) + (py - y) * (py - y)
                    min_dist = max(r, pr)
                    if dist_sq < min_dist * min_dist:
                        return False
        return True


def adaptive_sample_points(
    darkness: np.ndarray,
    target_density: np.ndarray,
    config: LineArtConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """
    Sample candidate line center coordinates adaptively based on target density.
    Uses variable-radius Poisson disc sampling to prevent clustering in dark
    areas while maintaining wide spacing in light areas.
    """
    height, width = darkness.shape
    base_spacing = config.base_spacing

    # Approximate minimum spacing for the spatial index cell size
    min_spacing = max(2.5, base_spacing * 0.35)
    spatial_index = SpatialHashIndex(min_spacing, width, height)

    sampled_points: List[Tuple[float, float]] = []

    # Calculate step size for stratified jittered candidate proposals
    step = max(3.0, base_spacing * 0.45)
    xs = np.arange(step / 2.0, width, step)
    ys = np.arange(step / 2.0, height, step)

    # Shuffle grid points to avoid directional bias
    grid_coords = [(x, y) for y in ys for x in xs]
    rng.shuffle(grid_coords)

    rho_max = max(1e-4, config.rho_max)

    for gx, gy in grid_coords:
        # Jitter inside cell
        jx = gx + rng.uniform(-step * 0.45, step * 0.45)
        jy = gy + rng.uniform(-step * 0.45, step * 0.45)

        ix = int(np.clip(jx, 0, width - 1))
        iy = int(np.clip(jy, 0, height - 1))

        d = darkness[iy, ix]
        rho = target_density[iy, ix]

        # Probability check based on density
        prob = min(1.0, (rho / rho_max) * 1.3)
        if prob < 0.03 or rng.uniform(0.0, 1.0) > prob:
            continue

        # Adaptive exclusion radius: smaller in dark regions, larger in light regions
        # Radius r ranges between min_spacing (dark) and base_spacing * 2.5 (bright)
        darkness_factor = math.pow(float(d), 0.75)
        local_radius = base_spacing * (1.6 - 1.15 * darkness_factor)
        local_radius = max(min_spacing, local_radius)

        if spatial_index.is_valid(jx, jy, local_radius):
            spatial_index.insert(jx, jy, local_radius)
            sampled_points.append((jx, jy))

    return sampled_points


def get_active_families(darkness_val: float, max_families: int) -> int:
    """
    Determine the number of active crosshatching families at a point based on darkness.
    - Highlights (darkness < 0.12): 0 families (pure paper white)
    - Quartertones (0.12 <= d < 0.35): 1 family (primary direction)
    - Midtones (0.35 <= d < 0.60): 2 families (crosshatch)
    - Shadows (0.60 <= d < 0.82): 3 families (triple hatch)
    - Deep shadows (d >= 0.82): 4 families (dense multi-angle hatch)
    """
    if darkness_val < 0.10:
        return 0
    elif darkness_val < 0.32:
        return min(1, max_families)
    elif darkness_val < 0.58:
        return min(2, max_families)
    elif darkness_val < 0.82:
        return min(3, max_families)
    else:
        return min(4, max_families)


def create_straight_segment(
    x0: float,
    y0: float,
    angle_rad: float,
    length: float,
    width_limit: int,
    height_limit: int,
    family_id: int = 0,
) -> LineSegment | None:
    """
    Create a straight line segment centered at (x0, y0) with given orientation and length,
    clipped to the image boundary.
    """
    half_len = length / 2.0
    dx = half_len * math.cos(angle_rad)
    dy = half_len * math.sin(angle_rad)

    x1 = x0 - dx
    y1 = y0 - dy
    x2 = x0 + dx
    y2 = y0 + dy

    # Bound check: skip lines completely outside
    if (x1 < 0 and x2 < 0) or (x1 >= width_limit and x2 >= width_limit):
        return None
    if (y1 < 0 and y2 < 0) or (y1 >= height_limit and y2 >= height_limit):
        return None

    # Clamp endpoints to image borders
    x1 = float(np.clip(x1, 0, width_limit - 1))
    y1 = float(np.clip(y1, 0, height_limit - 1))
    x2 = float(np.clip(x2, 0, width_limit - 1))
    y2 = float(np.clip(y2, 0, height_limit - 1))

    # Avoid zero-length degenerate segments
    if math.hypot(x2 - x1, y2 - y1) < 1.0:
        return None

    return LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, family_id=family_id)


def generate_edge_lines(
    edge_map: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    config: LineArtConfig,
    rng: np.random.Generator,
) -> List[LineSegment]:
    """
    Generate edge-aligned straight line segments along detected high-contrast boundaries.
    Segments are oriented tangent to edges: theta = atan2(Gy, Gx) + pi/2.
    Preserves recognizable silhouettes, contours, and facial features.
    """
    height, width = edge_map.shape
    edge_lines: List[LineSegment] = []

    # Find edge pixel coordinates
    edge_indices = np.argwhere(edge_map > 0.5)
    if len(edge_indices) == 0:
        return edge_lines

    # Spacing along edges
    edge_spacing = max(4.0, config.min_length * 0.75)
    spatial_index = SpatialHashIndex(edge_spacing, width, height)

    # Shuffle edge points
    rng.shuffle(edge_indices)

    line_len = config.min_length * 1.1

    for y, x in edge_indices:
        fx, fy = float(x), float(y)
        if not spatial_index.is_valid(fx, fy, edge_spacing):
            continue

        gx = gradient_x[y, x]
        gy = gradient_y[y, x]
        # Tangent to edge contour
        edge_angle = math.atan2(float(gy), float(gx)) + (math.pi / 2.0)

        # Small jitter to prevent robotic alignment
        edge_angle += rng.uniform(-0.08, 0.08)

        seg = create_straight_segment(
            fx, fy, edge_angle, line_len, width, height, family_id=99
        )
        if seg is not None:
            spatial_index.insert(fx, fy, edge_spacing)
            edge_lines.append(seg)

    return edge_lines


def generate_line_art(
    darkness: np.ndarray,
    contour_angle: np.ndarray,
    coherence: np.ndarray,
    edge_map: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    config: LineArtConfig,
    rgb_image: Optional[np.ndarray] = None,
    face_data: Optional[Any] = None,
) -> List[LineSegment]:
    """
    Main line generation orchestrator:
    1. Detects whether a face exists (if face_data not precomputed and rgb_image provided).
    2. Computes adaptive density field (boosted in facial features if face detected).
    3. Guides line orientations along anatomical facial contour angles and structural edges.
    4. Samples seed points using variable-radius spatial indexing.
    5. Generates directional and crosshatching line segments according to style.
    6. Integrates edge-reinforcing and facial feature straight lines.
    """
    height, width = darkness.shape
    rng = np.random.default_rng(config.random_seed)

    # Detect face if image is provided and face_data is not yet cached
    if face_data is None and rgb_image is not None and config.enable_face_detection:
        try:
            from face_processing import detect_face_features

            face_data = detect_face_features(rgb_image, config)
        except Exception:
            face_data = None

    # Minimalist / Parsimonious Pursuit Mode: Solves the minimum-line objective
    if config.style.lower() == "minimalist":
        from sparse_line_selection import select_parsimonious_lines

        return select_parsimonious_lines(
            darkness=darkness,
            edge_map=edge_map,
            contour_angle=contour_angle,
            coherence=coherence,
            config=config,
            line_budget=config.line_budget,
            line_penalty=config.line_penalty,
            face_data=face_data,
            rgb_image=rgb_image,
        )

    # If face is detected, guide contour angles along facial curves
    if face_data is not None and getattr(face_data, "has_face", False):
        if face_data.guided_contour_angles is not None:
            guide_mask = ~np.isnan(face_data.guided_contour_angles)
            contour_angle = contour_angle.copy()
            contour_angle[guide_mask] = face_data.guided_contour_angles[guide_mask]
            coherence = coherence.copy()
            coherence[guide_mask] = 1.0

    # 1. Target density
    target_density = compute_target_density(
        darkness,
        rho_min=config.rho_min,
        rho_max=config.rho_max,
        gamma=config.gamma,
    )

    # Boost target density in facial feature areas if face is detected
    if face_data is not None and getattr(face_data, "has_face", False):
        target_density = target_density * face_data.salience_map

    # 2. Adaptive point sampling
    sample_points = adaptive_sample_points(darkness, target_density, config, rng)


    lines: List[LineSegment] = []
    style = config.style.lower()

    # Pre-parse fixed angles in radians
    fixed_angles_rad = [math.radians(deg) for deg in config.orientation_angles]
    if not fixed_angles_rad:
        fixed_angles_rad = [math.pi / 4.0, 3.0 * math.pi / 4.0, 0.0, math.pi / 2.0]

    for x0, y0 in sample_points:
        ix = int(np.clip(x0, 0, width - 1))
        iy = int(np.clip(y0, 0, height - 1))

        d = float(darkness[iy, ix])
        num_active = get_active_families(d, config.num_families)
        if num_active <= 0:
            continue

        local_coherence = float(coherence[iy, ix])
        local_contour = float(contour_angle[iy, ix])

        # Adapt line length: shorter near curved edges, longer in uniform tonal fields
        len_t = 1.0 - math.pow(float(edge_map[iy, ix]), 0.5)
        seg_length = config.min_length + len_t * (config.max_length - config.min_length)

        # Slightly perturb length for natural hand-drawn appearance
        seg_length *= rng.uniform(0.85, 1.15)

        for fam_idx in range(num_active):
            if style == "contour":
                # Pure contour following and orthogonal hatching
                base_angle = local_contour
                family_offset = (fam_idx * math.pi) / float(max(1, config.num_families))
                angle = base_angle + family_offset
            elif style == "engraving":
                # Fixed global engraving angles, subtly bent by strong contours
                fixed_ang = fixed_angles_rad[fam_idx % len(fixed_angles_rad)]
                if local_coherence > 0.4 and fam_idx == 0:
                    # Blend fixed angle towards contour
                    angle = 0.7 * fixed_ang + 0.3 * local_contour
                else:
                    angle = fixed_ang
            elif style == "crosshatch":
                # Traditional fixed-angle crosshatching
                angle = fixed_angles_rad[fam_idx % len(fixed_angles_rad)]
            else:  # 'adaptive' default
                # Harmonious combination: primary family follows contour when coherent,
                # otherwise follows primary fixed angle; crosshatch families follow offsets.
                if fam_idx == 0:
                    if local_coherence > 0.3:
                        angle = local_contour
                    else:
                        angle = fixed_angles_rad[0]
                elif fam_idx == 1:
                    # Orthogonal or complementary hatch
                    angle = local_contour + (math.pi / 2.0) if local_coherence > 0.3 else fixed_angles_rad[1]
                elif fam_idx == 2:
                    angle = fixed_angles_rad[2 % len(fixed_angles_rad)]
                else:
                    angle = fixed_angles_rad[3 % len(fixed_angles_rad)]

            # Small angular jitter to avoid rigid moire patterns
            jitter = rng.uniform(-0.04, 0.04)
            final_angle = angle + jitter

            seg = create_straight_segment(
                x0, y0, final_angle, seg_length, width, height, family_id=fam_idx
            )
            if seg is not None:
                lines.append(seg)

    # 4. Edge reinforcement lines
    if config.edge_weight > 0.05:
        edge_lines = generate_edge_lines(edge_map, grad_x, grad_y, config, rng)
        lines.extend(edge_lines)

    # 5. Facial feature reinforcement lines (eyes, eyebrows, nose, lips, jawline)
    if face_data is not None and getattr(face_data, "has_face", False):
        lines.extend(face_data.feature_chords)

    return lines

