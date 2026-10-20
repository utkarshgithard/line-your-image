"""
Minimal-Line Objective & Parsimonious Line Pursuit Module.

Implements a sparse line-selection algorithm using Orthogonal Matching Pursuit (OMP)
principles with L0 line penalties to reconstruct images with the FEWEST POSSIBLE CLEAR LINES:
1. Candidate Generation: Edge tangent chords, Hough segments, and contour chords.
2. Structural Prioritization: Lines weighted by edge explanation and shape silhouette.
3. Fast Incremental Residual Evaluation: Bresenham raster scanning of candidate marginal gains.
4. Line Penalty: Fixed L0 cost per line preventing unnecessary marks.
5. Duplicate Rejection: Non-Maximum Suppression (NMS) and residual saturation.
6. Budget & Quality Stopping: Halts when line budget or target gain threshold is reached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from config import LineArtConfig, LineSegment


def bresenham_line_pixels(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integer pixel coordinates along straight line segment from (x1, y1) to (x2, y2)
    using vectorized Bresenham / interpolation clipped to image bounds.
    """
    length = int(math.hypot(x2 - x1, y2 - y1))
    if length <= 0:
        return np.array([np.clip(x1, 0, w - 1)]), np.array([np.clip(y1, 0, h - 1)])

    num_pts = max(2, length + 1)
    xs = np.linspace(x1, x2, num_pts)
    ys = np.linspace(y1, y2, num_pts)

    ixs = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
    iys = np.clip(np.round(ys).astype(np.int32), 0, h - 1)

    # Unique consecutive points
    mask = np.ones(len(ixs), dtype=bool)
    mask[1:] = (ixs[1:] != ixs[:-1]) | (iys[1:] != iys[:-1])
    return ixs[mask], iys[mask]


def generate_candidate_pool(
    darkness: np.ndarray,
    edge_map: np.ndarray,
    contour_angle: np.ndarray,
    coherence: np.ndarray,
    config: LineArtConfig,
    max_candidates: int = 4000,
    face_data: Optional[Any] = None,
) -> List[LineSegment]:
    """
    Step 1 & 2: Generate candidate straight line segments and prioritize lines
    that explain important edges, facial features, and large-scale shapes.
    """
    h, w = darkness.shape
    candidates: List[LineSegment] = []
    rng = np.random.default_rng(config.random_seed)

    # 0. If face detected, include anatomical facial feature chords directly
    if face_data is not None and getattr(face_data, "has_face", False):
        candidates.extend(face_data.feature_chords)


    # A. Probabilistic Hough Transform for salient straight linear boundaries
    uint8_edges = (edge_map * 255).astype(np.uint8)
    hough_lines = cv2.HoughLinesP(
        uint8_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(12, int(config.min_length)),
        maxLineGap=6,
    )
    if hough_lines is not None:
        for item in hough_lines:
            coords = item.reshape(-1)
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                candidates.append(LineSegment(float(x1), float(y1), float(x2), float(y2), family_id=99))

    # B. Contour chords along detected boundary curves
    contours, _ = cv2.findContours(uint8_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2)
        step = max(3, len(pts) // 15)
        for i in range(0, len(pts), step):
            for chord_len in (8, 16, 32):
                j = (i + chord_len) % len(pts)
                p1, p2 = pts[i], pts[j]
                dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if config.min_length <= dist <= config.max_length * 2.5:
                    candidates.append(LineSegment(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), family_id=99))

    # C. Edge Tangent Lines (oriented along contour angle at salient points)
    edge_indices = np.argwhere(edge_map > 0.4)
    if len(edge_indices) > 0:
        sample_count = min(len(edge_indices), 800)
        chosen = rng.choice(len(edge_indices), size=sample_count, replace=False)
        for idx in chosen:
            y, x = edge_indices[idx]
            angle = contour_angle[y, x]
            for length in (config.min_length * 1.2, config.max_length * 1.5):
                half = length / 2.0
                x1 = float(np.clip(x - half * math.cos(angle), 0, w - 1))
                y1 = float(np.clip(y - half * math.sin(angle), 0, h - 1))
                x2 = float(np.clip(x + half * math.cos(angle), 0, w - 1))
                y2 = float(np.clip(y + half * math.sin(angle), 0, h - 1))
                if math.hypot(x2 - x1, y2 - y1) >= config.min_length:
                    candidates.append(LineSegment(x1, y1, x2, y2, family_id=0))

    # D. Tonal shading chord candidates across dark regions
    dark_indices = np.argwhere(darkness > 0.35)
    if len(dark_indices) > 0:
        sample_count = min(len(dark_indices), 600)
        chosen = rng.choice(len(dark_indices), size=sample_count, replace=False)
        for idx in chosen:
            y, x = dark_indices[idx]
            # Primary contour angle and fixed angles
            ang = contour_angle[y, x] if coherence[y, x] > 0.3 else math.pi / 4.0
            for l_scale in (1.0, 1.8):
                length = config.max_length * l_scale
                half = length / 2.0
                x1 = float(np.clip(x - half * math.cos(ang), 0, w - 1))
                y1 = float(np.clip(y - half * math.sin(ang), 0, h - 1))
                x2 = float(np.clip(x + half * math.cos(ang), 0, w - 1))
                y2 = float(np.clip(y + half * math.sin(ang), 0, h - 1))
                candidates.append(LineSegment(x1, y1, x2, y2, family_id=1))

    # Limit and shuffle
    if len(candidates) > max_candidates:
        rng.shuffle(candidates)
        candidates = candidates[:max_candidates]

    return candidates


def line_similarity(l1: LineSegment, l2: LineSegment) -> float:
    """
    Measure similarity between two line segments based on angle and midpoint distance.
    Returns value in [0, 1] where 1.0 means near-identical duplicates.
    """
    # Angle difference modulo pi
    d_theta = abs(l1.angle_rad - l2.angle_rad) % math.pi
    if d_theta > math.pi / 2.0:
        d_theta = math.pi - d_theta

    if d_theta > 0.15:  # > ~8.5 degrees
        return 0.0

    # Midpoint distance
    m1 = l1.midpoint
    m2 = l2.midpoint
    dist = math.hypot(m1[0] - m2[0], m1[1] - m2[1])
    avg_len = (l1.length + l2.length) * 0.5
    if avg_len <= 0:
        return 0.0

    norm_dist = dist / avg_len
    if norm_dist > 0.25:
        return 0.0

    return (1.0 - (d_theta / 0.15)) * (1.0 - (norm_dist / 0.25))


def compute_stroke_intensity(config: LineArtConfig) -> float:
    """
    Calculate normalized stroke darkness intensity in [0, 1] based on
    config.line_color and config.background_color matching renderer.render_lines.
    """
    bg = config.background_color
    fg = config.line_color
    bg_gray = (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]) / 255.0
    fg_gray = (0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]) / 255.0
    return float(max(0.0, min(1.0, bg_gray - fg_gray)))


def rasterize_candidate_stroke(
    line: LineSegment,
    width: int,
    height: int,
    config: LineArtConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Rasterize an antialiased line stroke using the EXACT rendering parameters from renderer.py:
    - Same stroke width (config.line_width)
    - Same supersampling (config.supersampling)
    - Same cv2.LINE_AA antialiasing filter
    - Same INTER_AREA downsampling

    Returns:
        xs: 1D array of pixel x-coordinates affected by the stroke in working space
        ys: 1D array of pixel y-coordinates affected by the stroke in working space
        alphas: 1D array of float32 antialiased coverage weights in (0.0, 1.0]
        bbox: (min_x, max_x, min_y, max_y) in working coordinates
    """
    supersample = max(1, int(config.supersampling))
    margin = int(math.ceil(config.line_width + 3))

    bx1 = max(0, int(math.floor(min(line.x1, line.x2))) - margin)
    by1 = max(0, int(math.floor(min(line.y1, line.y2))) - margin)
    bx2 = min(width - 1, int(math.ceil(max(line.x1, line.x2))) + margin)
    by2 = min(height - 1, int(math.ceil(max(line.y1, line.y2))) + margin)

    pw = bx2 - bx1 + 1
    ph = by2 - by1 + 1

    if pw <= 0 or ph <= 0:
        empty = np.array([], dtype=np.int32)
        empty_f = np.array([], dtype=np.float32)
        return empty, empty, empty_f, (0, 0, 0, 0)

    if supersample > 1:
        super_pw = pw * supersample
        super_ph = ph * supersample
        patch = np.zeros((super_ph, super_pw), dtype=np.uint8)
        p1 = (
            int(round((line.x1 - bx1) * supersample)),
            int(round((line.y1 - by1) * supersample)),
        )
        p2 = (
            int(round((line.x2 - bx1) * supersample)),
            int(round((line.y2 - by1) * supersample)),
        )
        thickness = max(1, int(round(config.line_width * supersample)))
        cv2.line(patch, p1, p2, 255, thickness=thickness, lineType=cv2.LINE_AA)
        down = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
        local_ys, local_xs = np.where(down > 0)
        alphas = down[local_ys, local_xs].astype(np.float32) / 255.0
    else:
        patch = np.zeros((ph, pw), dtype=np.uint8)
        p1 = (int(round(line.x1 - bx1)), int(round(line.y1 - by1)))
        p2 = (int(round(line.x2 - bx1)), int(round(line.y2 - by1)))
        thickness = max(1, int(round(config.line_width)))
        cv2.line(patch, p1, p2, 255, thickness=thickness, lineType=cv2.LINE_AA)
        local_ys, local_xs = np.where(patch > 0)
        alphas = patch[local_ys, local_xs].astype(np.float32) / 255.0

    if len(local_xs) == 0:
        empty = np.array([], dtype=np.int32)
        empty_f = np.array([], dtype=np.float32)
        return empty, empty, empty_f, (bx1, bx2, by1, by2)

    xs = (local_xs + bx1).astype(np.int32)
    ys = (local_ys + by1).astype(np.int32)
    return xs, ys, alphas, (bx1, bx2, by1, by2)


def _evaluate_candidate_gain(
    xs: np.ndarray,
    ys: np.ndarray,
    length_salience: float,
    darkness: np.ndarray,
    current_canvas_darkness: np.ndarray,
    edge_map: np.ndarray,
    current_edge_coverage: np.ndarray,
    stroke_value: float = 0.85,
    edge_weight: float = 0.5,
    line_penalty: float = 0.08,
    alphas: np.ndarray | None = None,
    salience_map: np.ndarray | None = None,
) -> float:
    """
    Calculate the exact net error reduction when rendering a candidate line,
    using identical stroke width, antialiasing coverage (alphas), stroke intensity,
    and optical alpha compositing behavior as the renderer.
    Scales error reduction by salience_map to prioritize facial features.
    """
    if len(xs) == 0:
        return -1e9

    if alphas is None:
        alphas = np.ones(len(xs), dtype=np.float32)

    # Query current canvas and target along candidate line footprint
    d_target = darkness[ys, xs]
    d_before = current_canvas_darkness[ys, xs]

    # Exact optical compositing matching renderer.py:
    # Adding antialiased ink with fractional coverage alpha and stroke intensity:
    d_after = d_before + alphas * np.maximum(0.0, stroke_value - d_before)

    # Exact reduction in squared reconstruction error for darkness:
    delta_err_d = (d_target - d_before) ** 2 - (d_target - d_after) ** 2

    # Exact reduction in edge omission error:
    e_target = edge_map[ys, xs]
    e_before = current_edge_coverage[ys, xs]
    e_after = np.minimum(1.0, e_before + alphas)
    delta_err_e = np.maximum(0.0, (e_target - e_before) ** 2 - (e_target - e_after) ** 2)

    # Weighted reconstruction error reduction (prioritizes facial features via salience_map)
    if salience_map is not None:
        sal = salience_map[ys, xs]
        total_error_reduction = float(
            np.sum(delta_err_d * sal) + edge_weight * np.sum(delta_err_e * sal)
        )
    else:
        total_error_reduction = float(np.sum(delta_err_d) + edge_weight * np.sum(delta_err_e))

    # Average improvement per pixel weighted by antialiased coverage
    weight_sum = float(np.sum(alphas))
    if weight_sum <= 0:
        return -1e9

    mean_reduction = total_error_reduction / weight_sum
    return (mean_reduction * length_salience) - line_penalty


def select_parsimonious_lines(
    darkness: np.ndarray,
    edge_map: np.ndarray,
    contour_angle: np.ndarray,
    coherence: np.ndarray,
    config: LineArtConfig,
    line_budget: int = 150,
    line_penalty: float = 0.08,
    min_improvement: float = 0.005,
    progress_callback: Callable[[int, int, float], None] | None = None,
    face_data: Optional[Any] = None,
    rgb_image: Optional[np.ndarray] = None,
) -> List[LineSegment]:
    """
    The 6-Step Minimal-Line Objective & Selection Algorithm:
    1. Generate candidate straight line segments.
    2. Prioritize lines that explain important edges and large-scale shapes.
    3. Measure improvement (reduction of reconstruction error) when rendering each candidate.
    4. Penalize each additional line with L0 regularization cost.
    5. Reject duplicate or near-duplicate lines via Non-Maximum Suppression (NMS).
    6. Stop when line budget is reached or marginal improvement falls below threshold.

    Selects strictly ONE line per iteration: selects the best candidate, updates the
    reconstruction and residual, then recalculates candidate scores to prevent stale scores.
    Uses the EXACT same stroke width, intensity, antialiasing, and compositing behavior
    as renderer.render_lines.

    If a face is detected (face_data), activates face-aware processing:
    - Guides candidate generation along eyes, eyebrows, nose, lips, jawline.
    - Guides contour angles along facial landmark curves.
    - Allocates higher reconstruction priority to facial features via salience weighting.
    """
    h, w = darkness.shape

    # Automatically detect face if not precomputed and image is supplied
    if face_data is None and rgb_image is not None and config.enable_face_detection:
        try:
            from face_processing import detect_face_features

            face_data = detect_face_features(rgb_image, config)
        except Exception:
            face_data = None

    salience_map = None
    if face_data is not None and getattr(face_data, "has_face", False):
        salience_map = face_data.salience_map
        edge_map = np.maximum(edge_map, face_data.feature_edges)
        if face_data.guided_contour_angles is not None:
            guide_mask = ~np.isnan(face_data.guided_contour_angles)
            contour_angle = contour_angle.copy()
            contour_angle[guide_mask] = face_data.guided_contour_angles[guide_mask]
            coherence = coherence.copy()
            coherence[guide_mask] = 1.0

    candidates = generate_candidate_pool(
        darkness, edge_map, contour_angle, coherence, config, face_data=face_data
    )
    if not candidates:
        return []

    stroke_intensity = compute_stroke_intensity(config)
    edge_weight = config.edge_weight * 2.5
    supersample = max(1, int(config.supersampling))
    thickness = max(1, int(round(config.line_width * supersample)))

    # Real canvas representation identical to renderer.py
    super_w = w * supersample
    super_h = h * supersample
    super_canvas = np.full((super_h, super_w, 3), config.background_color, dtype=np.uint8)
    current_canvas_darkness = np.zeros((h, w), dtype=np.float32)
    current_edge_coverage = np.zeros((h, w), dtype=np.float32)

    bg_gray = (
        0.299 * config.background_color[0]
        + 0.587 * config.background_color[1]
        + 0.114 * config.background_color[2]
    ) / 255.0

    selected_lines: List[LineSegment] = []

    # Pre-rasterize candidate coordinates with exact stroke width, supersampling and antialiasing
    precomputed_paths = []
    num_candidates = len(candidates)
    scores = np.empty(num_candidates, dtype=np.float32)
    cand_min_x = np.empty(num_candidates, dtype=np.int32)
    cand_max_x = np.empty(num_candidates, dtype=np.int32)
    cand_min_y = np.empty(num_candidates, dtype=np.int32)
    cand_max_y = np.empty(num_candidates, dtype=np.int32)
    length_saliences = np.empty(num_candidates, dtype=np.float32)

    for i, line in enumerate(candidates):
        xs, ys, alphas, (bx1, bx2, by1, by2) = rasterize_candidate_stroke(line, w, h, config)
        precomputed_paths.append((line, xs, ys, alphas, (bx1, bx2, by1, by2)))

        if len(xs) == 0:
            scores[i] = -1e9
            cand_min_x[i] = cand_max_x[i] = cand_min_y[i] = cand_max_y[i] = 0
            length_saliences[i] = 0.0
            continue

        cand_min_x[i] = bx1
        cand_max_x[i] = bx2
        cand_min_y[i] = by1
        cand_max_y[i] = by2
        length = float(line.length)
        salience = min(1.0, math.sqrt(length / max(1.0, config.min_length)))
        length_saliences[i] = salience

        scores[i] = _evaluate_candidate_gain(
            xs,
            ys,
            salience,
            darkness,
            current_canvas_darkness,
            edge_map,
            current_edge_coverage,
            stroke_intensity,
            edge_weight,
            line_penalty,
            alphas=alphas,
            salience_map=salience_map,
        )

    # One line per iteration matching pursuit:
    # 1. Select the single best candidate line
    # 2. Update reconstruction and residual using the exact renderer model
    # 3. Recalculate scores for candidate lines affected by this placement
    while len(selected_lines) < line_budget:
        chosen_idx = -1
        while True:
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score < min_improvement:
                # No remaining candidate provides sufficient reconstruction improvement
                break

            cand_line = precomputed_paths[best_idx][0]

            # Step 5: Duplicate / redundancy rejection check against recent lines
            is_dup = False
            for existing in selected_lines[-40:]:
                if line_similarity(cand_line, existing) > 0.65:
                    is_dup = True
                    break

            if is_dup:
                scores[best_idx] = -1e9  # Disqualify duplicate line
                continue

            chosen_idx = best_idx
            break

        if chosen_idx == -1:
            break

        # Accept the single best candidate line
        best_line, xs, ys, alphas, (bx1, bx2, by1, by2) = precomputed_paths[chosen_idx]
        selected_lines.append(best_line)
        scores[chosen_idx] = -1e9  # Mark as selected

        # Render stroke onto super_canvas with identical parameters to renderer.py
        p1 = (int(round(best_line.x1 * supersample)), int(round(best_line.y1 * supersample)))
        p2 = (int(round(best_line.x2 * supersample)), int(round(best_line.y2 * supersample)))
        cv2.line(super_canvas, p1, p2, config.line_color, thickness=thickness, lineType=cv2.LINE_AA)

        # Update edge coverage
        current_edge_coverage[ys, xs] = np.minimum(1.0, current_edge_coverage[ys, xs] + alphas)

        # Update current_canvas_darkness from the rendered super_canvas (exact compositing)
        sub_super = super_canvas[
            by1 * supersample : (by2 + 1) * supersample,
            bx1 * supersample : (bx2 + 1) * supersample,
        ]
        if supersample > 1:
            sub_rgb = cv2.resize(sub_super, (bx2 - bx1 + 1, by2 - by1 + 1), interpolation=cv2.INTER_AREA)
        else:
            sub_rgb = sub_super
        sub_gray = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        current_canvas_darkness[by1 : by2 + 1, bx1 : bx2 + 1] = np.clip(
            (bg_gray - sub_gray) / max(1e-5, bg_gray), 0.0, 1.0
        )

        # Recalculate scores for candidate lines in the affected spatial region
        overlap_mask = (
            (scores > -1e8)
            & (cand_max_x >= bx1)
            & (cand_min_x <= bx2)
            & (cand_max_y >= by1)
            & (cand_min_y <= by2)
        )
        overlapping_indices = np.where(overlap_mask)[0]

        for idx in overlapping_indices:
            _, xs_c, ys_c, alphas_c, _ = precomputed_paths[idx]
            scores[idx] = _evaluate_candidate_gain(
                xs_c,
                ys_c,
                length_saliences[idx],
                darkness,
                current_canvas_darkness,
                edge_map,
                current_edge_coverage,
                stroke_intensity,
                edge_weight,
                line_penalty,
                alphas=alphas_c,
                salience_map=salience_map,
            )

        if progress_callback and (len(selected_lines) % 10 == 0 or len(selected_lines) == line_budget):
            progress_callback(len(selected_lines), line_budget, best_score)

    return selected_lines

