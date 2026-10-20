"""
Iterative Optimization module for Mathematical Line Art Reconstruction.
Implements:
- Multi-component reconstruction loss (intensity error, edge preservation, structural similarity, regularization)
- Residual darkness error map analysis
- Iterative candidate refinement (targeted addition in under-hatched regions, pruning in over-hatched regions)
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from config import LineArtConfig, LineSegment
from renderer import render_lines


def compute_simplified_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two float32 images in [0, 1].
    """
    # Gaussian window smoothing
    kernel_size = 11
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (kernel_size, kernel_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (kernel_size, kernel_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (kernel_size, kernel_size), sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-6)
    return float(np.mean(ssim_map))


def compute_reconstruction_loss(
    rendered_intensity: np.ndarray,
    target_intensity: np.ndarray,
    edge_map: np.ndarray,
    lines: Sequence[LineSegment],
    config: LineArtConfig,
) -> Tuple[float, dict[str, float]]:
    """
    Evaluate multi-component reconstruction loss:
    1. Pixel Intensity Error (MSE with visual perception blur)
    2. Edge Preservation Error (correlation / overlap along structural edges)
    3. Structural Similarity (1.0 - SSIM)
    4. Regularization Penalty (line budget and stray line penalty)

    Returns:
        total_loss: Weighted sum of all loss components
        components: Dictionary containing individual loss values
    """
    # 1. Perceptual intensity matching:
    # Human vision blurs thin lines into perceived tonal density
    perceived_rendered = cv2.GaussianBlur(rendered_intensity, (7, 7), sigmaX=2.0)
    perceived_target = cv2.GaussianBlur(target_intensity, (7, 7), sigmaX=1.5)

    intensity_loss = float(np.mean((perceived_rendered - perceived_target) ** 2))

    # 2. Edge preservation loss:
    # Compute gradients of the rendered linework
    gx = cv2.Sobel(rendered_intensity, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(rendered_intensity, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    rendered_edges = np.sqrt(gx * gx + gy * gy)
    rendered_edges = np.clip(rendered_edges * 4.0, 0.0, 1.0)

    # Edge loss: difference where target edge map is prominent
    edge_mask = edge_map > 0.3
    if np.any(edge_mask):
        edge_loss = float(np.mean(np.abs(rendered_edges[edge_mask] - edge_map[edge_mask])))
    else:
        edge_loss = 0.0

    # 3. Structural similarity loss:
    ssim_val = compute_simplified_ssim(perceived_rendered, perceived_target)
    structural_loss = max(0.0, 1.0 - ssim_val)

    # 4. Regularization penalty:
    # Penalize excessive line counts beyond expected bounds and stray lines
    height, width = target_intensity.shape
    expected_lines = (width * height) * (config.rho_max * 0.3)
    count_penalty = max(0.0, (len(lines) - expected_lines) / max(1.0, expected_lines))
    regularization_loss = float(count_penalty * 0.1)

    total_loss = (
        config.intensity_weight * intensity_loss
        + config.edge_loss_weight * edge_loss
        + config.structural_weight * structural_loss
        + config.regularization_weight * regularization_loss
    )

    components = {
        "total_loss": total_loss,
        "intensity_loss": intensity_loss,
        "edge_loss": edge_loss,
        "structural_loss": structural_loss,
        "regularization_loss": regularization_loss,
        "ssim": ssim_val,
        "line_count": float(len(lines)),
    }

    return total_loss, components


def refine_line_arrangements(
    initial_lines: List[LineSegment],
    target_intensity: np.ndarray,
    darkness: np.ndarray,
    edge_map: np.ndarray,
    contour_angle: np.ndarray,
    config: LineArtConfig,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> Tuple[List[LineSegment], List[dict[str, float]]]:
    """
    Iterative refinement loop:
    1. Render current lines.
    2. Compute reconstruction loss and residual darkness error map:
       Residual = D_target(x, y) - D_rendered(x, y).
    3. Add corrective lines in regions with significant under-coverage (Residual > 0).
    4. Remove/prune redundant lines in regions with over-coverage (Residual < 0).
    5. Retain edits if overall reconstruction loss improves.
    """
    if config.optimization_iterations <= 0:
        _, initial_intensity = render_lines(
            initial_lines, target_intensity.shape[1], target_intensity.shape[0], config
        )
        _, initial_stats = compute_reconstruction_loss(
            initial_intensity, target_intensity, edge_map, initial_lines, config
        )
        return initial_lines, [initial_stats]

    height, width = target_intensity.shape
    current_lines = list(initial_lines)
    history: List[dict[str, float]] = []

    rng = np.random.default_rng(config.random_seed + 101)

    # Initial evaluation
    _, rendered_gray = render_lines(current_lines, width, height, config)
    best_loss, stats = compute_reconstruction_loss(
        rendered_gray, target_intensity, edge_map, current_lines, config
    )
    history.append(stats)

    if progress_callback:
        progress_callback(0, config.optimization_iterations, stats)

    for iteration in range(1, config.optimization_iterations + 1):
        # 1. Compute darkness of rendered linework
        rendered_darkness = 1.0 - rendered_gray

        # Smooth to get local area density
        perceived_rendered_d = cv2.GaussianBlur(rendered_darkness, (9, 9), sigmaX=2.5)
        perceived_target_d = cv2.GaussianBlur(darkness, (9, 9), sigmaX=2.0)

        # Residual map: positive = under-covered (too bright); negative = over-covered (too dark)
        residual = perceived_target_d - perceived_rendered_d

        candidate_modifications: List[LineSegment] = []

        # A. Prune lines from severely over-covered regions (Residual < -0.18)
        pruned_lines: List[LineSegment] = []
        for line in current_lines:
            mx, my = line.midpoint
            imx = int(np.clip(mx, 0, width - 1))
            imy = int(np.clip(my, 0, height - 1))
            res_val = residual[imy, imx]

            # If over-hatched and not an essential edge line, consider pruning
            if res_val < -0.18 and line.family_id != 99:
                # Stochastic pruning proportional to over-darkness
                if rng.uniform(0.0, 1.0) < min(0.6, abs(res_val)):
                    continue  # Drop this line
            pruned_lines.append(line)

        # B. Propose additions in under-covered regions (Residual > 0.15)
        under_covered_coords = np.argwhere(residual > 0.16)
        if len(under_covered_coords) > 0:
            sample_size = min(len(under_covered_coords), max(10, int(len(current_lines) * 0.12)))
            chosen_indices = rng.choice(len(under_covered_coords), size=sample_size, replace=False)

            for idx in chosen_indices:
                y, x = under_covered_coords[idx]
                fx, fy = float(x), float(y)
                local_angle = float(contour_angle[y, x])

                # Cross angle
                cross_angle = local_angle + (math.pi / 2.0) + rng.uniform(-0.1, 0.1)
                seg_length = (config.min_length + config.max_length) * 0.5 * rng.uniform(0.8, 1.1)

                half_len = seg_length / 2.0
                x1 = float(np.clip(fx - half_len * math.cos(cross_angle), 0, width - 1))
                y1 = float(np.clip(fy - half_len * math.sin(cross_angle), 0, height - 1))
                x2 = float(np.clip(fx + half_len * math.cos(cross_angle), 0, width - 1))
                y2 = float(np.clip(fy + half_len * math.sin(cross_angle), 0, height - 1))

                if math.hypot(x2 - x1, y2 - y1) >= 2.0:
                    candidate_modifications.append(
                        LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, family_id=2)
                    )

        # Combine proposed adjustments
        trial_lines = pruned_lines + candidate_modifications

        # Evaluate trial candidate set
        _, trial_gray = render_lines(trial_lines, width, height, config)
        trial_loss, trial_stats = compute_reconstruction_loss(
            trial_gray, target_intensity, edge_map, trial_lines, config
        )

        # Greedy acceptance criterion
        if trial_loss < best_loss:
            best_loss = trial_loss
            current_lines = trial_lines
            rendered_gray = trial_gray
            stats = trial_stats
        else:
            # If trial was not strictly better, keep current lines and record stats
            stats = {**history[-1], "rejected_iteration": float(iteration)}

        history.append(stats)
        if progress_callback:
            progress_callback(iteration, config.optimization_iterations, stats)

    return current_lines, history
