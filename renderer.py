"""
Rendering module for Mathematical Line Art Reconstruction.
Renders straight line segments onto a clean white canvas using OpenCV with
anti-aliased supersampling, and provides high-resolution export and side-by-side
Matplotlib comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import LineArtConfig, LineSegment


def render_lines(
    lines: Sequence[LineSegment],
    width: int,
    height: int,
    config: LineArtConfig,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render line segments onto a canvas.

    Args:
        lines: Sequence of LineSegment objects defined in working coordinate space [0, width] x [0, height].
        width: Working canvas width in pixels.
        height: Working canvas height in pixels.
        config: LineArtConfig containing colors, line width, and supersampling settings.
        scale: Resolution multiplier for final export (default 1.0 = working resolution).

    Returns:
        rgb_canvas: uint8 numpy array of shape (final_height, final_width, 3) in RGB order.
        normalized_intensity: float32 array in [0.0, 1.0] representing grayscale brightness.
    """
    final_w = max(1, int(round(width * scale)))
    final_h = max(1, int(round(height * scale)))

    supersample = max(1, int(config.supersampling))
    super_w = final_w * supersample
    super_h = final_h * supersample

    # Fill canvas with background color (default white 255, 255, 255)
    bg_color = config.background_color
    canvas = np.full((super_h, super_w, 3), bg_color, dtype=np.uint8)

    # Line color
    line_color = config.line_color

    # Coordinate scaling factor from working coordinates to supersampled canvas
    coord_scale_x = float(super_w) / float(width)
    coord_scale_y = float(super_h) / float(height)

    # Line width at supersampled scale: thickness controls antialiasing radius
    thickness = max(1, int(round(config.line_width * scale * supersample)))

    for seg in lines:
        p1 = (
            int(round(seg.x1 * coord_scale_x)),
            int(round(seg.y1 * coord_scale_y)),
        )
        p2 = (
            int(round(seg.x2 * coord_scale_x)),
            int(round(seg.y2 * coord_scale_y)),
        )

        cv2.line(
            canvas,
            p1,
            p2,
            line_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # Downsample back to target dimensions with high quality INTER_AREA anti-aliasing
    if supersample > 1:
        rgb_canvas = cv2.resize(
            canvas,
            (final_w, final_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        rgb_canvas = canvas

    # Grayscale intensity normalized to [0, 1]
    gray = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2GRAY)
    normalized_intensity = gray.astype(np.float32) / 255.0

    return rgb_canvas, normalized_intensity


def save_image(image_rgb: np.ndarray, output_path: str | Path) -> None:
    """Save an RGB numpy array to disk as a high-resolution image (e.g. PNG)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(out_path, format="PNG", compress_level=6)


def create_comparison_figure(
    original_rgb: np.ndarray,
    grayscale_intensity: np.ndarray,
    line_art_rgb: np.ndarray,
    title: str = "Mathematical Line Art Reconstruction",
) -> plt.Figure:
    """
    Create a 1x3 Matplotlib comparison figure showing:
    1. Original Color/Input Image
    2. Target Grayscale Intensity
    3. Mathematical Line Art Reconstruction
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor("#f8f9fa")

    # 1. Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Input", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")

    # 2. Grayscale Target
    axes[1].imshow(grayscale_intensity, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grayscale Target I(x, y)", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")

    # 3. Line Art Reconstruction
    axes[2].imshow(line_art_rgb)
    axes[2].set_title(
        "Line Art Reconstruction (Thin Lines)",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="heavy", y=0.98)
    plt.tight_layout()
    return fig


def show_or_save_preview(
    original_rgb: np.ndarray,
    grayscale_intensity: np.ndarray,
    line_art_rgb: np.ndarray,
    save_path: str | Path | None = None,
    show_window: bool = False,
) -> None:
    """
    Display side-by-side preview via Matplotlib and/or save comparison image to disk.
    """
    fig = create_comparison_figure(original_rgb, grayscale_intensity, line_art_rgb)

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=180)

    if show_window:
        plt.show()
    else:
        plt.close(fig)
