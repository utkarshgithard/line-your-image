"""
Desktop real-time animator for Mathematical Line Art Reconstruction.
Displays a native window where lines are placed one by one on a virtual sheet of paper.

Usage:
    python animate.py --input samples/sample_portrait.png
    python animate.py --input samples/sample_landscape.png --speed 10 --style engraving
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import LineArtConfig, LineSegment
from image_processing import (
    apply_gaussian_blur,
    compute_darkness,
    compute_gradients,
    compute_structure_tensor,
    detect_edges,
    load_image,
)
from line_generation import generate_line_art
from optimization import refine_line_arrangements


def animate_lines(
    lines: list[LineSegment],
    width: int,
    height: int,
    config: LineArtConfig,
    lines_per_frame: int = 4,
    scale: float = 1.0,
    window_name: str = "Mathematical Line Art - Live Paper Sheet",
) -> None:
    """
    Renders lines one by one (or in small batches) in an OpenCV GUI window.
    """
    disp_w = int(round(width * scale))
    disp_h = int(round(height * scale))

    # Initialize pure white sheet of paper (255, 255, 255)
    canvas = np.full((disp_h, disp_w, 3), 255, dtype=np.uint8)

    # Sort lines: primary contour families first, followed by crosshatch families, then edge lines
    def sort_key(line: LineSegment):
        # 99 is edge line; 0 is primary family; 1, 2, 3 are crosshatches
        return (line.family_id, -line.length)

    sorted_lines = sorted(lines, key=sort_key)
    total_lines = len(sorted_lines)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Initialize stroke width (default thinner strokes: 0.8px)
    current_stroke_width = max(0.3, float(config.line_width))
    line_thickness = max(1, int(round(current_stroke_width * scale)))

    print("\n" + "=" * 60)
    print("  LIVE MATHEMATICAL LINE DRAWING ANIMATION")
    print("=" * 60)
    print(f"Total lines to draw: {total_lines:,}")
    print(f"Initial stroke width: {current_stroke_width:.1f} px (thickness {line_thickness})")
    print("Controls:")
    print("  [SPACE]       Pause / Resume")
    print("  [+] / [=]     Increase drawing speed")
    print("  [-] / [_]     Decrease drawing speed")
    print("  [W] / [[]     Thinner strokes (regulate line width down)")
    print("  [T] / []]     Thicker strokes (regulate line width up)")
    print("  [Trackbar]    Slide 'Stroke Width' bar at top of window")
    print("  [R]           Restart drawing")
    print("  [S]           Save current snapshot to 'animated_snapshot.png'")
    print("  [Q] / [ESC]   Exit")
    print("=" * 60 + "\n")

    current_idx = 0
    paused = False
    speed = max(1, lines_per_frame)
    color = tuple(reversed(config.line_color))  # BGR order for OpenCV

    last_pt = None
    width_changed = False

    def on_width_trackbar(val: int):
        nonlocal current_stroke_width, line_thickness, width_changed
        new_w = max(0.3, val / 10.0)
        if abs(new_w - current_stroke_width) > 0.04:
            current_stroke_width = new_w
            line_thickness = max(1, int(round(current_stroke_width * scale)))
            width_changed = True

    try:
        cv2.createTrackbar(
            "Stroke Width (x0.1px)",
            window_name,
            max(3, min(30, int(round(current_stroke_width * 10)))),
            30,
            on_width_trackbar,
        )
    except Exception:
        pass

    def redraw_canvas_to(idx: int):
        canvas.fill(255)
        for j in range(idx):
            seg = sorted_lines[j]
            p1 = (int(round(seg.x1 * scale)), int(round(seg.y1 * scale)))
            p2 = (int(round(seg.x2 * scale)), int(round(seg.y2 * scale)))
            cv2.line(canvas, p1, p2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    while True:
        if width_changed:
            redraw_canvas_to(current_idx)
            width_changed = False
            print(f"Stroke width regulated to: {current_stroke_width:.1f}px (thickness: {line_thickness})")

        if not paused and current_idx < total_lines:
            end_batch = min(total_lines, current_idx + speed)
            for i in range(current_idx, end_batch):
                seg = sorted_lines[i]
                pt1 = (int(round(seg.x1 * scale)), int(round(seg.y1 * scale)))
                pt2 = (int(round(seg.x2 * scale)), int(round(seg.y2 * scale)))

                cv2.line(
                    canvas,
                    pt1,
                    pt2,
                    color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )
                last_pt = pt2

            current_idx = end_batch

        # Create presentation frame with pen indicator and HUD
        display_frame = canvas.copy()

        # Draw animated pen tip highlight
        if last_pt is not None and current_idx < total_lines and not paused:
            cv2.circle(display_frame, last_pt, 4, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(display_frame, last_pt, 7, (0, 180, 255), 1, cv2.LINE_AA)

        # Status text overlay banner at bottom
        progress_pct = (current_idx / float(total_lines)) * 100.0 if total_lines > 0 else 100.0
        status_text = (
            f"Line: {current_idx:,}/{total_lines:,} ({progress_pct:.1f}%) | "
            f"Speed: {speed} | Width: {current_stroke_width:.1f}px (W/T) {'[PAUSED]' if paused else ''}"
        )
        if current_idx >= total_lines:
            status_text += " [COMPLETED - Press R to replay or S to save]"

        cv2.rectangle(display_frame, (0, disp_h - 26), (disp_w, disp_h), (245, 245, 245), -1)
        cv2.line(display_frame, (0, disp_h - 26), (disp_w, disp_h - 26), (210, 210, 210), 1)
        cv2.putText(
            display_frame,
            status_text,
            (10, disp_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display_frame)

        # Wait key for animation timing & user control
        key = cv2.waitKey(10 if not paused else 50) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif key in (ord("+"), ord("=")):
            speed = min(500, int(speed * 1.5) + 1)
            print(f"Speed increased to: {speed} lines/frame")
        elif key in (ord("-"), ord("_")):
            speed = max(1, int(speed / 1.5))
            print(f"Speed decreased to: {speed} lines/frame")
        elif key in (ord("w"), ord("W"), ord("[")):
            # Regulate stroke width thinner
            current_stroke_width = max(0.3, round(current_stroke_width - 0.2, 1))
            line_thickness = max(1, int(round(current_stroke_width * scale)))
            width_changed = True
            try:
                cv2.setTrackbarPos("Stroke Width (x0.1px)", window_name, int(round(current_stroke_width * 10)))
            except Exception:
                pass
        elif key in (ord("t"), ord("T"), ord("]")):
            # Regulate stroke width thicker
            current_stroke_width = min(3.0, round(current_stroke_width + 0.2, 1))
            line_thickness = max(1, int(round(current_stroke_width * scale)))
            width_changed = True
            try:
                cv2.setTrackbarPos("Stroke Width (x0.1px)", window_name, int(round(current_stroke_width * 10)))
            except Exception:
                pass
        elif key in (ord("r"), ord("R")):
            canvas.fill(255)
            current_idx = 0
            paused = False
            last_pt = None
            print("Restarting animation...")
        elif key in (ord("s"), ord("S")):
            save_path = "animated_snapshot.png"
            cv2.imwrite(save_path, canvas)
            print(f"[Saved] Snapshot saved to {save_path}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live animation of mathematical line drawing.")
    parser.add_argument("--input", "-i", type=str, default="samples/sample_portrait.png", help="Input image path")
    parser.add_argument("--style", type=str, default="adaptive", choices=["adaptive", "engraving", "contour", "crosshatch", "minimalist"])
    parser.add_argument("--line-width", "-w", type=float, default=0.8, help="Line stroke width in pixels (e.g. 0.6 - 1.2)")
    parser.add_argument("--speed", type=int, default=3, help="Number of lines placed per animation frame")
    parser.add_argument("--max-dimension", type=int, default=700, help="Max working size")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return 1

    config = LineArtConfig(
        max_dimension=args.max_dimension,
        style=args.style,
        line_width=args.line_width,
        optimization_iterations=1,
    )


    print(f"Loading {input_path} and computing mathematical line fields...")
    _, intensity = load_image(input_path, max_dimension=config.max_dimension)
    height, width = intensity.shape

    darkness = compute_darkness(intensity)
    smoothed = apply_gaussian_blur(intensity, config.gaussian_sigma)
    gx, gy, _ = compute_gradients(smoothed)
    contour, coherence = compute_structure_tensor(intensity)
    edge_map = detect_edges(intensity, sensitivity=config.edge_sensitivity)

    lines = generate_line_art(darkness, contour, coherence, edge_map, gx, gy, config)
    lines, _ = refine_line_arrangements(lines, intensity, darkness, edge_map, contour, config)

    animate_lines(lines, width, height, config, lines_per_frame=args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
