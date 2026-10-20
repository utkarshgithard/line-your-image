"""
Main entry point and CLI orchestration for Mathematical Line Art Reconstruction.

Example usage:
    python main.py --input image.jpg --output result.png
    python main.py --input portrait.png --style engraving --preview
    python main.py --picker --output drawing.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import LineArtConfig
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
from renderer import render_lines, save_image, show_or_save_preview


def open_file_picker() -> Optional[str]:
    """Launch a native desktop file picker dialog to select an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Input Image for Line Art Reconstruction",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[Warning] GUI File picker not available: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct an image using mathematical thin straight line art.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # File I/O
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input image file (JPG, PNG, WEBP, etc.)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_lineart.png",
        help="Path to save output line art PNG",
    )
    parser.add_argument(
        "--picker",
        action="store_true",
        help="Open a desktop file picker dialog to choose input image",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional path to a JSON configuration file",
    )

    # Resolution & Preprocessing
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=800,
        help="Maximum working dimension (preserves aspect ratio)",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Resolution multiplier for exported image (e.g. 2.0 for double resolution)",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.2,
        help="Smoothing strength to reduce noise before gradient analysis",
    )

    # Mathematical Density & Spacing
    parser.add_argument(
        "--rho-min",
        type=float,
        default=0.0,
        help="Minimum line density in bright areas (0 allows pure white regions)",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=0.08,
        help="Maximum line density in dark areas",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.2,
        help="Darkness response exponent (D^gamma)",
    )
    parser.add_argument(
        "--base-spacing",
        type=float,
        default=12.0,
        help="Base line spacing scale in pixels",
    )

    # Line Geometry
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.0,
        help="Line thickness in pixels at working resolution",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=8.0,
        help="Minimum line segment length in pixels",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=28.0,
        help="Maximum line segment length in pixels",
    )

    # Style & Orientation
    parser.add_argument(
        "--style",
        type=str,
        choices=["adaptive", "engraving", "contour", "crosshatch", "minimalist"],
        default="adaptive",
        help="Artistic rendering style ('minimalist' uses the 6-step parsimonious pursuit algorithm)",
    )
    parser.add_argument(
        "--line-budget",
        type=int,
        default=2000,
        help="Maximum line budget for minimalist / parsimonious reconstruction (1000 - 3000 lines)",
    )
    parser.add_argument(
        "--line-penalty",
        type=float,
        default=0.08,
        help="L0 penalty per line discouraging superfluous strokes in minimalist mode",
    )
    parser.add_argument(
        "--orientation-angles",
        type=float,
        nargs="+",
        default=[45.0, 135.0, 0.0, 90.0],
        help="Fixed crosshatching angles in degrees",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=4,
        help="Maximum number of active crosshatching families in deep shadows",
    )

    # Edge Detection
    parser.add_argument(
        "--edge-sensitivity",
        type=float,
        default=0.4,
        help="Sensitivity for edge detection (0.0 to 1.0)",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.35,
        help="Strength of edge alignment lines",
    )

    # Optimization & Quality
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterative optimization and refinement passes (0 to skip)",
    )
    parser.add_argument(
        "--supersampling",
        type=int,
        default=2,
        help="Supersampling factor for anti-aliasing (1, 2, 3, or 4)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic output",
    )

    # Face Detection & Landmark Guidance
    parser.add_argument(
        "--disable-face-detection",
        action="store_true",
        help="Disable automatic MediaPipe face and facial landmark detection",
    )
    parser.add_argument(
        "--face-salience",
        type=float,
        default=3.0,
        help="Salience multiplier for eyes, eyebrows, nose, lips, and jawline when a face is detected",
    )

    # Preview & Comparison
    parser.add_argument(
        "--preview",
        "--show",
        action="store_true",
        help="Display side-by-side Matplotlib interactive comparison window",
    )
    parser.add_argument(
        "--save-comparison",
        type=str,
        default=None,
        help="Path to save side-by-side comparison image (e.g. comparison.png)",
    )


    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> LineArtConfig:
    """Build and validate LineArtConfig from CLI arguments or JSON file."""
    if args.config_file:
        config = LineArtConfig.load_json(args.config_file)
        return config

    config = LineArtConfig(
        max_dimension=args.max_dimension,
        output_scale=args.output_scale,
        gaussian_sigma=args.gaussian_sigma,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        gamma=args.gamma,
        base_spacing=args.base_spacing,
        line_width=args.line_width,
        min_length=args.min_length,
        max_length=args.max_length,
        style=args.style,
        orientation_angles=tuple(args.orientation_angles),
        num_families=args.num_families,
        edge_sensitivity=args.edge_sensitivity,
        edge_weight=args.edge_weight,
        line_budget=args.line_budget,
        line_penalty=args.line_penalty,
        optimization_iterations=args.iterations,
        supersampling=args.supersampling,
        random_seed=args.random_seed,
        enable_face_detection=not args.disable_face_detection,
        face_salience_weight=args.face_salience,
    )
    config.validate()
    return config


def main() -> int:
    args = parse_args()

    # Determine input path
    input_path = args.input
    if args.picker or input_path is None:
        print("[Info] Opening file picker...")
        picked = open_file_picker()
        if picked:
            input_path = picked
        elif input_path is None:
            print("[Error] No input image provided. Specify --input <path> or use --picker.")
            return 1

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[Error] File not found: {input_file}")
        return 1

    config = build_config_from_args(args)

    print("=" * 64)
    print("  MATHEMATICAL LINE ART RECONSTRUCTION")
    print("=" * 64)
    print(f"Input:         {input_file}")
    print(f"Output:        {args.output}")
    print(f"Style:         {config.style}")
    print(f"Max Dimension: {config.max_dimension} px")
    print(f"Supersampling: {config.supersampling}x")
    print(f"Face-Aware:    {'Enabled' if config.enable_face_detection else 'Disabled'}")
    print(f"Random Seed:   {config.random_seed}")
    print("-" * 64)

    start_time = time.perf_counter()

    # 1. Load and Preprocess Image
    print("[1/5] Loading and analyzing image...")
    rgb_img, intensity = load_image(input_file, max_dimension=config.max_dimension)
    height, width = intensity.shape
    print(f"      Working resolution: {width} x {height} px")

    # 2. Mathematical Analysis
    print("[2/5] Computing darkness, Sobel gradients, structure tensor, and edges...")
    darkness = compute_darkness(intensity)
    smoothed_intensity = apply_gaussian_blur(intensity, config.gaussian_sigma)
    grad_x, grad_y, grad_mag = compute_gradients(smoothed_intensity)
    contour_angle, coherence = compute_structure_tensor(intensity)
    edge_map = detect_edges(intensity, sensitivity=config.edge_sensitivity)

    mean_darkness = float(np.mean(darkness))
    edge_coverage = float(np.mean(edge_map > 0.5) * 100.0)
    print(f"      Mean image darkness: {mean_darkness:.3f} | Edge pixels: {edge_coverage:.1f}%")

    # Check for face detection
    if config.enable_face_detection:
        try:
            from face_processing import detect_face_features

            face_data = detect_face_features(rgb_img, config)
            if face_data.has_face:
                print(
                    f"      [Face-Aware] Detected {face_data.num_faces} face(s)! "
                    f"Prioritizing eyes, eyebrows, nose, lips, and jawline ({len(face_data.feature_chords)} chords)."
                )
            else:
                print("      [Face-Aware] No face detected. Running standard image reconstruction.")
        except Exception:
            face_data = None
    else:
        face_data = None

    # 3. Line Generation
    print("[3/5] Generating adaptive straight line segments...")
    t_gen_start = time.perf_counter()
    lines = generate_line_art(
        darkness=darkness,
        contour_angle=contour_angle,
        coherence=coherence,
        edge_map=edge_map,
        grad_x=grad_x,
        grad_y=grad_y,
        config=config,
        rgb_image=rgb_img,
        face_data=face_data,
    )

    t_gen = time.perf_counter() - t_gen_start
    print(f"      Initial lines generated: {len(lines):,} segments in {t_gen:.2f}s")

    # 4. Iterative Optimization
    if config.optimization_iterations > 0:
        print(f"[4/5] Running iterative refinement ({config.optimization_iterations} iterations)...")

        def on_step(step: int, total_steps: int, stats: dict[str, float]):
            loss_val = stats.get("total_loss", 0.0)
            ssim_val = stats.get("ssim", 0.0)
            count = int(stats.get("line_count", len(lines)))
            print(
                f"      Iter {step}/{total_steps}: Loss = {loss_val:.5f} | "
                f"SSIM = {ssim_val:.4f} | Lines = {count:,}"
            )

        lines, history = refine_line_arrangements(
            initial_lines=lines,
            target_intensity=intensity,
            darkness=darkness,
            edge_map=edge_map,
            contour_angle=contour_angle,
            config=config,
            progress_callback=on_step,
        )
    else:
        print("[4/5] Optimization skipped (--iterations 0).")

    # 5. Final High-Resolution Rendering
    print(f"[5/5] Rendering final linework (Scale: {config.output_scale}x, Supersample: {config.supersampling}x)...")
    final_rgb, _ = render_lines(
        lines=lines,
        width=width,
        height=height,
        config=config,
        scale=config.output_scale,
    )

    # Save output line art
    save_image(final_rgb, args.output)
    print(f"[Success] Saved line art to: {args.output}")

    # Save or show preview
    comparison_path = args.save_comparison
    if args.preview and comparison_path is None:
        comparison_path = Path(args.output).with_name(f"{Path(args.output).stem}_comparison.png")

    if comparison_path is not None or args.preview:
        print(f"[Preview] Generating side-by-side comparison figure...")
        show_or_save_preview(
            original_rgb=rgb_img,
            grayscale_intensity=intensity,
            line_art_rgb=final_rgb,
            save_path=comparison_path,
            show_window=args.preview,
        )
        if comparison_path is not None:
            print(f"[Success] Saved comparison to: {comparison_path}")

    elapsed = time.perf_counter() - start_time
    print("-" * 64)
    print(f"Total processing completed in {elapsed:.2f} seconds.")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
