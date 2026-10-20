"""
Sample generator script.
Creates benchmark images and runs line art reconstruction in multiple styles,
saving output line drawings and comparison previews.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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


def create_sample_images(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classical Portrait Silhouette / Stylized Face
    portrait_path = output_dir / "sample_portrait.png"
    p_img = Image.new("RGB", (600, 700), color=(245, 240, 235))
    draw_p = ImageDraw.Draw(p_img)
    # Head and neck
    draw_p.rectangle([250, 480, 350, 620], fill=(210, 180, 160))
    draw_p.ellipse([150, 120, 450, 520], fill=(225, 200, 180))
    # Hair volume
    draw_p.chord([120, 80, 480, 380], start=170, end=370, fill=(30, 25, 25))
    draw_p.polygon([(120, 250), (100, 520), (160, 500), (180, 320)], fill=(30, 25, 25))
    draw_p.polygon([(480, 250), (500, 520), (440, 500), (420, 320)], fill=(30, 25, 25))
    # Eyebrows & Eyes
    draw_p.arc([200, 250, 260, 275], start=190, end=350, fill=(40, 30, 30), width=4)
    draw_p.arc([340, 250, 400, 275], start=190, end=350, fill=(40, 30, 30), width=4)
    draw_p.ellipse([215, 275, 245, 295], fill=(25, 20, 20))
    draw_p.ellipse([355, 275, 385, 295], fill=(25, 20, 20))
    # Nose & Lips
    draw_p.line([(300, 280), (290, 360), (315, 365)], fill=(120, 90, 80), width=4)
    draw_p.chord([260, 400, 340, 435], start=0, end=180, fill=(160, 70, 70))
    draw_p.arc([260, 395, 340, 435], start=180, end=360, fill=(130, 50, 50), width=3)
    p_img.save(portrait_path)

    # 2. Dramatic Landscape (Mountains, Sunset Gradient Sky, and Lake)
    landscape_path = output_dir / "sample_landscape.png"
    land_arr = np.zeros((500, 700, 3), dtype=np.uint8)
    for y in range(300):
        # Gradient sunset sky
        t = y / 300.0
        r = int(240 * (1.0 - t) + 255 * t)
        g = int(140 * (1.0 - t) + 210 * t)
        b = int(60 * (1.0 - t) + 120 * t)
        land_arr[y, :] = [r, g, b]
    # Lake in foreground
    for y in range(300, 500):
        t = (y - 300) / 200.0
        land_arr[y, :] = [int(40 + 30 * t), int(60 + 40 * t), int(90 + 50 * t)]

    land_pil = Image.fromarray(land_arr)
    draw_l = ImageDraw.Draw(land_pil)
    # Sun disk
    draw_l.ellipse([300, 160, 400, 260], fill=(255, 245, 210))
    # Layered Mountains
    draw_l.polygon([(0, 320), (140, 160), (280, 320)], fill=(70, 60, 80))
    draw_l.polygon([(200, 320), (380, 130), (520, 320)], fill=(50, 45, 65))
    draw_l.polygon([(420, 320), (580, 170), (700, 320)], fill=(40, 35, 55))
    # Forefront ridge
    draw_l.polygon([(50, 350), (260, 250), (450, 350)], fill=(30, 30, 40))
    land_pil.save(landscape_path)

    # 3. Geometric High-Contrast Emblem Logo
    logo_path = output_dir / "sample_logo.png"
    logo = Image.new("RGB", (600, 600), color=(255, 255, 255))
    draw_logo = ImageDraw.Draw(logo)
    # Outer dark hexagon / circle
    draw_logo.ellipse([50, 50, 550, 550], fill=(15, 20, 30))
    # Inner white star / geometric ring
    draw_logo.ellipse([100, 100, 500, 500], fill=(255, 255, 255))
    draw_logo.ellipse([140, 140, 460, 460], fill=(15, 20, 30))
    # Center white polygon
    draw_logo.polygon(
        [(300, 180), (420, 300), (370, 420), (230, 420), (180, 300)],
        fill=(255, 255, 255),
    )
    draw_logo.ellipse([260, 260, 340, 340], fill=(15, 20, 30))
    logo.save(logo_path)

    return {
        "portrait": portrait_path,
        "landscape": landscape_path,
        "logo": logo_path,
    }


def run_sample_reconstructions():
    output_dir = Path("samples")
    print(f"Creating sample images in {output_dir.resolve()}...")
    images = create_sample_images(output_dir)

    demos = [
        ("portrait", images["portrait"], "adaptive", 1.2),
        ("landscape", images["landscape"], "engraving", 1.3),
        ("logo", images["logo"], "crosshatch", 1.0),
    ]

    for name, img_path, style, gamma in demos:
        print(f"\nProcessing {name} in '{style}' style...")
        config = LineArtConfig(
            max_dimension=600,
            style=style,
            gamma=gamma,
            optimization_iterations=2,
            supersampling=2,
            random_seed=42,
        )

        rgb_img, intensity = load_image(img_path, max_dimension=config.max_dimension)
        height, width = intensity.shape
        darkness = compute_darkness(intensity)
        smoothed = apply_gaussian_blur(intensity, config.gaussian_sigma)
        gx, gy, _ = compute_gradients(smoothed)
        contour, coherence = compute_structure_tensor(intensity)
        edge_map = detect_edges(intensity, sensitivity=config.edge_sensitivity)

        lines = generate_line_art(darkness, contour, coherence, edge_map, gx, gy, config)
        lines, history = refine_line_arrangements(
            lines, intensity, darkness, edge_map, contour, config
        )

        final_rgb, _ = render_lines(lines, width, height, config)

        out_line_path = output_dir / f"{name}_lineart_{style}.png"
        out_comp_path = output_dir / f"{name}_comparison_{style}.png"

        save_image(final_rgb, out_line_path)
        show_or_save_preview(
            rgb_img,
            intensity,
            final_rgb,
            save_path=out_comp_path,
            show_window=False,
        )
        print(f"  -> Saved line art: {out_line_path}")
        print(f"  -> Saved comparison: {out_comp_path} ({len(lines):,} lines)")


if __name__ == "__main__":
    run_sample_reconstructions()
