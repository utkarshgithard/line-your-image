"""
End-to-End integration tests for all image categories:
1. Portrait / stylized human face
2. Landscape with mountain & tonal gradient sky
3. High-contrast geometric logo
4. Image with extensive pure white areas (testing rho_min=0 empty space)
"""

import numpy as np
import pytest
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
from renderer import render_lines, save_image


@pytest.fixture
def test_images(tmp_path):
    """Generate representative benchmark images for testing."""
    img_dir = tmp_path / "benchmarks"
    img_dir.mkdir()

    # 1. High contrast logo
    logo_path = img_dir / "logo.png"
    logo = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(logo)
    draw.rectangle([40, 40, 160, 160], fill=(20, 20, 20))
    draw.ellipse([70, 70, 130, 130], fill=(255, 255, 255))
    logo.save(logo_path)

    # 2. Large white areas (minimalist icon with mostly empty canvas)
    white_path = img_dir / "white_areas.png"
    white_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw_w = ImageDraw.Draw(white_img)
    draw_w.rectangle([95, 95, 105, 105], fill=(10, 10, 10))  # tiny central box
    white_img.save(white_path)

    # 3. Landscape (gradient sky + dark mountain triangles)
    landscape_path = img_dir / "landscape.png"
    landscape = np.zeros((200, 250, 3), dtype=np.uint8)
    # Gradient sky: top light blue, horizon soft white
    for y in range(120):
        val = int(180 + (75 * (y / 120.0)))
        landscape[y, :, :] = [val - 30, val - 10, val]
    # Ground: darker
    landscape[120:, :, :] = [80, 100, 70]
    land_pil = Image.fromarray(landscape)
    draw_l = ImageDraw.Draw(land_pil)
    # Mountains
    draw_l.polygon([(20, 120), (80, 40), (140, 120)], fill=(40, 40, 50))
    draw_l.polygon([(110, 120), (170, 60), (230, 120)], fill=(30, 30, 40))
    land_pil.save(landscape_path)

    # 4. Portrait / stylized face
    portrait_path = img_dir / "portrait.png"
    portrait = Image.new("RGB", (200, 240), color=(230, 220, 210))
    draw_p = ImageDraw.Draw(portrait)
    # Head contour
    draw_p.ellipse([50, 40, 150, 180], fill=(200, 180, 160))
    # Hair
    draw_p.chord([40, 30, 160, 120], start=180, end=360, fill=(30, 20, 15))
    # Eyes
    draw_p.ellipse([70, 95, 85, 105], fill=(20, 20, 20))
    draw_p.ellipse([115, 95, 130, 105], fill=(20, 20, 20))
    # Mouth
    draw_p.arc([80, 140, 120, 155], start=0, end=180, fill=(40, 20, 20), width=3)
    portrait.save(portrait_path)

    return {
        "logo": logo_path,
        "white_areas": white_path,
        "landscape": landscape_path,
        "portrait": portrait_path,
    }


def run_pipeline(image_path, config: LineArtConfig):
    rgb_img, intensity = load_image(image_path, max_dimension=config.max_dimension)
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
    rendered_rgb, rendered_gray = render_lines(lines, width, height, config)
    return rendered_rgb, rendered_gray, lines, (width, height)


def test_logo_reconstruction(test_images, tmp_path):
    config = LineArtConfig(max_dimension=150, optimization_iterations=1, style="crosshatch")
    rgb, gray, lines, dims = run_pipeline(test_images["logo"], config)

    assert len(lines) > 20
    assert rgb.shape == (dims[1], dims[0], 3)
    # Outside background should be white
    assert rgb[10, 10, 0] > 240
    # Center circle should be predominantly bright
    center_y, center_x = dims[1] // 2, dims[0] // 2
    assert rgb[center_y, center_x, 0] > 200


def test_large_white_areas_remain_sparse(test_images):
    # Testing that rho_min=0 allows bright areas to stay clean
    config = LineArtConfig(
        max_dimension=150,
        rho_min=0.0,
        gamma=1.5,
        optimization_iterations=0,
    )
    rgb, gray, lines, dims = run_pipeline(test_images["white_areas"], config)

    # Vast majority of canvas must remain pure white
    white_ratio = float(np.mean(gray > 0.98))
    assert white_ratio > 0.92, f"Expected >92% white pixels, got {white_ratio*100:.1f}%"


def test_landscape_reconstruction(test_images):
    config = LineArtConfig(max_dimension=150, optimization_iterations=1, style="engraving")
    rgb, gray, lines, dims = run_pipeline(test_images["landscape"], config)

    assert len(lines) > 50
    # Sky (top) should have significantly fewer lines than mountains (middle)
    top_half_density = np.mean(1.0 - gray[:60, :])
    mid_half_density = np.mean(1.0 - gray[60:120, :])
    assert mid_half_density > top_half_density


def test_portrait_reconstruction(test_images):
    config = LineArtConfig(max_dimension=150, optimization_iterations=1, style="adaptive")
    rgb, gray, lines, dims = run_pipeline(test_images["portrait"], config)

    assert len(lines) > 30
    # Eyes and hair regions should have dense marks
    hair_density = np.mean(1.0 - gray[30:60, 60:140])
    assert hair_density > 0.15
