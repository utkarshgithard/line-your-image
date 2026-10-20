"""
Interactive Web Studio for Mathematical Line Art Reconstruction.
Provides real-time drag-and-drop image upload, sample gallery, and streams
line coordinates to an interactive HTML5 Canvas animator that draws lines one-by-one.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
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
from renderer import render_lines

app = Flask(__name__)

# Sample benchmarks
SAMPLES_DIR = Path("samples")


def serialize_image_to_base64(image_rgb: np.ndarray, format: str = "JPEG") -> str:
    """Helper to convert uint8 RGB image to base64 data URI."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return available pre-generated sample benchmarks."""
    sample_list = [
        {
            "id": "portrait",
            "title": "Classical Portrait",
            "category": "Organic Contours",
            "path": "samples/sample_portrait.png",
            "suggested_style": "adaptive",
            "gamma": 1.2,
        },
        {
            "id": "landscape",
            "title": "Mountain Sunset",
            "category": "Tonal Gradients",
            "path": "samples/sample_landscape.png",
            "suggested_style": "engraving",
            "gamma": 1.3,
        },
        {
            "id": "logo",
            "title": "Emblem Logo",
            "category": "High-Contrast Geometry",
            "path": "samples/sample_logo.png",
            "suggested_style": "crosshatch",
            "gamma": 1.0,
        },
    ]
    return jsonify(sample_list)


@app.route("/api/process", methods=["POST"])
def process_image():
    """
    Process an uploaded image or sample selection:
    1. Runs mathematical analysis (gradients, structure tensor, Canny edges).
    2. Generates adaptive straight line segments.
    3. Returns line coordinates for client-side live stroke-by-stroke animation.
    """
    start_time = time.perf_counter()

    # Determine input source
    if "file" in request.files and request.files["file"].filename != "":
        uploaded_file = request.files["file"]
        try:
            pil_img = Image.open(uploaded_file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {e}"}), 400
    elif "sample_id" in request.form:
        sample_id = request.form.get("sample_id")
        sample_path = SAMPLES_DIR / f"sample_{sample_id}.png"
        if not sample_path.exists():
            return jsonify({"error": f"Sample not found: {sample_id}"}), 404
        pil_img = Image.open(sample_path).convert("RGB")
    else:
        return jsonify({"error": "No image or sample provided."}), 400

    # Parse parameters
    max_dimension = int(request.form.get("max_dimension", 650))
    style = request.form.get("style", "adaptive")
    gamma = float(request.form.get("gamma", 1.2))
    rho_min = float(request.form.get("rho_min", 0.0))
    rho_max = float(request.form.get("rho_max", 0.08))
    base_spacing = float(request.form.get("base_spacing", 12.0))
    edge_sensitivity = float(request.form.get("edge_sensitivity", 0.4))
    line_width = float(request.form.get("line_width", 1.1))
    iterations = int(request.form.get("iterations", 1))
    line_budget = int(request.form.get("line_budget", 2000))
    line_penalty = float(request.form.get("line_penalty", 0.08))

    # Resize while preserving aspect ratio
    w, h = pil_img.size
    if max_dimension > 0 and max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    rgb_array = np.array(pil_img, dtype=np.uint8)
    import cv2

    gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    intensity = gray_array.astype(np.float32) / 255.0
    height, width = intensity.shape

    enable_face_detection = request.form.get("face_detection", "true").lower() in ("true", "1", "yes")

    config = LineArtConfig(
        max_dimension=max_dimension,
        style=style,
        gamma=gamma,
        rho_min=rho_min,
        rho_max=rho_max,
        base_spacing=base_spacing,
        edge_sensitivity=edge_sensitivity,
        line_width=line_width,
        line_budget=line_budget,
        line_penalty=line_penalty,
        optimization_iterations=iterations,
        random_seed=42,
        enable_face_detection=enable_face_detection,
    )

    # Face detection
    face_data = None
    if config.enable_face_detection:
        try:
            from face_processing import detect_face_features

            face_data = detect_face_features(rgb_array, config)
        except Exception:
            face_data = None

    # Mathematical Analysis
    darkness = compute_darkness(intensity)
    smoothed = apply_gaussian_blur(intensity, config.gaussian_sigma)
    gx, gy, grad_mag = compute_gradients(smoothed)
    contour_angle, coherence = compute_structure_tensor(intensity)
    edge_map = detect_edges(intensity, sensitivity=config.edge_sensitivity)

    # Line generation
    lines = generate_line_art(
        darkness,
        contour_angle,
        coherence,
        edge_map,
        gx,
        gy,
        config,
        rgb_image=rgb_array,
        face_data=face_data,
    )

    # Optional optimization
    if iterations > 0:
        lines, _ = refine_line_arrangements(
            lines, intensity, darkness, edge_map, contour_angle, config
        )

    # Progressive sorting for stroke-by-stroke drawing
    # Order: Primary contours/hatching -> secondary crosshatch -> deep shadows -> face feature chords -> edge lines
    def stroke_sort_key(line: LineSegment):
        # family_id: 0 (light tone), 1 (mid), 2 (shadow), 3 (deep), 98 (face landmarks), 99 (edge details)
        return (line.family_id, -line.length)

    lines.sort(key=stroke_sort_key)

    # Serialize lines into lightweight array [x1, y1, x2, y2, family, length]
    serialized_lines = [
        [
            round(l.x1, 1),
            round(l.y1, 1),
            round(l.x2, 1),
            round(l.y2, 1),
            l.family_id,
            round(l.length, 1),
        ]
        for l in lines
    ]

    calc_time = round(time.perf_counter() - start_time, 3)

    # Generate previews
    orig_preview = serialize_image_to_base64(rgb_array, format="JPEG")
    gray_preview = serialize_image_to_base64(gray_array, format="JPEG")

    face_meta = {
        "detected": bool(face_data and face_data.has_face),
        "num_faces": face_data.num_faces if face_data else 0,
        "summary": face_data.feature_summary if face_data else {},
    }

    return jsonify(
        {
            "success": True,
            "width": width,
            "height": height,
            "total_lines": len(serialized_lines),
            "lines": serialized_lines,
            "metadata": {
                "computation_time_s": calc_time,
                "mean_darkness": round(float(np.mean(darkness)), 3),
                "edge_coverage_pct": round(float(np.mean(edge_map > 0.5) * 100.0), 1),
                "style": style,
                "face_aware": face_meta,
            },
            "preview_original": orig_preview,
            "preview_grayscale": gray_preview,
        }
    )



@app.route("/api/export", methods=["POST"])
def export_high_res():
    """Renders and returns a high-resolution supersampled PNG image."""
    data = request.get_json(force=True)
    lines_raw = data.get("lines", [])
    width = int(data.get("width", 600))
    height = int(data.get("height", 700))
    scale = float(data.get("scale", 2.0))
    line_width = float(data.get("line_width", 1.0))
    supersampling = int(data.get("supersampling", 2))

    config = LineArtConfig(
        line_width=line_width,
        supersampling=supersampling,
    )

    line_segments = [
        LineSegment(x1=r[0], y1=r[1], x2=r[2], y2=r[3], family_id=r[4])
        for r in lines_raw
    ]

    rendered_rgb, _ = render_lines(
        lines=line_segments,
        width=width,
        height=height,
        config=config,
        scale=scale,
    )

    buf = io.BytesIO()
    pil_img = Image.fromarray(rendered_rgb)
    pil_img.save(buf, format="PNG", compress_level=6)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="mathematical_line_art.png",
    )


if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  MATHEMATICAL LINE ART - INTERACTIVE WEB STUDIO")
    print("=" * 64)
    print("  Local Studio URL: http://127.0.0.1:5000")
    print("  Open this URL in your web browser to upload images and watch")
    print("  the mathematical function draw lines one by one on the sheet!")
    print("=" * 64 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
