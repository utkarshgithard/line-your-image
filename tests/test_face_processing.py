"""
Unit tests for Face-Aware Line Art Processing.
Verifies:
1. Detect whether a face exists (positive on portrait, negative on landscape).
2. Locate facial features (eyes, eyebrows, nose, lips, jawline).
3. Allocate higher reconstruction priority via salience map and contour angle guidance.
4. End-to-end parsimonious selection prioritizing facial anatomical chords.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from config import LineArtConfig
from face_processing import detect_face_features, FaceAwareData
from image_processing import detect_face_and_features
from line_generation import generate_line_art
from sparse_line_selection import select_parsimonious_lines


SAMPLES_DIR = Path("samples")
PORTRAIT_PATH = SAMPLES_DIR / "sample_portrait.png"
LANDSCAPE_PATH = SAMPLES_DIR / "sample_landscape.png"


def test_face_detection_positive_on_portrait():
    """Verify that MediaPipe FaceLandmarker detects face on sample portrait."""
    assert PORTRAIT_PATH.exists(), "Portrait sample missing"
    bgr = cv2.imread(str(PORTRAIT_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = LineArtConfig(enable_face_detection=True)
    face_data = detect_face_features(rgb, cfg)

    assert isinstance(face_data, FaceAwareData)
    assert face_data.has_face is True
    assert face_data.num_faces >= 1
    assert len(face_data.faces) >= 1


def test_face_detection_negative_on_landscape():
    """Verify that no face is detected on landscape, running normally without modifications."""
    assert LANDSCAPE_PATH.exists(), "Landscape sample missing"
    bgr = cv2.imread(str(LANDSCAPE_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = LineArtConfig(enable_face_detection=True)
    face_data = detect_face_features(rgb, cfg)

    assert face_data.has_face is False
    assert face_data.num_faces == 0
    assert len(face_data.faces) == 0
    assert len(face_data.feature_chords) == 0
    # Salience map should remain neutral 1.0 everywhere
    assert np.all(face_data.salience_map == 1.0)


def test_locate_facial_features():
    """Verify that eyes, eyebrows, nose, lips, and jawline are individually localized."""
    bgr = cv2.imread(str(PORTRAIT_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = LineArtConfig(enable_face_detection=True)
    face_data = detect_face_features(rgb, cfg)
    assert face_data.has_face is True

    face = face_data.faces[0]
    required_features = ["eyes", "eyebrows", "nose", "lips", "jawline"]
    for feat_name in required_features:
        assert feat_name in face.features, f"Missing feature: {feat_name}"
        feat = face.features[feat_name]
        assert len(feat.points) > 0, f"No points for feature: {feat_name}"
        assert len(feat.segments) > 0, f"No segments for feature: {feat_name}"
        assert len(feat.chords) > 0, f"No chords generated for feature: {feat_name}"


def test_salience_map_allocates_higher_priority_to_face():
    """Verify that salience map has elevated weights on facial features."""
    bgr = cv2.imread(str(PORTRAIT_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = LineArtConfig(enable_face_detection=True, face_salience_weight=3.5)
    face_data = detect_face_features(rgb, cfg)

    salience = face_data.salience_map
    assert salience.shape == rgb.shape[:2]
    # Max salience should reach the boosted feature weight
    assert salience.max() >= 3.0
    # Min salience in non-face regions should be 1.0
    assert salience.min() == 1.0
    # Mean salience on portrait should be significantly above baseline
    assert salience.mean() > 1.05


def test_guided_contour_angles_follow_facial_curves():
    """Verify that guided contour angles are computed along facial feature landmarks."""
    bgr = cv2.imread(str(PORTRAIT_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = LineArtConfig(enable_face_detection=True)
    face_data = detect_face_features(rgb, cfg)

    assert face_data.guided_contour_angles is not None
    valid_angles = face_data.guided_contour_angles[~np.isnan(face_data.guided_contour_angles)]
    assert len(valid_angles) > 100
    # Angles should be bounded in [-pi/2, pi/2]
    assert np.all(valid_angles >= -np.pi / 2.0 - 1e-4)
    assert np.all(valid_angles <= np.pi / 2.0 + 1e-4)


def test_parsimonious_selection_with_face_awareness():
    """Verify that sparse line selection utilizes face_data and prioritizes face details."""
    bgr = cv2.imread(str(PORTRAIT_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    darkness = 1.0 - gray
    edge_map = np.zeros((h, w), dtype=np.float32)
    contour_angle = np.zeros((h, w), dtype=np.float32)
    coherence = np.ones((h, w), dtype=np.float32)

    cfg = LineArtConfig(
        style="minimalist",
        line_budget=40,
        line_penalty=0.02,
        enable_face_detection=True,
    )

    face_data = detect_face_features(rgb, cfg)
    assert face_data.has_face is True

    lines = select_parsimonious_lines(
        darkness=darkness,
        edge_map=edge_map,
        contour_angle=contour_angle,
        coherence=coherence,
        config=cfg,
        line_budget=40,
        line_penalty=0.02,
        face_data=face_data,
        rgb_image=rgb,
    )

    assert len(lines) > 0
    assert len(lines) <= 40

    # Verify that facial feature chords (family_id 98) or lines inside face bbox are selected
    face_bbox = face_data.faces[0].bbox
    bx1, by1, bx2, by2 = face_bbox
    lines_in_face = [
        l for l in lines
        if bx1 <= l.midpoint[0] <= bx2 and by1 <= l.midpoint[1] <= by2
    ]
    assert len(lines_in_face) > 0, "No lines placed in facial region!"
