"""
Face-Aware Line Art Processing Module using MediaPipe Face Landmarker.

Implements:
1. Face existence detection:
   - Detects whether faces exist in the input image.
   - If no face is detected, the pipeline runs normally.
   - If a face is detected, activates face-aware processing.
2. Facial feature localization:
   - Identifies detailed facial landmarks for eyes, eyebrows, nose, lips, and jawline.
3. Facial detail allocation:
   - Generates priority salience map (allocates more line budget to facial features).
   - Generates contour angles along landmark curves to guide line direction.
   - Generates anatomical candidate chords along facial curves for parsimonious pursuit.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import LineArtConfig, LineSegment

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL_PATH = Path("models") / "face_landmarker.task"

_CACHED_DETECTOR = None
_CACHED_MODEL_PATH = None


@dataclass
class FacialFeature:
    """Represents a localized facial feature with landmark points and chords."""

    name: str  # 'eyes', 'eyebrows', 'nose', 'lips', 'jawline'
    points: List[Tuple[float, float]]
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    chords: List[LineSegment] = field(default_factory=list)
    salience_boost: float = 3.0


@dataclass
class FaceInfo:
    """Extracted landmarks and feature groups for a single detected face."""

    bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    all_landmarks: List[Tuple[float, float]]
    features: Dict[str, FacialFeature]


@dataclass
class FaceAwareData:
    """Complete face-aware analysis results passed to line generation & selection."""

    has_face: bool
    num_faces: int
    faces: List[FaceInfo]
    salience_map: np.ndarray  # (H, W) float32 in [1.0, face_salience_weight]
    feature_edges: np.ndarray  # (H, W) float32 in [0.0, 1.0] edge map of landmarks
    guided_contour_angles: Optional[np.ndarray] = None  # (H, W) float32 tangent angles
    feature_chords: List[LineSegment] = field(default_factory=list)
    feature_summary: Dict[str, int] = field(default_factory=dict)


def ensure_face_landmarker_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Optional[Path]:
    """
    Ensure the MediaPipe face landmarker model file exists.
    If missing, attempts to download it from Google's official model repository.
    Returns Path if available, None if unavailable or offline.
    """
    p = Path(model_path)
    if p.exists() and p.stat().st_size > 100000:
        return p

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(p))
        if p.exists() and p.stat().st_size > 100000:
            return p
    except Exception:
        pass

    return p if p.exists() else None


def get_face_landmarker(model_path: str | Path = DEFAULT_MODEL_PATH):
    """
    Get or initialize a cached MediaPipe FaceLandmarker instance.
    Returns None if mediapipe is not installed or model is missing.
    """
    global _CACHED_DETECTOR, _CACHED_MODEL_PATH

    resolved_path = ensure_face_landmarker_model(model_path)
    if resolved_path is None:
        return None

    if _CACHED_DETECTOR is not None and _CACHED_MODEL_PATH == str(resolved_path):
        return _CACHED_DETECTOR

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(resolved_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=4,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        _CACHED_DETECTOR = detector
        _CACHED_MODEL_PATH = str(resolved_path)
        return _CACHED_DETECTOR
    except Exception:
        return None


def detect_face_features(
    rgb_image: np.ndarray,
    config: Optional[LineArtConfig] = None,
) -> FaceAwareData:
    """
    1. Detect whether a face exists in the input image.
    2. If no face is detected, returns FaceAwareData(has_face=False) allowing the
       existing algorithm to run normally.
    3. If a face is detected, activates face-aware processing:
       - Locates eyes, eyebrows, nose, lips, and jawline landmarks.
       - Constructs priority salience map.
       - Guides contour angles along facial curves.
       - Generates anatomical feature candidate chords.
    """
    h, w = rgb_image.shape[:2]
    cfg = config or LineArtConfig()

    # Default empty container (used when no face is found or detection is disabled)
    empty_result = FaceAwareData(
        has_face=False,
        num_faces=0,
        faces=[],
        salience_map=np.ones((h, w), dtype=np.float32),
        feature_edges=np.zeros((h, w), dtype=np.float32),
        guided_contour_angles=None,
        feature_chords=[],
        feature_summary={"faces": 0},
    )

    if not cfg.enable_face_detection:
        return empty_result

    detector = get_face_landmarker(cfg.face_model_path)
    if detector is None:
        return empty_result

    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

        # MediaPipe expects contiguous uint8 RGB
        c_rgb = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=c_rgb)
        detection_result = detector.detect(mp_image)
    except Exception:
        return empty_result

    if not detection_result or not detection_result.face_landmarks:
        # Step 1: No face detected -> return normally without modification
        return empty_result

    # Face detected! Activate face-aware processing
    c = FaceLandmarksConnections
    feature_connection_groups = {
        "eyes": list(c.FACE_LANDMARKS_LEFT_EYE) + list(c.FACE_LANDMARKS_RIGHT_EYE),
        "eyebrows": list(c.FACE_LANDMARKS_LEFT_EYEBROW) + list(c.FACE_LANDMARKS_RIGHT_EYEBROW),
        "nose": list(c.FACE_LANDMARKS_NOSE),
        "lips": list(c.FACE_LANDMARKS_LIPS),
        "jawline": list(c.FACE_LANDMARKS_FACE_OVAL),
    }

    faces_info: List[FaceInfo] = []
    salience_map = np.ones((h, w), dtype=np.float32)
    feature_edges = np.zeros((h, w), dtype=np.float32)
    guided_contour_angles = np.full((h, w), np.nan, dtype=np.float32)
    all_feature_chords: List[LineSegment] = []
    summary_counts: Dict[str, int] = {"faces": len(detection_result.face_landmarks)}

    boost_high = cfg.face_salience_weight  # default 3.0 for fine details
    boost_jaw = 1.0 + (boost_high - 1.0) * 0.6  # ~2.2 for jawline
    boost_face_region = 1.0 + (boost_high - 1.0) * 0.35  # ~1.7 for general face area

    for face_idx, landmarks in enumerate(detection_result.face_landmarks):
        all_pts = [(float(lm.x * w), float(lm.y * h)) for lm in landmarks]
        np_pts = np.array(all_pts, dtype=np.int32)

        # Bounding box of face
        min_x = int(np.clip(np.min(np_pts[:, 0]), 0, w - 1))
        max_x = int(np.clip(np.max(np_pts[:, 0]), 0, w - 1))
        min_y = int(np.clip(np.min(np_pts[:, 1]), 0, h - 1))
        max_y = int(np.clip(np.max(np_pts[:, 1]), 0, h - 1))

        # Fill general face convex hull with base facial priority
        hull = cv2.convexHull(np_pts)
        cv2.fillConvexPoly(salience_map, hull, float(boost_face_region))

        face_features: Dict[str, FacialFeature] = {}

        for feat_name, conns in feature_connection_groups.items():
            feat_pts_set = set()
            feat_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            feat_mask = np.zeros((h, w), dtype=np.uint8)
            feat_boost = boost_jaw if feat_name == "jawline" else boost_high

            for conn in conns:
                lm1 = landmarks[conn.start]
                lm2 = landmarks[conn.end]
                p1 = (float(lm1.x * w), float(lm1.y * h))
                p2 = (float(lm2.x * w), float(lm2.y * h))
                feat_pts_set.add(p1)
                feat_pts_set.add(p2)
                feat_segments.append((p1, p2))

                ip1 = (int(round(p1[0])), int(round(p1[1])))
                ip2 = (int(round(p2[0])), int(round(p2[1])))

                # Render structural feature edge onto feature_edges
                cv2.line(feature_edges, ip1, ip2, 1.0, thickness=1, lineType=cv2.LINE_AA)

                # Dilation radius for salience assignment
                dilate_radius = 4 if feat_name in ("eyes", "lips", "eyebrows") else 6
                cv2.line(feat_mask, ip1, ip2, 255, thickness=dilate_radius)

                # Guided contour angle along facial landmark tangent
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                    tangent_ang = math.atan2(dy, dx)
                    # Wrap to [-pi/2, pi/2]
                    tangent_ang = ((tangent_ang + math.pi / 2.0) % math.pi) - (math.pi / 2.0)

                    # Paint angle into guided_contour_angles ribbon
                    ribbon = np.zeros((h, w), dtype=np.uint8)
                    cv2.line(ribbon, ip1, ip2, 255, thickness=4)
                    guided_contour_angles[ribbon > 0] = float(tangent_ang)

            # Assign high priority salience to feature region
            salience_map[feat_mask > 0] = float(feat_boost)

            # Step 3: Generate candidate chords directly along feature curves
            feat_pts_list = list(feat_pts_set)
            feat_chords: List[LineSegment] = []

            # A. Add individual connection segments as candidates
            for p1, p2 in feat_segments:
                dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist >= cfg.min_length * 0.7:
                    seg = LineSegment(p1[0], p1[1], p2[0], p2[1], family_id=98)
                    feat_chords.append(seg)
                    all_feature_chords.append(seg)

            # B. Add multi-step chords along the feature (spanning 2 to 4 segments for smooth flow)
            for i in range(len(feat_segments)):
                for step in (2, 3, 4):
                    j = (i + step) % len(feat_segments)
                    p1 = feat_segments[i][0]
                    p2 = feat_segments[j][1]
                    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    if cfg.min_length <= dist <= cfg.max_length * 2.0:
                        seg = LineSegment(p1[0], p1[1], p2[0], p2[1], family_id=98)
                        feat_chords.append(seg)
                        all_feature_chords.append(seg)

            summary_counts[feat_name] = len(feat_chords)
            face_features[feat_name] = FacialFeature(
                name=feat_name,
                points=feat_pts_list,
                segments=feat_segments,
                chords=feat_chords,
                salience_boost=feat_boost,
            )

        faces_info.append(
            FaceInfo(
                bbox=(min_x, min_y, max_x, max_y),
                all_landmarks=all_pts,
                features=face_features,
            )
        )

    # Smooth the salience map slightly so priority falls off continuously
    salience_map = cv2.GaussianBlur(salience_map, (5, 5), sigmaX=1.2)
    salience_map = np.clip(salience_map, 1.0, boost_high)

    return FaceAwareData(
        has_face=True,
        num_faces=len(faces_info),
        faces=faces_info,
        salience_map=salience_map,
        feature_edges=np.clip(feature_edges, 0.0, 1.0),
        guided_contour_angles=guided_contour_angles,
        feature_chords=all_feature_chords,
        feature_summary=summary_counts,
    )
