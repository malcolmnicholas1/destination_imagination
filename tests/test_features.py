"""Unit tests for preprocessing and geometric feature extraction."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sorter.camera import render_synthetic_shape_frame
from sorter.features import extract_features
from sorter.preprocessor import FramePreprocessor


def test_feature_invariance_and_separation() -> None:
    """Verify that Cube, Sphere, and Pyramid produce clearly separated geometric features."""
    prep = FramePreprocessor()
    rng = np.random.default_rng(7)

    cube_frame = render_synthetic_shape_frame("cube", size=80, angle_deg=0.0, rng=rng)
    sphere_frame = render_synthetic_shape_frame("sphere", size=80, angle_deg=0.0, rng=rng)
    pyramid_frame = render_synthetic_shape_frame("pyramid", size=80, angle_deg=0.0, rng=rng)

    cube_cnt = prep.process(cube_frame)[0][0]
    sphere_cnt = prep.process(sphere_frame)[0][0]
    pyramid_cnt = prep.process(pyramid_frame)[0][0]

    f_cube = extract_features(cube_cnt)
    f_sphere = extract_features(sphere_cnt)
    f_pyramid = extract_features(pyramid_cnt)

    # 1. Pyramid must have significantly lower extent and taper_ratio than Cube and Sphere
    assert f_pyramid.extent < 0.62
    assert f_pyramid.taper_ratio < 0.55
    assert f_pyramid.vertices == 3

    # 2. Cube must have high extent (> 0.85) and flat top (top_to_mid_ratio > 0.88)
    assert f_cube.extent > 0.85
    assert f_cube.top_to_mid_ratio > 0.88
    assert f_cube.vertices == 4

    # 3. Sphere must have high circularity (> 0.82) and >= 6 polygon vertices
    assert f_sphere.circularity > 0.82
    assert f_sphere.vertices >= 6
    assert f_sphere.top_to_mid_ratio < f_cube.top_to_mid_ratio


def test_border_clipping_detection() -> None:
    """Verify that objects partially outside the left/right ROI border are flagged."""
    prep = FramePreprocessor()
    rng = np.random.default_rng(11)

    # Render an object right at the left edge of the ROI (x=25)
    edge_frame = render_synthetic_shape_frame("cube", center_x=25, size=70, rng=rng)
    contours, _ = prep.process(edge_frame)
    assert len(contours) == 1
    assert contours[0].touches_roi_border is True

    # Render an object in the center of the frame (x=160)
    center_frame = render_synthetic_shape_frame("cube", center_x=160, size=70, rng=rng)
    contours_center, _ = prep.process(center_frame)
    assert len(contours_center) == 1
    assert contours_center[0].touches_roi_border is False

