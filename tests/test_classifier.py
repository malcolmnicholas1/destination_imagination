"""Accuracy tests for single-frame classification, calibration, and multi-frame belt tracking."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sorter.camera import SyntheticConveyorSource, render_synthetic_shape_frame
from sorter.classifier import ShapeClassifier, calibrate_from_directory
from sorter.config import ClassifierConfig
from sorter.features import extract_features
from sorter.preprocessor import FramePreprocessor
from sorter.tracker import ConveyorTracker


def test_rule_based_classifier_accuracy() -> None:
    """Verify >95% classification accuracy across 60 varied Cubes, Spheres, and Pyramids."""
    prep = FramePreprocessor()
    clf = ShapeClassifier(ClassifierConfig(mode="rules", model_path="nonexistent.json"))
    rng = np.random.default_rng(2026)

    correct = 0
    total = 0

    for shape in ("cube", "sphere", "pyramid"):
        for _ in range(20):
            size = int(rng.integers(56, 92))
            cx = int(rng.integers(100, 220))
            angle = float(rng.uniform(-6.0, 6.0))
            frame = render_synthetic_shape_frame(
                shape_name=shape,
                center_x=cx,
                size=size,
                angle_deg=angle,
                noise_sigma=4.5,
                rng=rng,
            )
            contours, _ = prep.process(frame)
            assert len(contours) >= 1
            res = clf.classify(extract_features(contours[0]))
            total += 1
            if res.label == shape:
                correct += 1

    accuracy = correct / total
    assert accuracy >= 0.95, f"Expected >=95% accuracy, got {accuracy:.2%}"


def test_calibrated_classifier_and_conveyor_tracking(tmp_path: Path) -> None:
    """
    Verify end-to-end sample calibration and multi-frame Virtual Tripwire voting
    as a sequence of 6 objects moves down the simulated conveyor belt.
    """
    samples_dir = PROJECT_ROOT / "data" / "samples"
    model_file = tmp_path / "shapes_model.json"

    payload = calibrate_from_directory(samples_dir, model_file)
    assert set(payload["classes"].keys()) == {"cube", "sphere", "pyramid"}

    expected_sequence = ["cube", "sphere", "pyramid", "cube", "pyramid", "sphere"]
    cam = SyntheticConveyorSource(
        sequence=expected_sequence,
        step_px=12,
        max_objects=len(expected_sequence),
        seed=99,
    )
    prep = FramePreprocessor()
    clf = ShapeClassifier(
        ClassifierConfig(mode="hybrid", model_path=str(model_file))
    )
    tracker = ConveyorTracker()

    decisions: list[str] = []
    while True:
        frame = cam.read()
        if frame is None:
            break
        contours, _ = prep.process(frame)
        primary = contours[0] if contours else None
        pred = clf.classify(extract_features(primary)) if primary else None
        decision = tracker.update(primary, pred, frame.shape[1])
        if decision is not None:
            decisions.append(decision.label)

    assert decisions == expected_sequence

