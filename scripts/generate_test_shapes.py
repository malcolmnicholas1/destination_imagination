#!/usr/bin/env python3
"""
Generate synthetic side-profile sample images for Cube, Sphere, and Pyramid
and create a visual inspection summary sheet.
"""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure src/ is importable when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sorter.camera import render_synthetic_shape_frame
from sorter.classifier import ShapeClassifier, calibrate_from_directory
from sorter.config import SorterConfig
from sorter.features import extract_features
from sorter.preprocessor import FramePreprocessor
from sorter.tracker import ConveyorTracker
from sorter.visualizer import draw_debug_overlay


SHAPES = ("cube", "sphere", "pyramid")


def generate_dataset(
    output_root: Path, samples_per_class: int = 25, seed: int = 12345
) -> None:
    rng = np.random.default_rng(seed)

    for shape in SHAPES:
        shape_dir = output_root / shape
        shape_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            size = int(rng.integers(58, 92))
            center_x = int(rng.integers(110, 210))
            angle_deg = float(rng.uniform(-6.0, 6.0))
            noise_sigma = float(rng.uniform(2.0, 6.0))

            frame = render_synthetic_shape_frame(
                shape_name=shape,
                width=320,
                height=240,
                center_x=center_x,
                size=size,
                angle_deg=angle_deg,
                noise_sigma=noise_sigma,
                rng=rng,
            )
            out_path = shape_dir / f"{shape}_{i:03d}.png"
            cv2.imwrite(str(out_path), frame)


def generate_preview_grid(output_path: Path, config: SorterConfig) -> None:
    """Create a 1x3 annotated HUD preview showing Cube, Sphere, and Pyramid classification."""
    preprocessor = FramePreprocessor(config.preprocessing)
    classifier = ShapeClassifier(config.classifier)
    tracker = ConveyorTracker(config.tracker)
    rng = np.random.default_rng(99)

    panels: list[np.ndarray] = []
    for shape in SHAPES:
        frame = render_synthetic_shape_frame(
            shape_name=shape,
            width=320,
            height=240,
            center_x=160,
            size=80,
            angle_deg=2.0,
            noise_sigma=3.0,
            rng=rng,
        )
        contours, _ = preprocessor.process(frame)
        primary = contours[0] if contours else None
        pred = classifier.classify(extract_features(primary)) if primary else None
        roi_bounds = preprocessor.get_roi_bounds(frame.shape)
        vis = draw_debug_overlay(
            frame=frame,
            roi_bounds=roi_bounds,
            tracker=tracker,
            contour=primary,
            prediction=pred,
            last_decision=None,
            fps=20.0,
        )
        panels.append(vis)

    grid = np.hstack(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def main() -> None:
    config = SorterConfig.from_yaml(PROJECT_ROOT / "config" / "settings.yaml")
    samples_dir = PROJECT_ROOT / "data" / "samples"

    print(f"Generating synthetic shape dataset in {samples_dir} ...")
    generate_dataset(samples_dir, samples_per_class=25)

    model_path = PROJECT_ROOT / config.classifier.model_path
    print(f"Calibrating shape model -> {model_path} ...")
    payload = calibrate_from_directory(
        samples_dir=samples_dir,
        output_model_path=model_path,
        prep_config=config.preprocessing,
    )
    for label, stats in payload["classes"].items():
        print(f"  - {label:8s}: {stats['count']} samples, mean={stats['mean']}")

    preview_path = PROJECT_ROOT / "data" / "preview_hud_grid.png"
    generate_preview_grid(preview_path, config)
    print(f"Saved annotated HUD preview grid -> {preview_path}")


if __name__ == "__main__":
    main()

