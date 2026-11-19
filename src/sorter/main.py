"""Main CLI entry point for the Destination Imagination Conveyor Belt Shape Sorter."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import platform
import time
import cv2

from .actuator import MockDisplayActuator, PiGPIOActuator, SorterActuator
from .camera import (
    CameraSource,
    ImageFolderSource,
    MacWebcamSource,
    PiCamera2Source,
    SyntheticConveyorSource,
)
from .classifier import ShapeClassifier, calibrate_from_directory
from .config import SorterConfig
from .features import extract_features
from .preprocessor import FramePreprocessor
from .tracker import ConveyorTracker
from .visualizer import draw_debug_overlay


def is_raspberry_pi() -> bool:
    """Detect whether we are running on a Raspberry Pi (ARM Linux)."""
    if platform.system() != "Linux":
        return False
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            return "raspberry pi" in model_path.read_text(errors="ignore").lower()
        except OSError:
            pass
    return False


def create_camera_source(
    config: SorterConfig, mode: str, replay_path: str | None = None
) -> CameraSource:
    if mode == "simulate" or config.camera.source == "simulate":
        return SyntheticConveyorSource(
            config=config.camera,
            sequence=["cube", "sphere", "pyramid", "cube", "pyramid", "sphere"],
            max_objects=6,
        )
    if mode == "replay" or config.camera.source == "folder":
        folder = replay_path or "data/samples"
        return ImageFolderSource(
            folder=folder, width=config.camera.width, height=config.camera.height
        )
    if config.camera.source == "picamera2" or (
        config.camera.source == "auto" and is_raspberry_pi()
    ):
        return PiCamera2Source(config.camera)
    return MacWebcamSource(config.camera)


def create_actuator(config: SorterConfig) -> SorterActuator:
    if is_raspberry_pi() and config.actuation.enabled_on_pi:
        return PiGPIOActuator(config.actuation)
    return MockDisplayActuator(config.actuation)


def run_pipeline(args: argparse.Namespace) -> int:
    config = SorterConfig.from_yaml(args.config)

    if args.mode == "calibrate":
        summary = calibrate_from_directory(
            samples_dir=args.samples_dir,
            output_model_path=config.classifier.model_path,
            prep_config=config.preprocessing,
        )
        logging.info(
            "Calibrated model saved to %s with %d shape classes: %s",
            config.classifier.model_path,
            len(summary["classes"]),
            ", ".join(
                f"{k} (n={v['count']})" for k, v in summary["classes"].items()
            ),
        )
        return 0

    camera = create_camera_source(config, args.mode, args.replay_path)
    preprocessor = FramePreprocessor(config.preprocessing)
    classifier = ShapeClassifier(config.classifier)
    tracker = ConveyorTracker(config.tracker)
    actuator = create_actuator(config)

    collect_dir: Path | None = None
    if args.mode == "collect":
        if not args.shape:
            raise ValueError("--shape <name> is required when --mode collect is used.")
        collect_dir = Path(args.samples_dir) / args.shape.lower()
        collect_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Collecting sample frames into %s", collect_dir)

    saved_count = 0
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            frame_count += 1
            elapsed = max(1e-6, time.perf_counter() - t_start)
            fps = frame_count / elapsed

            contours, _ = preprocessor.process(frame)
            primary = contours[0] if contours else None
            prediction = None

            if primary is not None:
                feats = extract_features(primary)
                prediction = classifier.classify(feats)

                if (
                    collect_dir is not None
                    and tracker.is_in_tripwire(primary, frame.shape[1])
                    and frame_count % 3 == 0
                ):
                    out_file = collect_dir / f"{args.shape.lower()}_{ int(time.time() * 1000) }_{saved_count:03d}.png"
                    cv2.imwrite(str(out_file), frame)
                    saved_count += 1
                    logging.info("Saved sample #%d -> %s", saved_count, out_file)

            decision = tracker.update(primary, prediction, frame.shape[1])
            if decision is not None:
                actuator.trigger_sort(decision)

            if not args.headless:
                roi_bounds = preprocessor.get_roi_bounds(frame.shape)
                vis = draw_debug_overlay(
                    frame=frame,
                    roi_bounds=roi_bounds,
                    tracker=tracker,
                    contour=primary,
                    prediction=prediction,
                    last_decision=tracker.last_decision,
                    fps=fps,
                )
                cv2.imshow("DI Conveyor Shape Sorter", vis)
                delay_ms = 30 if args.mode == "simulate" else 1
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        camera.close()
        actuator.close()
        if not args.headless:
            cv2.destroyAllWindows()

    logging.info(
        "Processed %d frames (%.1f FPS). Total objects sorted: %d",
        frame_count,
        frame_count / max(1e-6, time.perf_counter() - t_start),
        len(tracker.history),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Destination Imagination Conveyor Belt Shape Sorter"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "live", "calibrate", "collect", "replay"],
        default="simulate",
        help="Execution mode (default: simulate)",
    )
    parser.add_argument(
        "--samples-dir",
        default="data/samples",
        help="Root directory for shape sample images",
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="Shape name when running in --mode collect (e.g., cube, sphere, pyramid)",
    )
    parser.add_argument(
        "--replay-path",
        default=None,
        help="Image folder path when running in --mode replay",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an OpenCV GUI window (for SSH / Pi Zero / automated tests)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop automatically after N frames (0 = run until stream ends or 'q' pressed)",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())

