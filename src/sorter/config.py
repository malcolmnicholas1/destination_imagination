"""Configuration loader and defaults for the shape sorter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class CameraConfig:
    source: str = "auto"
    device_index: int = 0
    width: int = 320
    height: int = 240
    fps: int = 20


@dataclass
class PreprocessingConfig:
    roi: tuple[float, float, float, float] = (0.05, 0.10, 0.95, 0.90)
    blur_kernel: int = 5
    threshold_mode: str = "otsu"
    fixed_threshold: int = 100
    invert_threshold: bool = False
    min_contour_area: int = 600
    max_contour_area: int = 45000


@dataclass
class TrackerConfig:
    tripwire_x_range: tuple[float, float] = (0.25, 0.75)
    min_votes_required: int = 3
    max_frames_gap: int = 5


@dataclass
class ClassifierConfig:
    mode: str = "hybrid"
    model_path: str = "config/shapes_model.json"
    confidence_threshold: float = 0.55


@dataclass
class ActuationConfig:
    enabled_on_pi: bool = True
    travel_delay_sec: float = 0.75
    hold_duration_sec: float = 1.0
    servo_pin: int = 18
    default_angle: float = 0.0
    shape_angles: dict[str, float] = field(
        default_factory=lambda: {
            "cube": -45.0,
            "sphere": 0.0,
            "pyramid": 45.0,
            "unknown": 0.0,
        }
    )


@dataclass
class SorterConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SorterConfig":
        path_obj = Path(path)
        if not path_obj.exists():
            return cls()
        with path_obj.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        cam_raw = raw.get("camera", {})
        prep_raw = raw.get("preprocessing", {})
        trk_raw = raw.get("tracker", {})
        clf_raw = raw.get("classifier", {})
        act_raw = raw.get("actuation", {})

        if "roi" in prep_raw:
            prep_raw["roi"] = tuple(prep_raw["roi"])
        if "tripwire_x_range" in trk_raw:
            trk_raw["tripwire_x_range"] = tuple(trk_raw["tripwire_x_range"])

        return cls(
            camera=CameraConfig(**cam_raw),
            preprocessing=PreprocessingConfig(**prep_raw),
            tracker=TrackerConfig(**trk_raw),
            classifier=ClassifierConfig(**clf_raw),
            actuation=ActuationConfig(**act_raw),
        )

